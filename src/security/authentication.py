"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Security / Authentication
File         : authentication.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Local username/password authentication for OnlyNativeIQ.

Security Model:
    - Passwords are never stored in plain text.
    - Passwords are hashed with hashlib.scrypt using a unique random salt.
    - User roles are loaded from the trusted server-side user store.
    - The browser/user never supplies their own authorization role.
    - Authentication and authorization remain separate concerns.

Production Note:
    This local authentication provider is suitable for a capstone, controlled
    deployment, or small internal deployment. For enterprise production,
    replace it with SSO/OIDC/SAML/Entra ID/Google Workspace/etc. while keeping
    the same AuthenticatedUser interface.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
from pathlib import Path
from typing import Any

import yaml

from src.security.access_control import (
    UserRole,
)


# =============================================================================
# Exceptions
# =============================================================================

class AuthenticationError(RuntimeError):
    """Base authentication exception."""


class AuthenticationConfigurationError(
    AuthenticationError
):
    """Raised when the authentication user store is invalid."""


# =============================================================================
# Authenticated identity
# =============================================================================

@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """
    Trusted authenticated identity returned after credential validation.
    """

    username: str

    display_name: str

    role: UserRole

    enabled: bool = True


# =============================================================================
# Password hashing
# =============================================================================

class PasswordHasher:
    """
    Dependency-free scrypt password hashing.

    Stored values:
        salt_b64
        password_hash_b64
    """

    DEFAULT_N = 2**14
    DEFAULT_R = 8
    DEFAULT_P = 1
    KEY_LENGTH = 32
    SALT_LENGTH = 16

    @classmethod
    def hash_password(
        cls,
        password: str,
        *,
        salt: bytes | None = None,
        n: int = DEFAULT_N,
        r: int = DEFAULT_R,
        p: int = DEFAULT_P,
    ) -> dict[str, Any]:

        password = (
            password or ""
        )

        if not password:
            raise ValueError(
                "password cannot be empty."
            )

        salt = (
            salt
            if salt is not None
            else __import__(
                "secrets"
            ).token_bytes(
                cls.SALT_LENGTH
            )
        )

        derived = hashlib.scrypt(
            password.encode(
                "utf-8"
            ),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=(
                cls.KEY_LENGTH
            ),
        )

        return {
            "algorithm":
                "scrypt",

            "salt_b64":
                base64.b64encode(
                    salt
                ).decode(
                    "ascii"
                ),

            "password_hash_b64":
                base64.b64encode(
                    derived
                ).decode(
                    "ascii"
                ),

            "n":
                int(n),

            "r":
                int(r),

            "p":
                int(p),
        }

    @classmethod
    def verify_password(
        cls,
        password: str,
        record: dict[str, Any],
    ) -> bool:

        if not password:
            return False

        if (
            str(
                record.get(
                    "algorithm",
                    ""
                )
            ).lower()
            != "scrypt"
        ):
            return False

        try:

            salt = (
                base64.b64decode(
                    record[
                        "salt_b64"
                    ]
                )
            )

            expected = (
                base64.b64decode(
                    record[
                        "password_hash_b64"
                    ]
                )
            )

            actual = hashlib.scrypt(
                password.encode(
                    "utf-8"
                ),
                salt=salt,
                n=int(
                    record.get(
                        "n",
                        cls.DEFAULT_N,
                    )
                ),
                r=int(
                    record.get(
                        "r",
                        cls.DEFAULT_R,
                    )
                ),
                p=int(
                    record.get(
                        "p",
                        cls.DEFAULT_P,
                    )
                ),
                dklen=len(
                    expected
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

        return hmac.compare_digest(
            actual,
            expected,
        )


# =============================================================================
# Authentication provider
# =============================================================================

class LocalAuthenticationProvider:
    """
    Validate credentials against config/users.yaml.
    """

    def __init__(
        self,
        users_path: str | Path,
    ) -> None:

        self.users_path = Path(
            users_path
        )

        self._users = (
            self._load_users()
        )

    def authenticate(
        self,
        username: str,
        password: str,
    ) -> AuthenticatedUser | None:
        """
        Authenticate one username/password pair.

        Returns None for any invalid credential condition. This deliberately
        avoids exposing whether the username exists or the account is disabled.
        """

        username = (
            username or ""
        ).strip()

        password = (
            password or ""
        )

        if (
            not username
            or not password
        ):
            return None

        record = self._users.get(
            username.lower()
        )

        if not record:
            return None

        if not bool(
            record.get(
                "enabled",
                True,
            )
        ):
            return None

        if not PasswordHasher.verify_password(
            password,
            record,
        ):
            return None

        try:

            role = UserRole(
                str(
                    record[
                        "role"
                    ]
                ).strip().lower()
            )

        except (
            KeyError,
            ValueError,
        ) as exc:

            raise AuthenticationConfigurationError(
                f"Invalid role configuration for "
                f"user '{username}'."
            ) from exc

        return AuthenticatedUser(
            username=username,
            display_name=(
                str(
                    record.get(
                        "display_name",
                        username,
                    )
                ).strip()
                or username
            ),
            role=role,
            enabled=True,
        )

    def reload(
        self,
    ) -> None:
        """
        Reload user configuration from disk.
        """

        self._users = (
            self._load_users()
        )

    def _load_users(
        self,
    ) -> dict[str, dict[str, Any]]:

        if not self.users_path.exists():
            raise AuthenticationConfigurationError(
                f"Authentication user store not found: "
                f"{self.users_path}"
            )

        try:

            raw = yaml.safe_load(
                self.users_path.read_text(
                    encoding="utf-8"
                )
            ) or {}

        except Exception as exc:

            raise AuthenticationConfigurationError(
                f"Unable to load authentication user store: "
                f"{exc}"
            ) from exc

        users = raw.get(
            "users"
        )

        if not isinstance(
            users,
            dict,
        ):
            raise AuthenticationConfigurationError(
                "users.yaml must contain a top-level "
                "'users' mapping."
            )

        normalized: dict[
            str,
            dict[str, Any]
        ] = {}

        for username, record in (
            users.items()
        ):

            normalized_username = (
                str(
                    username
                ).strip().lower()
            )

            if not normalized_username:
                continue

            if not isinstance(
                record,
                dict,
            ):
                raise AuthenticationConfigurationError(
                    f"Invalid user record for "
                    f"'{username}'."
                )

            normalized[
                normalized_username
            ] = dict(
                record
            )

        return normalized
