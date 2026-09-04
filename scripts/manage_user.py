"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Security Administration
File         : manage_user.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Adds or updates users in config/users.yaml.

Security:
    - Password input is hidden.
    - Plain-text passwords are never written to disk.
    - Passwords are stored only as salted scrypt hashes.
================================================================================
"""

from __future__ import annotations

from getpass import getpass
from pathlib import Path
import sys

import yaml


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# OnlyNativeIQ imports
# =============================================================================

from src.security.authentication import (
    PasswordHasher,
)

from src.security.access_control import (
    UserRole,
)


# =============================================================================
# User store
# =============================================================================

USERS_PATH = (
    PROJECT_ROOT
    / "config"
    / "users.yaml"
)


def load_store() -> dict:
    """
    Load existing users.yaml.

    If it does not exist, return a fresh store.
    """

    if not USERS_PATH.exists():

        return {
            "users": {}
        }

    raw = yaml.safe_load(
        USERS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if raw is None:

        return {
            "users": {}
        }

    if not isinstance(
        raw,
        dict,
    ):

        raise ValueError(
            "users.yaml must contain a YAML mapping."
        )

    if (
        "users"
        not in raw
        or not isinstance(
            raw["users"],
            dict,
        )
    ):

        raw[
            "users"
        ] = {}

    return raw


def save_store(
    store: dict,
) -> None:
    """
    Persist user store and verify that it was written.
    """

    USERS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    yaml_text = yaml.safe_dump(
        store,
        sort_keys=False,
        allow_unicode=True,
    )

    USERS_PATH.write_text(
        yaml_text,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Verify write
    # -------------------------------------------------------------------------

    if not USERS_PATH.exists():

        raise RuntimeError(
            f"Failed to create user store: "
            f"{USERS_PATH}"
        )

    verification = yaml.safe_load(
        USERS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not verification:

        raise RuntimeError(
            "users.yaml was created but appears empty."
        )


def main() -> None:

    print(
        "=" * 90
    )

    print(
        "OnlyNativeIQ - User Administration"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"User store:"
    )

    print(
        USERS_PATH
    )

    print()

    # -------------------------------------------------------------------------
    # Username
    # -------------------------------------------------------------------------

    username = input(
        "Username: "
    ).strip().lower()

    if not username:

        raise ValueError(
            "Username cannot be empty."
        )

    # -------------------------------------------------------------------------
    # Display name
    # -------------------------------------------------------------------------

    display_name = input(
        "Display name: "
    ).strip()

    if not display_name:

        display_name = username

    # -------------------------------------------------------------------------
    # Role
    # -------------------------------------------------------------------------

    print()

    print(
        "Available roles:"
    )

    for role in UserRole:

        print(
            f"  - {role.value}"
        )

    print()

    role_value = input(
        "Role: "
    ).strip().lower()

    try:

        role = UserRole(
            role_value
        )

    except ValueError as exc:

        valid_roles = ", ".join(
            role.value
            for role in UserRole
        )

        raise ValueError(
            f"Invalid role '{role_value}'. "
            f"Valid roles: {valid_roles}"
        ) from exc

    # -------------------------------------------------------------------------
    # Password
    # -------------------------------------------------------------------------

    print()

    password = getpass(
        "Password: "
    )

    confirm_password = getpass(
        "Confirm password: "
    )

    if password != confirm_password:

        raise ValueError(
            "Passwords do not match."
        )

    if len(
        password
    ) < 3:

        raise ValueError(
            "Password must contain at least "
            "3 characters."
        )

    # -------------------------------------------------------------------------
    # Hash password
    # -------------------------------------------------------------------------

    password_record = (
        PasswordHasher.hash_password(
            password
        )
    )

    # -------------------------------------------------------------------------
    # Load existing store
    # -------------------------------------------------------------------------

    store = load_store()

    users = store.setdefault(
        "users",
        {}
    )

    # -------------------------------------------------------------------------
    # Add/update user
    # -------------------------------------------------------------------------

    users[
        username
    ] = {
        "display_name":
            display_name,

        "role":
            role.value,

        "enabled":
            True,

        **password_record,
    }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_store(
        store
    )

    # -------------------------------------------------------------------------
    # Final confirmation
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 90
    )

    print(
        "USER SAVED SUCCESSFULLY"
    )

    print(
        "=" * 90
    )

    print(
        f"Username     : {username}"
    )

    print(
        f"Display Name : {display_name}"
    )

    print(
        f"Role         : {role.value}"
    )

    print(
        f"Enabled      : True"
    )

    print()

    print(
        f"Written to:"
    )

    print(
        USERS_PATH
    )

    print()

    print(
        "Password stored as salted scrypt hash."
    )


if __name__ == "__main__":
    main()