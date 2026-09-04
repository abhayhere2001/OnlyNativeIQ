"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Security
File         : access_control.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Defines OnlyNativeIQ role-aware access-control rules for retrieved content.

Responsibilities:
    - Represent the requesting user's application role.
    - Decide whether a retrieval result is authorized for that role.
    - Enforce both access_level and explicit audience metadata.
    - Fail closed for missing/unclassified metadata when configured.

Security Principles:
    - Public content may be returned to customers and internal users.
    - Internal content is not customer-visible.
    - Confidential content is restricted to explicitly authorized roles.
    - Audience restrictions are enforced even when access_level alone appears
      permissive.
    - Unknown/unclassified content is denied by default.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.schemas.retrieval import RetrievalResult
from src.utils.config_loader import ConfigLoader


class AccessControlError(RuntimeError):
    """Raised when security configuration is invalid."""


class UserRole(str, Enum):
    """Supported OnlyNativeIQ application roles."""

    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    PRODUCTION_EMPLOYEE = "production_employee"
    MANAGER = "manager"


@dataclass(frozen=True, slots=True)
class UserContext:
    """
    Minimal authorization context supplied to retrieval.

    Authentication/session management will provide this object later.
    """

    role: UserRole

    @classmethod
    def from_role(
        cls,
        role: str | UserRole,
    ) -> "UserContext":
        """Create UserContext from a string or UserRole."""

        if isinstance(role, UserRole):
            return cls(role=role)

        try:
            return cls(
                role=UserRole(
                    role.strip().lower()
                )
            )
        except (
            ValueError,
            AttributeError,
        ) as exc:
            raise AccessControlError(
                f"Unsupported user role '{role}'."
            ) from exc


class AccessController:
    """
    Determine whether retrieved knowledge is visible to the requesting role.
    """

    DEFAULT_ACCESS_MATRIX = {
        "public": {
            UserRole.CUSTOMER,
            UserRole.EMPLOYEE,
            UserRole.PRODUCTION_EMPLOYEE,
            UserRole.MANAGER,
        },
        "internal": {
            UserRole.EMPLOYEE,
            UserRole.PRODUCTION_EMPLOYEE,
            UserRole.MANAGER,
        },
        "confidential": {
            UserRole.PRODUCTION_EMPLOYEE,
            UserRole.MANAGER,
        },
    }

    def __init__(
        self,
        enforce_access_control: bool = True,
        deny_unclassified_documents: bool = True,
        default_access_level: str = "unclassified",
    ) -> None:
        self.enforce_access_control = (
            enforce_access_control
        )

        self.deny_unclassified_documents = (
            deny_unclassified_documents
        )

        self.default_access_level = (
            default_access_level.strip().lower()
        )

    @classmethod
    def from_config(
        cls,
        config: ConfigLoader,
    ) -> "AccessController":
        """Create AccessController from centralized configuration."""

        return cls(
            enforce_access_control=config.get(
                "security.enforce_access_control",
                default=True,
            ),
            deny_unclassified_documents=config.get(
                "security.deny_unclassified_documents",
                default=True,
            ),
            default_access_level=config.get(
                "security.default_access_level",
                default="unclassified",
            ),
        )

    def is_authorized(
        self,
        result: RetrievalResult,
        user: UserContext,
    ) -> bool:
        """
        Return True only when both access level and audience allow the user.
        """

        if not self.enforce_access_control:
            return True

        access_level = (
            result.access_level
            or self.default_access_level
        ).strip().lower()

        if access_level not in self.DEFAULT_ACCESS_MATRIX:
            return (
                not self.deny_unclassified_documents
            )

        if (
            user.role
            not in self.DEFAULT_ACCESS_MATRIX[
                access_level
            ]
        ):
            return False

        # Audience is an additional restriction.
        # If present, the user role must explicitly appear.
        if result.audience:
            normalized_audience = {
                str(role).strip().lower()
                for role in result.audience
                if str(role).strip()
            }

            if user.role.value not in normalized_audience:
                return False

        return True

    def authorize_or_raise(
        self,
        result: RetrievalResult,
        user: UserContext,
    ) -> None:
        """
        Raise AccessControlError when a result is not authorized.
        """

        if not self.is_authorized(
            result,
            user,
        ):
            raise AccessControlError(
                f"Role '{user.role.value}' is not authorized "
                f"for chunk '{result.chunk_id}'."
            )
