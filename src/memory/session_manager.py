"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Memory
File         : session_manager.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Manages multiple in-memory OnlyNativeIQ conversation sessions.

Responsibilities:
    - Create conversation sessions.
    - Retrieve existing sessions.
    - Lazily create sessions when appropriate.
    - Reset a session without deleting its identity.
    - Delete sessions.
    - List active session IDs.
    - Apply common memory configuration to newly created sessions.

Important Design Principle:
    Session management is independent of authentication and authorization.
    A session NEVER determines a user's access level.

Storage:
    This implementation is process-local/in-memory.
    Persistent storage can later replace this implementation behind the same
    public API.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

from src.memory.conversation_memory import (
    ConversationMemory,
)


# =============================================================================
# Exceptions
# =============================================================================

class SessionManagerError(RuntimeError):
    """Base session-manager exception."""


class SessionNotFoundError(
    SessionManagerError
):
    """Raised when a requested session does not exist."""


class DuplicateSessionError(
    SessionManagerError
):
    """Raised when creating a session with an existing ID."""


# =============================================================================
# Session manager
# =============================================================================

@dataclass(slots=True)
class SessionManager:
    """
    Manage active OnlyNativeIQ conversation memories.

    Parameters
    ----------
    max_turns:
        Default number of individual messages retained by each new session.

    max_context_characters:
        Default memory-context character limit for each new session.
    """

    max_turns: int = 10

    max_context_characters: int = 6000

    _sessions: dict[
        str,
        ConversationMemory
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate manager defaults."""

        if self.max_turns <= 0:
            raise ValueError(
                "max_turns must be greater than zero."
            )

        if self.max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than zero."
            )

    # =========================================================================
    # Creation
    # =========================================================================

    def create_session(
        self,
        session_id: str | None = None,
    ) -> ConversationMemory:
        """
        Create and register a new conversation session.

        When session_id is omitted, a UUID is generated.
        """

        normalized_session_id = (
            session_id.strip()
            if session_id
            else str(
                uuid4()
            )
        )

        if not normalized_session_id:
            raise ValueError(
                "session_id cannot be empty."
            )

        with self._lock:

            if (
                normalized_session_id
                in self._sessions
            ):
                raise DuplicateSessionError(
                    f"Session "
                    f"'{normalized_session_id}' "
                    f"already exists."
                )

            memory = ConversationMemory(
                session_id=(
                    normalized_session_id
                ),
                max_turns=(
                    self.max_turns
                ),
                max_context_characters=(
                    self.max_context_characters
                ),
            )

            self._sessions[
                normalized_session_id
            ] = memory

            return memory

    # =========================================================================
    # Retrieval
    # =========================================================================

    def get_session(
        self,
        session_id: str,
    ) -> ConversationMemory:
        """Return an existing session or raise SessionNotFoundError."""

        normalized_session_id = (
            session_id
            or ""
        ).strip()

        if not normalized_session_id:
            raise ValueError(
                "session_id cannot be empty."
            )

        with self._lock:

            memory = self._sessions.get(
                normalized_session_id
            )

            if memory is None:
                raise SessionNotFoundError(
                    f"Session "
                    f"'{normalized_session_id}' "
                    f"was not found."
                )

            return memory

    def get_or_create_session(
        self,
        session_id: str,
    ) -> ConversationMemory:
        """
        Return an existing session, or create it if it does not exist.
        """

        normalized_session_id = (
            session_id
            or ""
        ).strip()

        if not normalized_session_id:
            raise ValueError(
                "session_id cannot be empty."
            )

        with self._lock:

            existing = self._sessions.get(
                normalized_session_id
            )

            if existing is not None:
                return existing

            memory = ConversationMemory(
                session_id=(
                    normalized_session_id
                ),
                max_turns=(
                    self.max_turns
                ),
                max_context_characters=(
                    self.max_context_characters
                ),
            )

            self._sessions[
                normalized_session_id
            ] = memory

            return memory

    # =========================================================================
    # Existence / listing
    # =========================================================================

    def has_session(
        self,
        session_id: str,
    ) -> bool:
        """Return True when a session exists."""

        normalized_session_id = (
            session_id
            or ""
        ).strip()

        if not normalized_session_id:
            return False

        with self._lock:
            return (
                normalized_session_id
                in self._sessions
            )

    def list_session_ids(
        self,
    ) -> tuple[
        str,
        ...
    ]:
        """Return active session IDs in sorted order."""

        with self._lock:
            return tuple(
                sorted(
                    self._sessions.keys()
                )
            )

    @property
    def session_count(
        self,
    ) -> int:
        """Return number of active sessions."""

        with self._lock:
            return len(
                self._sessions
            )

    # =========================================================================
    # Reset / deletion
    # =========================================================================

    def reset_session(
        self,
        session_id: str,
    ) -> ConversationMemory:
        """
        Clear a session's conversation state while keeping the session alive.
        """

        memory = self.get_session(
            session_id
        )

        memory.clear()

        return memory

    def delete_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Delete one session.

        Returns True if a session was deleted, otherwise False.
        """

        normalized_session_id = (
            session_id
            or ""
        ).strip()

        if not normalized_session_id:
            raise ValueError(
                "session_id cannot be empty."
            )

        with self._lock:
            return (
                self._sessions.pop(
                    normalized_session_id,
                    None,
                )
                is not None
            )

    def clear_all(
        self,
    ) -> None:
        """Delete all active sessions."""

        with self._lock:
            self._sessions.clear()
