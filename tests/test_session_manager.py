"""Unit tests for src.memory.session_manager."""

import pytest

from src.memory.session_manager import (
    DuplicateSessionError,
    SessionManager,
    SessionNotFoundError,
)


def test_manager_is_created_empty():
    manager = SessionManager()

    assert manager.session_count == 0
    assert manager.list_session_ids() == ()


def test_invalid_manager_configuration_is_rejected():
    with pytest.raises(ValueError, match="max_turns"):
        SessionManager(max_turns=0)

    with pytest.raises(ValueError, match="max_context_characters"):
        SessionManager(max_context_characters=0)


def test_create_named_session():
    manager = SessionManager()

    memory = manager.create_session("session-001")

    assert memory.session_id == "session-001"
    assert manager.session_count == 1
    assert manager.has_session("session-001") is True


def test_create_session_without_id_generates_id():
    manager = SessionManager()

    memory = manager.create_session()

    assert memory.session_id
    assert manager.has_session(memory.session_id) is True


def test_duplicate_session_is_rejected():
    manager = SessionManager()
    manager.create_session("same-session")

    with pytest.raises(DuplicateSessionError):
        manager.create_session("same-session")


def test_get_existing_session_returns_same_memory():
    manager = SessionManager()

    created = manager.create_session("s1")
    fetched = manager.get_session("s1")

    assert fetched is created


def test_get_missing_session_raises():
    manager = SessionManager()

    with pytest.raises(SessionNotFoundError):
        manager.get_session("missing")


def test_get_or_create_returns_existing_session():
    manager = SessionManager()

    created = manager.create_session("s1")
    fetched = manager.get_or_create_session("s1")

    assert fetched is created
    assert manager.session_count == 1


def test_get_or_create_creates_missing_session():
    manager = SessionManager()

    memory = manager.get_or_create_session("new-session")

    assert memory.session_id == "new-session"
    assert manager.session_count == 1


def test_sessions_keep_independent_memory():
    manager = SessionManager()

    first = manager.create_session("first")
    second = manager.create_session("second")

    first.add_user_message("Classic Thekua")
    first.set_last_referenced_entity("Classic Thekua")

    second.add_user_message("Classic Gujiya")
    second.set_last_referenced_entity("Classic Gujiya")

    assert first.turns[0].content == "Classic Thekua"
    assert second.turns[0].content == "Classic Gujiya"
    assert first.last_referenced_entity == "Classic Thekua"
    assert second.last_referenced_entity == "Classic Gujiya"


def test_new_sessions_receive_manager_defaults():
    manager = SessionManager(
        max_turns=6,
        max_context_characters=3000,
    )

    memory = manager.create_session("s1")

    assert memory.max_turns == 6
    assert memory.max_context_characters == 3000


def test_list_session_ids_is_sorted():
    manager = SessionManager()

    manager.create_session("z-session")
    manager.create_session("a-session")
    manager.create_session("m-session")

    assert manager.list_session_ids() == (
        "a-session",
        "m-session",
        "z-session",
    )


def test_has_session_returns_false_for_missing_session():
    manager = SessionManager()

    assert manager.has_session("missing") is False
    assert manager.has_session("") is False


def test_reset_session_clears_memory_but_keeps_session():
    manager = SessionManager()

    memory = manager.create_session("s1")
    memory.add_user_message("Hello")
    memory.set_summary("Summary")
    memory.set_last_referenced_entity("Classic Thekua")

    reset_memory = manager.reset_session("s1")

    assert reset_memory is memory
    assert manager.has_session("s1") is True
    assert memory.is_empty is True


def test_delete_existing_session_returns_true():
    manager = SessionManager()

    manager.create_session("s1")

    assert manager.delete_session("s1") is True
    assert manager.has_session("s1") is False
    assert manager.session_count == 0


def test_delete_missing_session_returns_false():
    manager = SessionManager()

    assert manager.delete_session("missing") is False


def test_clear_all_deletes_all_sessions():
    manager = SessionManager()

    manager.create_session("s1")
    manager.create_session("s2")

    manager.clear_all()

    assert manager.session_count == 0
    assert manager.list_session_ids() == ()
