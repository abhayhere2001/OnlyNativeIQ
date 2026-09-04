"""Unit tests for src.memory.conversation_memory."""

import pytest

from src.memory.conversation_memory import (
    ConversationMemory,
    ConversationRole,
    ConversationTurn,
    InvalidConversationRoleError,
)


def test_valid_memory_is_created():
    memory = ConversationMemory(session_id="session-001")
    assert memory.session_id == "session-001"
    assert memory.turn_count == 0
    assert memory.is_empty is True


@pytest.mark.parametrize("session_id", ["", "   "])
def test_empty_session_id_is_rejected(session_id):
    with pytest.raises(ValueError, match="session_id cannot be empty"):
        ConversationMemory(session_id=session_id)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="max_turns"):
        ConversationMemory(session_id="s1", max_turns=0)

    with pytest.raises(ValueError, match="max_context_characters"):
        ConversationMemory(session_id="s1", max_context_characters=0)


def test_add_user_and_assistant_messages():
    memory = ConversationMemory(session_id="s1")

    user_turn = memory.add_user_message("What is the shelf life?")
    assistant_turn = memory.add_assistant_message("30 days.")

    assert user_turn.role == ConversationRole.USER
    assert user_turn.content == "What is the shelf life?"
    assert assistant_turn.role == ConversationRole.ASSISTANT
    assert assistant_turn.content == "30 days."
    assert memory.turn_count == 2


def test_invalid_conversation_role_is_rejected():
    with pytest.raises(InvalidConversationRoleError):
        ConversationTurn.create(role="system", content="test")


def test_empty_turn_content_is_rejected():
    with pytest.raises(ValueError, match="content cannot be empty"):
        ConversationTurn.create(role="user", content="   ")


def test_history_is_trimmed_to_max_turns():
    memory = ConversationMemory(session_id="s1", max_turns=3)

    memory.add_user_message("Turn 1")
    memory.add_assistant_message("Turn 2")
    memory.add_user_message("Turn 3")
    memory.add_assistant_message("Turn 4")

    assert memory.turn_count == 3
    assert [turn.content for turn in memory.turns] == [
        "Turn 2",
        "Turn 3",
        "Turn 4",
    ]


def test_add_turns_trims_history():
    memory = ConversationMemory(session_id="s1", max_turns=2)

    memory.add_turns(
        [
            ConversationTurn.create("user", "One"),
            ConversationTurn.create("assistant", "Two"),
            ConversationTurn.create("user", "Three"),
        ]
    )

    assert [turn.content for turn in memory.turns] == ["Two", "Three"]


def test_last_referenced_entity_is_tracked_and_cleared():
    memory = ConversationMemory(session_id="s1")

    memory.set_last_referenced_entity("Classic Thekua")
    assert memory.last_referenced_entity == "Classic Thekua"

    memory.set_last_referenced_entity("   ")
    assert memory.last_referenced_entity is None

    memory.set_last_referenced_entity("Jaggery Thekua")
    memory.set_last_referenced_entity(None)
    assert memory.last_referenced_entity is None


def test_summary_management():
    memory = ConversationMemory(session_id="s1")

    memory.set_summary("Classic Thekua discussed.")
    memory.append_summary("Shelf life was requested.")

    assert "Classic Thekua discussed." in memory.summary
    assert "Shelf life was requested." in memory.summary


def test_recent_history_renders_roles():
    memory = ConversationMemory(session_id="s1")

    memory.add_user_message("What is the shelf life?")
    memory.add_assistant_message("30 days.")

    rendered = memory.render_recent_history()

    assert "User: What is the shelf life?" in rendered
    assert "Assistant: 30 days." in rendered


def test_render_recent_history_respects_requested_limit():
    memory = ConversationMemory(session_id="s1", max_turns=5)

    memory.add_user_message("One")
    memory.add_assistant_message("Two")
    memory.add_user_message("Three")

    rendered = memory.render_recent_history(max_turns=2)

    assert "One" not in rendered
    assert "Two" in rendered
    assert "Three" in rendered


def test_build_context_includes_summary_entity_and_history():
    memory = ConversationMemory(session_id="s1")

    memory.set_summary("Product discussion.")
    memory.set_last_referenced_entity("Classic Thekua")
    memory.add_user_message("What about its shelf life?")

    context = memory.build_context()

    assert "Conversation Summary:" in context
    assert "Product discussion." in context
    assert "Last Referenced Entity:" in context
    assert "Classic Thekua" in context
    assert "Recent Conversation:" in context
    assert "What about its shelf life?" in context


def test_snapshot_captures_current_state():
    memory = ConversationMemory(session_id="s1")

    memory.add_user_message("Hello")
    memory.set_summary("Greeting.")
    memory.set_last_referenced_entity("Classic Thekua")

    snapshot = memory.snapshot()

    assert snapshot.session_id == "s1"
    assert snapshot.turn_count == 1
    assert snapshot.summary == "Greeting."
    assert snapshot.last_referenced_entity == "Classic Thekua"
    assert snapshot.turns[0].content == "Hello"


def test_clear_resets_all_memory_state():
    memory = ConversationMemory(session_id="s1")

    memory.add_user_message("Hello")
    memory.set_summary("Summary")
    memory.set_last_referenced_entity("Classic Thekua")

    memory.clear()

    assert memory.turn_count == 0
    assert memory.summary == ""
    assert memory.last_referenced_entity is None
    assert memory.is_empty is True
