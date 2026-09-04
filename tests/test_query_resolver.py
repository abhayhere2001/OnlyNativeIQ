"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_query_resolver.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Unit tests for conversational follow-up query resolution.

Test Coverage:
    - Empty query validation
    - No-memory behavior
    - No remembered entity behavior
    - Explicit-entity queries remain unchanged
    - Short follow-up queries
    - Pronoun/reference resolution
    - "What about Jaggery one?" style resolution
    - Non-follow-up queries remain unchanged
    - Resolution never changes authorization itself
================================================================================
"""

import pytest

from src.memory.conversation_memory import (
    ConversationMemory,
)

from src.memory.query_resolver import (
    QueryResolver,
)


def _memory(
    entity: str | None = "Classic Thekua",
) -> ConversationMemory:

    memory = ConversationMemory(
        session_id="test-session"
    )

    if entity:
        memory.set_last_referenced_entity(
            entity
        )

    return memory


# =============================================================================
# Validation
# =============================================================================

def test_empty_query_is_rejected() -> None:

    resolver = QueryResolver()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        resolver.resolve(
            query="   ",
            memory=_memory(),
        )


# =============================================================================
# No memory / no entity
# =============================================================================

def test_no_memory_returns_original_query() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query="What is the shelf life?",
        memory=None,
    )

    assert (
        result.resolved_query
        == "What is the shelf life?"
    )

    assert (
        result.was_resolved
        is False
    )

    assert (
        result.referenced_entity
        is None
    )


def test_memory_without_entity_returns_original_query() -> None:

    resolver = QueryResolver()

    memory = _memory(
        entity=None
    )

    result = resolver.resolve(
        query="Shelf life?",
        memory=memory,
    )

    assert (
        result.resolved_query
        == "Shelf life?"
    )

    assert (
        result.was_resolved
        is False
    )


# =============================================================================
# Explicit entity already present
# =============================================================================

def test_query_with_explicit_remembered_entity_is_not_rewritten() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        memory=_memory(),
    )

    assert (
        result.resolved_query
        == (
            "What is the shelf life "
            "of Classic Thekua?"
        )
    )

    assert (
        result.was_resolved
        is False
    )


# =============================================================================
# Short follow-up queries
# =============================================================================

@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "Shelf life?",
            "Shelf life of Classic Thekua?",
        ),
        (
            "Price?",
            "Price of Classic Thekua?",
        ),
        (
            "Storage?",
            "Storage of Classic Thekua?",
        ),
        (
            "Allergens?",
            "Allergens of Classic Thekua?",
        ),
    ],
)
def test_short_follow_up_uses_last_entity(
    query,
    expected,
) -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query=query,
        memory=_memory(),
    )

    assert (
        result.resolved_query
        == expected
    )

    assert (
        result.was_resolved
        is True
    )

    assert (
        result.referenced_entity
        == "Classic Thekua"
    )


# =============================================================================
# Pronoun resolution
# =============================================================================

def test_it_reference_is_resolved() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query=(
            "How much ghee does it use?"
        ),
        memory=_memory(),
    )

    assert (
        "Classic Thekua"
        in result.resolved_query
    )

    assert (
        "it"
        not in result.resolved_query.lower()
    )

    assert (
        result.was_resolved
        is True
    )


def test_its_reference_is_resolved() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query=(
            "What is its price?"
        ),
        memory=_memory(),
    )

    assert (
        "Classic Thekua"
        in result.resolved_query
    )

    assert (
        result.was_resolved
        is True
    )


def test_this_one_reference_is_resolved() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query=(
            "What is the shelf life "
            "of this one?"
        ),
        memory=_memory(),
    )

    assert (
        "Classic Thekua"
        in result.resolved_query
    )

    assert (
        result.was_resolved
        is True
    )


# =============================================================================
# Product-family follow-up
# =============================================================================

def test_what_about_jaggery_one_uses_product_family() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query=(
            "What about Jaggery one?"
        ),
        memory=_memory(
            entity="Classic Thekua"
        ),
    )

    assert (
        result.resolved_query
        == "What about Jaggery Thekua?"
    )

    assert (
        result.was_resolved
        is True
    )


def test_how_about_garlic_one_uses_product_family() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query=(
            "How about Garlic one?"
        ),
        memory=_memory(
            entity="Classic Nimki"
        ),
    )

    assert (
        result.resolved_query
        == "How about Garlic Nimki?"
    )

    assert (
        result.was_resolved
        is True
    )


# =============================================================================
# Ordinary standalone query
# =============================================================================

def test_non_follow_up_query_remains_unchanged() -> None:

    resolver = QueryResolver()

    query = (
        "Tell me about OnlyNative's "
        "shipping policy."
    )

    result = resolver.resolve(
        query=query,
        memory=_memory(),
    )

    assert (
        result.resolved_query
        == query
    )

    assert (
        result.was_resolved
        is False
    )


# =============================================================================
# Diagnostic fields
# =============================================================================

def test_resolution_result_preserves_original_query() -> None:

    resolver = QueryResolver()

    result = resolver.resolve(
        query="Price?",
        memory=_memory(),
    )

    assert (
        result.original_query
        == "Price?"
    )

    assert (
        result.referenced_entity
        == "Classic Thekua"
    )

    assert (
        "last referenced entity"
        in result.reason.lower()
    )


# =============================================================================
# Security-design regression
# =============================================================================

def test_resolver_only_rewrites_query_and_does_not_store_restricted_fact() -> None:
    """
    This test documents the security boundary.

    QueryResolver may resolve:
        "How much ghee does it use?"
    to an explicit Classic Thekua query.

    It does NOT provide the answer and does NOT modify access permissions.
    """

    resolver = QueryResolver()

    memory = _memory(
        entity="Classic Thekua"
    )

    result = resolver.resolve(
        query=(
            "How much ghee does it use?"
        ),
        memory=memory,
    )

    assert (
        "Classic Thekua"
        in result.resolved_query
    )

    # No restricted formulation fact is introduced by the resolver.
    assert (
        "75"
        not in result.resolved_query
    )

    assert (
        memory.last_referenced_entity
        == "Classic Thekua"
    )