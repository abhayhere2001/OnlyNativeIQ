"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_conversation_summarizer.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Unit tests for the security-conscious hybrid-memory conversation summarizer.

Test Coverage:
    - Empty input behavior
    - Existing-summary preservation
    - Product/entity extraction from user turns
    - Topic extraction
    - Assistant factual answers are not copied into summary memory
    - Last referenced entity preservation
    - Summary sanitization for numeric facts
    - Deduplication
    - Multiple products/topics
    - Security boundary for restricted quantities
================================================================================
"""

from src.memory.conversation_memory import (
    ConversationTurn,
)

from src.memory.conversation_summarizer import (
    ConversationSummarizer,
)


def _user(
    text: str,
) -> ConversationTurn:
    return ConversationTurn.create(
        role="user",
        content=text,
    )


def _assistant(
    text: str,
) -> ConversationTurn:
    return ConversationTurn.create(
        role="assistant",
        content=text,
    )


# =============================================================================
# Empty / existing summary
# =============================================================================

def test_empty_turns_return_existing_summary() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[],
        existing_summary=(
            "Products/entities discussed: Classic Thekua."
        ),
        last_referenced_entity="Classic Thekua",
    )

    assert (
        result.summary
        == "Products/entities discussed: Classic Thekua."
    )

    assert (
        result.turns_processed
        == 0
    )

    assert (
        result.used_existing_summary
        is True
    )

    assert (
        result.entities
        == ("Classic Thekua",)
    )


def test_empty_turns_without_existing_summary_returns_empty_summary() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[],
    )

    assert (
        result.summary
        == ""
    )

    assert (
        result.turns_processed
        == 0
    )

    assert (
        result.used_existing_summary
        is False
    )


# =============================================================================
# Entity extraction
# =============================================================================

def test_extracts_classic_thekua_from_user_turn() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "What is the shelf life of Classic Thekua?"
            )
        ],
    )

    assert (
        "Classic Thekua"
        in result.entities
    )

    assert (
        "Classic Thekua"
        in result.summary
    )


def test_extracts_multiple_products() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "Tell me about Classic Thekua."
            ),
            _user(
                "What about Jaggery Thekua?"
            ),
        ],
    )

    assert (
        "Classic Thekua"
        in result.entities
    )

    assert (
        "Jaggery Thekua"
        in result.entities
    )


def test_last_referenced_entity_is_preserved() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "What is the price?"
            )
        ],
        last_referenced_entity=(
            "Classic Thekua"
        ),
    )

    assert (
        "Classic Thekua"
        in result.entities
    )

    assert (
        "Current conversational focus: Classic Thekua."
        in result.summary
    )


# =============================================================================
# Topic extraction
# =============================================================================

def test_detects_shelf_life_topic() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "What is the shelf life of Classic Thekua?"
            )
        ],
    )

    assert (
        "shelf life"
        in result.topics
    )


def test_detects_multiple_topics() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "What is the price of Classic Thekua?"
            ),
            _user(
                "How should I store it?"
            ),
            _user(
                "Does it contain gluten?"
            ),
        ],
        last_referenced_entity=(
            "Classic Thekua"
        ),
    )

    assert (
        "price"
        in result.topics
    )

    assert (
        "storage"
        in result.topics
    )

    assert (
        "allergens"
        in result.topics
    )


def test_detects_formulation_topic_without_copying_quantity() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "How much ghee is required for one batch of Classic Thekua?"
            ),
            _assistant(
                "75 g of ghee is required."
            ),
        ],
    )

    assert (
        "ingredients/formulation"
        in result.topics
    )

    assert (
        "75"
        not in result.summary
    )

    assert (
        "75 g"
        not in result.summary
    )


def test_detects_preparation_process_topic() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "Tell me the preparation process for Classic Thekua."
            )
        ],
    )

    assert (
        "preparation process"
        in result.topics
    )


# =============================================================================
# Security behavior
# =============================================================================

def test_assistant_factual_answer_is_not_copied_into_summary() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "What is the shelf life of Classic Thekua?"
            ),
            _assistant(
                "The shelf life of Classic Thekua is 30 days."
            ),
        ],
        last_referenced_entity=(
            "Classic Thekua"
        ),
    )

    assert (
        "30 days"
        not in result.summary
    )

    assert (
        "30"
        not in result.summary
    )


def test_restricted_formulation_answer_is_not_stored() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "How much ghee does Classic Thekua use?"
            ),
            _assistant(
                "Classic Thekua uses 75 g ghee, which is 15.0% of base flour."
            ),
        ],
        last_referenced_entity=(
            "Classic Thekua"
        ),
    )

    assert (
        "75 g"
        not in result.summary
    )

    assert (
        "15.0%"
        not in result.summary
    )

    assert (
        "ingredients/formulation"
        in result.summary
    )


def test_sanitize_summary_removes_numeric_facts() -> None:

    sanitized = (
        ConversationSummarizer
        .sanitize_summary(
            "Classic Thekua has 30 days shelf life, "
            "costs INR 160, uses 75 g ghee and 15.0% moin."
        )
    )

    assert (
        "30 days"
        not in sanitized
    )

    assert (
        "INR 160"
        not in sanitized
    )

    assert (
        "75 g"
        not in sanitized
    )

    assert (
        "15.0%"
        not in sanitized
    )

    assert (
        sanitized.count(
            "[fact omitted]"
        )
        >= 4
    )


# =============================================================================
# Existing-summary handling
# =============================================================================

def test_existing_summary_is_preserved_and_extended() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "How should I store Classic Thekua?"
            )
        ],
        existing_summary=(
            "Products/entities discussed: Classic Thekua."
        ),
        last_referenced_entity=(
            "Classic Thekua"
        ),
    )

    assert (
        result.used_existing_summary
        is True
    )

    assert (
        "Products/entities discussed: Classic Thekua."
        in result.summary
    )

    assert (
        "storage"
        in result.summary
    )


def test_duplicate_summary_lines_are_removed() -> None:

    summarizer = ConversationSummarizer()

    result = summarizer.summarize(
        turns=[
            _user(
                "What is the shelf life of Classic Thekua?"
            )
        ],
        existing_summary=(
            "Products/entities discussed: Classic Thekua.\n"
            "Products/entities discussed: Classic Thekua."
        ),
        last_referenced_entity=(
            "Classic Thekua"
        ),
    )

    lines = [
        line
        for line
        in result.summary.splitlines()
        if line.strip()
    ]

    assert (
        lines.count(
            "Products/entities discussed: Classic Thekua."
        )
        == 1
    )


# =============================================================================
# Diagnostics
# =============================================================================

def test_turn_count_is_reported() -> None:

    summarizer = ConversationSummarizer()

    turns = [
        _user(
            "What is the shelf life of Classic Thekua?"
        ),
        _assistant(
            "30 days."
        ),
        _user(
            "How should I store it?"
        ),
    ]

    result = summarizer.summarize(
        turns=turns,
        last_referenced_entity=(
            "Classic Thekua"
        ),
    )

    assert (
        result.turns_processed
        == 3
    )
