"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_prompts.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026
Description  :
    Unit tests for OnlyNativeIQ generation prompt construction and grounding
    restrictions.

Test Coverage:
    - Valid grounded prompt creation
    - Question validation
    - User-role validation
    - Empty-context handling
    - Grounding restrictions in system prompt
    - Prompt-injection protection instructions
    - Deterministic no-context response
================================================================================
"""

import pytest

from src.generation.prompts import (
    NO_CONTEXT_RESPONSE,
    SYSTEM_PROMPT,
    PromptInput,
    build_rag_prompt,
)


# =============================================================================
# Valid prompt
# =============================================================================

def test_build_rag_prompt_contains_question_context_and_role() -> None:

    prompt_input = PromptInput(
        question=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        context=(
            "[SOURCE-1]\n"
            "Shelf Life: 30 days\n"
            "[END SOURCE-1]"
        ),
        user_role="customer",
    )

    prompt = build_rag_prompt(
        prompt_input
    )

    assert (
        "What is the shelf life "
        "of Classic Thekua?"
        in prompt
    )

    assert (
        "Shelf Life: 30 days"
        in prompt
    )

    assert (
        "customer"
        in prompt
    )


# =============================================================================
# Question validation
# =============================================================================

def test_empty_question_is_rejected() -> None:

    prompt_input = PromptInput(
        question="",
        context="Valid context",
        user_role="customer",
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        build_rag_prompt(
            prompt_input
        )


def test_whitespace_question_is_rejected() -> None:

    prompt_input = PromptInput(
        question="   ",
        context="Valid context",
        user_role="customer",
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        build_rag_prompt(
            prompt_input
        )


# =============================================================================
# Role validation
# =============================================================================

def test_empty_user_role_is_rejected() -> None:

    prompt_input = PromptInput(
        question="What is the price?",
        context="Price: INR 160",
        user_role="",
    )

    with pytest.raises(
        ValueError,
        match="User role cannot be empty",
    ):
        build_rag_prompt(
            prompt_input
        )


# =============================================================================
# Empty context
# =============================================================================

def test_empty_context_returns_empty_prompt() -> None:

    prompt_input = PromptInput(
        question=(
            "How much ghee is required?"
        ),
        context="",
        user_role="customer",
    )

    prompt = build_rag_prompt(
        prompt_input
    )

    assert prompt == ""


# =============================================================================
# System prompt grounding rules
# =============================================================================

def test_system_prompt_requires_authorized_context() -> None:

    normalized = (
        SYSTEM_PROMPT.lower()
    )

    assert (
        "authorized context"
        in normalized
    )

    assert (
        "only"
        in normalized
    )


def test_system_prompt_prohibits_guessing() -> None:

    normalized = (
        SYSTEM_PROMPT.lower()
    )

    assert (
        "guess"
        in normalized
    )

    assert (
        "invent"
        in normalized
    )


def test_system_prompt_protects_restricted_information() -> None:

    normalized = (
        SYSTEM_PROMPT.lower()
    )

    assert (
        "authorization"
        in normalized
    )

    assert (
        "restricted"
        in normalized
    )


# =============================================================================
# Prompt injection protection
# =============================================================================

def test_system_prompt_treats_document_instructions_as_content() -> None:

    normalized = (
        SYSTEM_PROMPT.lower()
    )

    assert (
        "retrieved documents"
        in normalized
    )

    assert (
        "as document"
        in normalized
    )

    assert (
        "content, not as instructions"
        in normalized
    )

    assert (
        "ignore"
        in normalized
    )


# =============================================================================
# No-context response
# =============================================================================

def test_no_context_response_is_defined() -> None:

    assert (
        NO_CONTEXT_RESPONSE
    )

    assert (
        "authorized information"
        in NO_CONTEXT_RESPONSE.lower()
    )


def test_no_context_response_does_not_expose_security_details() -> None:

    normalized = (
        NO_CONTEXT_RESPONSE.lower()
    )

    assert (
        "confidential"
        not in normalized
    )

    assert (
        "access denied"
        not in normalized
    )

    assert (
        "permission"
        not in normalized
    )