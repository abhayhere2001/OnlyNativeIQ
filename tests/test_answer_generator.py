"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_answer_generator.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Unit tests for grounded OnlyNativeIQ answer generation.
================================================================================
"""

from unittest.mock import MagicMock

import pytest

from langchain_core.messages import (
    AIMessage,
)

from src.generation.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
)

from src.generation.prompts import (
    NO_CONTEXT_RESPONSE,
)

from src.schemas.retrieval import (
    RetrievalResult,
)


# =============================================================================
# Helper
# =============================================================================

def _result(
    content: str = (
        "Classic Thekua has a "
        "shelf life of 30 days."
    ),
) -> RetrievalResult:

    return RetrievalResult(
        chunk_id=(
            "ON-PROD-0001-S002-C001"
        ),
        document_id="ON-PROD-0001",
        content=content,
        score=5.0,
        rank=1,
        file_name=(
            "Classic_Thekua_"
            "Product_Guide.docx"
        ),
        section="Description",
        access_level="public",
        audience=[
            "customer",
            "employee",
            "production_employee",
            "manager",
        ],
        source="hybrid+reranker",
    )


# =============================================================================
# Valid grounded answer
# =============================================================================

def test_valid_evidence_invokes_llm() -> None:

    llm = MagicMock()

    llm.invoke.return_value = (
        AIMessage(
            content=(
                "Classic Thekua has "
                "a shelf life of 30 days."
            )
        )
    )

    generator = AnswerGenerator(
        llm=llm
    )

    result = generator.generate(
        question=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        user_role="customer",
        retrieval_results=[
            _result()
        ],
    )

    assert result.used_llm is True

    assert result.has_evidence is True

    assert result.evidence_count == 1

    assert (
        "30 days"
        in result.answer
    )

    llm.invoke.assert_called_once()


# =============================================================================
# No evidence = no LLM call
# =============================================================================

def test_no_evidence_does_not_invoke_llm() -> None:

    llm = MagicMock()

    generator = AnswerGenerator(
        llm=llm
    )

    result = generator.generate(
        question=(
            "How much ghee is required "
            "for Classic Thekua?"
        ),
        user_role="customer",
        retrieval_results=[],
    )

    assert (
        result.answer
        == NO_CONTEXT_RESPONSE
    )

    assert result.used_llm is False

    assert result.has_evidence is False

    assert result.citations == []

    llm.invoke.assert_not_called()


# =============================================================================
# Validation
# =============================================================================

def test_empty_question_is_rejected() -> None:

    llm = MagicMock()

    generator = AnswerGenerator(
        llm=llm
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):

        generator.generate(
            question="",
            user_role="customer",
            retrieval_results=[],
        )


def test_empty_role_is_rejected() -> None:

    llm = MagicMock()

    generator = AnswerGenerator(
        llm=llm
    )

    with pytest.raises(
        ValueError,
        match="User role cannot be empty",
    ):

        generator.generate(
            question="Question",
            user_role="",
            retrieval_results=[],
        )


# =============================================================================
# Trusted citations
# =============================================================================

def test_citations_are_built_from_grounding_metadata() -> None:

    llm = MagicMock()

    llm.invoke.return_value = (
        AIMessage(
            content="30 days."
        )
    )

    generator = AnswerGenerator(
        llm=llm
    )

    result = generator.generate(
        question="Shelf life?",
        user_role="customer",
        retrieval_results=[
            _result()
        ],
    )

    assert len(
        result.citations
    ) == 1

    citation = (
        result.citations[0]
    )

    assert (
        citation.file_name
        == (
            "Classic_Thekua_"
            "Product_Guide.docx"
        )
    )

    assert (
        citation.chunk_id
        == (
            "ON-PROD-0001-S002-C001"
        )
    )


# =============================================================================
# Sources formatting
# =============================================================================

def test_final_answer_contains_trusted_sources() -> None:

    llm = MagicMock()

    llm.invoke.return_value = (
        AIMessage(
            content="30 days."
        )
    )

    generator = AnswerGenerator(
        llm=llm,
        include_sources=True,
    )

    generated = generator.generate(
        question="Shelf life?",
        user_role="customer",
        retrieval_results=[
            _result()
        ],
    )

    final = (
        generator.format_final_answer(
            generated
        )
    )

    assert (
        "30 days."
        in final
    )

    assert (
        "Sources:"
        in final
    )

    assert (
        "Classic_Thekua_Product_Guide.docx"
        in final
    )


# =============================================================================
# LLM failure
# =============================================================================

def test_llm_failure_is_wrapped() -> None:

    llm = MagicMock()

    llm.invoke.side_effect = (
        RuntimeError(
            "Provider unavailable"
        )
    )

    generator = AnswerGenerator(
        llm=llm
    )

    with pytest.raises(
        AnswerGenerationError
    ):

        generator.generate(
            question="Shelf life?",
            user_role="customer",
            retrieval_results=[
                _result()
            ],
        )


# =============================================================================
# Empty LLM answer
# =============================================================================

def test_empty_llm_response_is_rejected() -> None:

    llm = MagicMock()

    llm.invoke.return_value = (
        AIMessage(
            content=""
        )
    )

    generator = AnswerGenerator(
        llm=llm
    )

    with pytest.raises(
        AnswerGenerationError,
        match="empty answer",
    ):

        generator.generate(
            question="Shelf life?",
            user_role="customer",
            retrieval_results=[
                _result()
            ],
        )