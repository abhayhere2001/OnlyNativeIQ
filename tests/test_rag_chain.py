"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_rag_chain.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026
Updated On   : 04-Sep-2026

Description  :
    Unit tests for OnlyNativeIQ end-to-end RAG orchestration with
    conversational memory.

Test Coverage:
    - Authorized retrieval results flow into generation.
    - Only final authorized/relevant results reach AnswerGenerator.
    - Retrieval failures are wrapped.
    - Generation failures are wrapped.
    - Empty questions are rejected.
    - Missing user context is rejected.
    - Structured response statistics are preserved.
    - Session memory is optional and backward compatible.
    - Original and resolved questions are separated correctly.
    - Follow-up queries are resolved using remembered entities.
    - Conversation turns are stored in session memory.
    - Product entity is inferred from authoritative result filenames.
    - Customer security is preserved for memory-resolved restricted queries.
================================================================================
"""

from unittest.mock import (
    ANY,
    MagicMock,
)

import pytest

from src.generation.answer_generator import (
    GeneratedAnswer,
)
from src.generation.rag_chain import (
    RAGChain,
    RAGChainError,
)
from src.memory.query_resolver import (
    QueryResolutionResult,
)
from src.memory.session_manager import (
    SessionManager,
)
from src.retrieval.retrieval_pipeline import (
    RetrievalPipelineResult,
)
from src.schemas.retrieval import (
    RetrievalResult,
)
from src.security.access_control import (
    UserContext,
    UserRole,
)


# =============================================================================
# Test helpers
# =============================================================================

def _retrieval_result(
    chunk_id: str = "C1",
    *,
    file_name: str = "Classic_Thekua_Product_Guide.docx",
    section: str = "Product Facts",
    content: str = "Classic Thekua shelf life is 30 days.",
    score: float = 5.0,
    access_level: str = "public",
    audience: list[str] | None = None,
) -> RetrievalResult:

    if audience is None:
        audience = [
            "customer",
            "employee",
            "production_employee",
            "manager",
        ]

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="ON-PROD-0001",
        content=content,
        score=score,
        rank=1,
        file_name=file_name,
        section=section,
        access_level=access_level,
        audience=audience,
        source="hybrid+reranker",
    )


def _retrieval_output(
    final_results=None,
    *,
    hybrid_candidates_count: int = 20,
    reranked_results_count: int = 10,
    filtered_results_count: int = 2,
    relevance_filtered_results_count: int = 0,
) -> RetrievalPipelineResult:

    final_results = (
        final_results
        if final_results is not None
        else [_retrieval_result()]
    )

    return RetrievalPipelineResult(
        query="What is the shelf life?",
        user_role="customer",
        hybrid_candidates_count=(
            hybrid_candidates_count
        ),
        reranked_results_count=(
            reranked_results_count
        ),
        authorized_results_count=len(
            final_results
        ),
        filtered_results_count=(
            filtered_results_count
        ),
        reranking_enabled=True,
        access_control_enabled=True,
        relevance_filtered_results_count=(
            relevance_filtered_results_count
        ),
        minimum_relevance_score=0.0,
        hybrid_results=[],
        reranked_results=[],
        final_results=final_results,
    )


def _generated_answer(
    *,
    answer: str = "Classic Thekua has a shelf life of 30 days.",
    used_llm: bool = True,
    has_evidence: bool = True,
) -> GeneratedAnswer:

    return GeneratedAnswer(
        answer=answer,
        citations=[],
        sources_text="",
        grounding_context=None,
        used_llm=used_llm,
        has_evidence=has_evidence,
        evidence_count=1 if has_evidence else 0,
    )


def _customer() -> UserContext:
    return UserContext(
        role=UserRole.CUSTOMER
    )


def _production_employee() -> UserContext:
    return UserContext(
        role=UserRole.PRODUCTION_EMPLOYEE
    )


# =============================================================================
# Existing non-conversational behavior
# =============================================================================

def test_rag_chain_passes_only_final_results_to_generator() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    authorized_result = _retrieval_result(
        "AUTHORIZED"
    )

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output(
            final_results=[
                authorized_result
            ]
        )
    )

    answer_generator.generate.return_value = (
        _generated_answer()
    )

    answer_generator.format_final_answer.return_value = (
        "Classic Thekua has a shelf life of 30 days."
    )

    chain = RAGChain(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        answer_generator=(
            answer_generator
        ),
    )

    response = chain.invoke(
        question=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        user=_customer(),
    )

    retrieval_pipeline.retrieve.assert_called_once_with(
        query=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        user=ANY,
    )

    answer_generator.generate.assert_called_once_with(
        question=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        user_role="customer",
        retrieval_results=[
            authorized_result
        ],
    )

    assert (
        response.answer
        == "Classic Thekua has a shelf life of 30 days."
    )

    assert (
        response.resolved_question
        == (
            "What is the shelf life "
            "of Classic Thekua?"
        )
    )

    assert (
        response.session_id
        is None
    )

    assert (
        response.query_was_resolved
        is False
    )


def test_zero_authorized_results_are_passed_as_empty_generation_context() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output(
            final_results=[]
        )
    )

    generated = _generated_answer(
        answer=(
            "I don't have enough authorized information "
            "in the OnlyNative knowledge base to answer "
            "that question."
        ),
        used_llm=False,
        has_evidence=False,
    )

    answer_generator.generate.return_value = (
        generated
    )

    answer_generator.format_final_answer.return_value = (
        generated.answer
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
    )

    response = chain.invoke(
        question=(
            "How much ghee is required "
            "for Classic Thekua?"
        ),
        user=_customer(),
    )

    answer_generator.generate.assert_called_once_with(
        question=(
            "How much ghee is required "
            "for Classic Thekua?"
        ),
        user_role="customer",
        retrieval_results=[],
    )

    assert (
        response.used_llm
        is False
    )

    assert (
        response.has_evidence
        is False
    )

    assert (
        response.authorized_results
        == 0
    )


# =============================================================================
# Error handling
# =============================================================================

def test_retrieval_failure_is_wrapped() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    retrieval_pipeline.retrieve.side_effect = (
        RuntimeError(
            "retrieval failed"
        )
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
    )

    with pytest.raises(
        RAGChainError,
        match="retrieval failed",
    ):

        chain.invoke(
            question="Test question",
            user=_customer(),
        )


def test_generation_failure_is_wrapped() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output()
    )

    answer_generator.generate.side_effect = (
        RuntimeError(
            "generation failed"
        )
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
    )

    with pytest.raises(
        RAGChainError,
        match="generation failed",
    ):

        chain.invoke(
            question="Test question",
            user=_customer(),
        )


def test_query_resolution_failure_is_wrapped() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()
    query_resolver = MagicMock()

    query_resolver.resolve.side_effect = (
        RuntimeError(
            "resolver failed"
        )
    )

    chain = RAGChain(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        answer_generator=(
            answer_generator
        ),
        query_resolver=(
            query_resolver
        ),
    )

    with pytest.raises(
        RAGChainError,
        match="resolver failed",
    ):

        chain.invoke(
            question="Shelf life?",
            user=_customer(),
            session_id="s1",
        )


# =============================================================================
# Input validation
# =============================================================================

def test_empty_question_is_rejected() -> None:

    chain = RAGChain(
        MagicMock(),
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):

        chain.invoke(
            question="",
            user=_customer(),
        )


def test_none_user_is_rejected() -> None:

    chain = RAGChain(
        MagicMock(),
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="user cannot be None",
    ):

        chain.invoke(
            question="Question",
            user=None,
        )


def test_blank_session_id_is_rejected() -> None:

    chain = RAGChain(
        MagicMock(),
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="session_id cannot be empty",
    ):

        chain.invoke(
            question="Question",
            user=_customer(),
            session_id="   ",
        )


# =============================================================================
# Retrieval diagnostics
# =============================================================================

def test_response_contains_retrieval_statistics() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output(
            relevance_filtered_results_count=3
        )
    )

    answer_generator.generate.return_value = (
        _generated_answer()
    )

    answer_generator.format_final_answer.return_value = (
        "Answer"
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
    )

    response = chain.invoke(
        question="Question",
        user=_customer(),
    )

    assert (
        response.retrieval_candidates
        == 20
    )

    assert (
        response.reranked_candidates
        == 10
    )

    assert (
        response.security_filtered
        == 2
    )

    assert (
        response.relevance_filtered
        == 3
    )

    assert (
        response.authorized_results
        == 1
    )

    assert (
        response.evidence_count
        == 1
    )


# =============================================================================
# Session memory integration
# =============================================================================

def test_session_is_created_when_session_id_is_supplied() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output()
    )

    answer_generator.generate.return_value = (
        _generated_answer()
    )

    answer_generator.format_final_answer.return_value = (
        "Answer"
    )

    manager = SessionManager()

    chain = RAGChain(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        answer_generator=(
            answer_generator
        ),
        session_manager=manager,
    )

    response = chain.invoke(
        question=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        user=_customer(),
        session_id="session-001",
    )

    assert (
        response.session_id
        == "session-001"
    )

    assert (
        manager.has_session(
            "session-001"
        )
        is True
    )


def test_original_question_and_answer_are_stored_in_memory() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output()
    )

    answer_generator.generate.return_value = (
        _generated_answer(
            answer="30 days."
        )
    )

    answer_generator.format_final_answer.return_value = (
        "30 days."
    )

    manager = SessionManager()

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
        session_manager=manager,
    )

    response = chain.invoke(
        question=(
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        user=_customer(),
        session_id="s1",
    )

    memory = manager.get_session(
        "s1"
    )

    assert [
        turn.content
        for turn in memory.turns
    ] == [
        (
            "What is the shelf life "
            "of Classic Thekua?"
        ),
        "30 days.",
    ]

    assert (
        response.memory_turn_count
        == 2
    )


def test_entity_is_inferred_from_product_result_filename() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    result = _retrieval_result(
        file_name=(
            "Jaggery_Thekua_Product_Guide.docx"
        ),
        content=(
            "Shelf Life | 30 days"
        ),
    )

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output(
            final_results=[
                result
            ]
        )
    )

    answer_generator.generate.return_value = (
        _generated_answer(
            answer="30 days."
        )
    )

    answer_generator.format_final_answer.return_value = (
        "30 days."
    )

    manager = SessionManager()

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
        session_manager=manager,
    )

    response = chain.invoke(
        question=(
            "What is the shelf life "
            "of Jaggery Thekua?"
        ),
        user=_customer(),
        session_id="s1",
    )

    memory = manager.get_session(
        "s1"
    )

    assert (
        memory.last_referenced_entity
        == "Jaggery Thekua"
    )

    assert (
        response.remembered_entity
        == "Jaggery Thekua"
    )


def test_non_product_result_does_not_replace_remembered_entity() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    manager = SessionManager()
    memory = manager.create_session(
        "s1"
    )

    memory.set_last_referenced_entity(
        "Classic Thekua"
    )

    result = _retrieval_result(
        file_name="Company_FAQ.docx",
        section="Support",
        content="Support is available through WhatsApp.",
    )

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output(
            final_results=[
                result
            ]
        )
    )

    answer_generator.generate.return_value = (
        _generated_answer(
            answer="Through WhatsApp."
        )
    )

    answer_generator.format_final_answer.return_value = (
        "Through WhatsApp."
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
        session_manager=manager,
    )

    chain.invoke(
        question=(
            "How can customers contact support?"
        ),
        user=_customer(),
        session_id="s1",
    )

    assert (
        memory.last_referenced_entity
        == "Classic Thekua"
    )


# =============================================================================
# Query resolution integration
# =============================================================================

def test_resolved_question_is_used_for_retrieval_and_generation() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()
    query_resolver = MagicMock()

    manager = SessionManager()
    memory = manager.create_session(
        "s1"
    )

    memory.set_last_referenced_entity(
        "Classic Thekua"
    )

    query_resolver.resolve.return_value = (
        QueryResolutionResult(
            original_query="Shelf life?",
            resolved_query=(
                "Shelf life of Classic Thekua?"
            ),
            was_resolved=True,
            referenced_entity=(
                "Classic Thekua"
            ),
            reason=(
                "Resolved using memory."
            ),
        )
    )

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output()
    )

    answer_generator.generate.return_value = (
        _generated_answer(
            answer="30 days."
        )
    )

    answer_generator.format_final_answer.return_value = (
        "30 days."
    )

    chain = RAGChain(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        answer_generator=(
            answer_generator
        ),
        session_manager=manager,
        query_resolver=query_resolver,
    )

    response = chain.invoke(
        question="Shelf life?",
        user=_customer(),
        session_id="s1",
    )

    retrieval_pipeline.retrieve.assert_called_once_with(
        query=(
            "Shelf life of Classic Thekua?"
        ),
        user=ANY,
    )

    answer_generator.generate.assert_called_once_with(
        question=(
            "Shelf life of Classic Thekua?"
        ),
        user_role="customer",
        retrieval_results=(
            retrieval_pipeline
            .retrieve
            .return_value
            .final_results
        ),
    )

    assert (
        response.question
        == "Shelf life?"
    )

    assert (
        response.resolved_question
        == (
            "Shelf life of Classic Thekua?"
        )
    )

    assert (
        response.query_was_resolved
        is True
    )


def test_real_query_resolver_handles_follow_up_with_remembered_entity() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    manager = SessionManager()
    memory = manager.create_session(
        "s1"
    )

    memory.set_last_referenced_entity(
        "Classic Thekua"
    )

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output()
    )

    answer_generator.generate.return_value = (
        _generated_answer(
            answer="30 days."
        )
    )

    answer_generator.format_final_answer.return_value = (
        "30 days."
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
        session_manager=manager,
    )

    response = chain.invoke(
        question="Shelf life?",
        user=_customer(),
        session_id="s1",
    )

    assert (
        response.resolved_question
        == "Shelf life of Classic Thekua?"
    )

    assert (
        response.query_was_resolved
        is True
    )


# =============================================================================
# Memory + security regression
# =============================================================================

def test_memory_resolved_restricted_query_still_receives_only_final_results() -> None:
    """
    Memory may resolve:
        How much ghee does it use?
    into:
        How much ghee does Classic Thekua use?

    But the RAG chain must still pass ONLY retrieval_output.final_results
    to AnswerGenerator. For a customer, that may be empty after security.
    """

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    manager = SessionManager()
    memory = manager.create_session(
        "customer-session"
    )

    memory.set_last_referenced_entity(
        "Classic Thekua"
    )

    # Security/relevance pipeline returns nothing for the customer.
    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output(
            final_results=[]
        )
    )

    no_context = _generated_answer(
        answer=(
            "I don't have enough authorized information "
            "in the OnlyNative knowledge base to answer "
            "that question."
        ),
        used_llm=False,
        has_evidence=False,
    )

    answer_generator.generate.return_value = (
        no_context
    )

    answer_generator.format_final_answer.return_value = (
        no_context.answer
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
        session_manager=manager,
    )

    response = chain.invoke(
        question=(
            "How much ghee does it use?"
        ),
        user=_customer(),
        session_id="customer-session",
    )

    assert (
        "Classic Thekua"
        in response.resolved_question
    )

    answer_generator.generate.assert_called_once_with(
        question=(
            response.resolved_question
        ),
        user_role="customer",
        retrieval_results=[],
    )

    assert (
        response.authorized_results
        == 0
    )

    assert (
        response.used_llm
        is False
    )

    assert (
        response.has_evidence
        is False
    )


def test_production_employee_memory_resolved_query_can_receive_authorized_result() -> None:

    retrieval_pipeline = MagicMock()
    answer_generator = MagicMock()

    manager = SessionManager()
    memory = manager.create_session(
        "prod-session"
    )

    memory.set_last_referenced_entity(
        "Classic Thekua"
    )

    confidential_result = (
        _retrieval_result(
            chunk_id="FORMULATION",
            section=(
                "Standard Batch Formulation"
            ),
            content=(
                "Ghee (Moin) | 75 | 15.0%"
            ),
            score=8.0,
            access_level="confidential",
            audience=[
                "production_employee",
                "manager",
            ],
        )
    )

    retrieval_pipeline.retrieve.return_value = (
        _retrieval_output(
            final_results=[
                confidential_result
            ]
        )
    )

    answer_generator.generate.return_value = (
        _generated_answer(
            answer="75 g of ghee is required."
        )
    )

    answer_generator.format_final_answer.return_value = (
        "75 g of ghee is required."
    )

    chain = RAGChain(
        retrieval_pipeline,
        answer_generator,
        session_manager=manager,
    )

    response = chain.invoke(
        question=(
            "How much ghee does it use?"
        ),
        user=_production_employee(),
        session_id="prod-session",
    )

    assert (
        "Classic Thekua"
        in response.resolved_question
    )

    assert (
        response.authorized_results
        == 1
    )

    assert (
        response.used_llm
        is True
    )


# =============================================================================
# Entity filename helper
# =============================================================================

@pytest.mark.parametrize(
    "file_name,expected",
    [
        (
            "Classic_Thekua_Product_Guide.docx",
            "Classic Thekua",
        ),
        (
            "Jaggery_Thekua_Product_Guide.docx",
            "Jaggery Thekua",
        ),
        (
            "Garlic_Blended_Nimki_Product_Guide.docx",
            "Garlic Blended Nimki",
        ),
        (
            "Company_FAQ.docx",
            None,
        ),
        (
            "",
            None,
        ),
    ],
)
def test_product_entity_from_file_name(
    file_name,
    expected,
) -> None:

    assert (
        RAGChain._product_entity_from_file_name(
            file_name
        )
        == expected
    )
