"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_retrieval_pipeline.py
Author       : Abhay Kumar Pandey
Updated On   : 04-Sep-2026
Description  :
    Tests retrieval -> reranking -> security -> relevance-gate orchestration.
================================================================================
"""

from unittest.mock import MagicMock

from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline,
)
from src.schemas.retrieval import (
    RetrievalResult,
)
from src.security.access_control import (
    AccessController,
    UserContext,
    UserRole,
)
from src.security.document_filter import (
    DocumentFilter,
)


def _result(
    chunk_id: str,
    rank: int,
    score: float,
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
        content=f"Content {chunk_id}",
        score=score,
        rank=rank,
        file_name="Test.txt",
        access_level=access_level,
        audience=audience,
        source="hybrid+reranker",
    )


def _pipeline(
    hybrid,
    reranker,
    minimum_relevance_score: float = 0.0,
) -> RetrievalPipeline:

    return RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        document_filter=(
            DocumentFilter(
                AccessController()
            )
        ),
        reranking_enabled=True,
        access_control_enabled=True,
        hybrid_candidate_top_k=20,
        rerank_output_top_k=20,
        final_top_k=5,
        minimum_relevance_score=(
            minimum_relevance_score
        ),
    )


def test_customer_confidential_result_is_removed_by_security() -> None:

    hybrid = MagicMock()
    reranker = MagicMock()

    confidential = _result(
        chunk_id="FORMULATION",
        rank=1,
        score=8.0,
        access_level="confidential",
        audience=[
            "production_employee",
            "manager",
        ],
    )

    public = _result(
        chunk_id="PUBLIC",
        rank=2,
        score=4.0,
    )

    hybrid.retrieve.return_value = [
        confidential,
        public,
    ]

    reranker.rerank.return_value = [
        confidential,
        public,
    ]

    output = _pipeline(
        hybrid,
        reranker,
    ).retrieve(
        query="Classic Thekua",
        user=UserContext(
            UserRole.CUSTOMER
        ),
    )

    assert [
        result.chunk_id
        for result
        in output.final_results
    ] == [
        "PUBLIC"
    ]

    assert (
        output.filtered_results_count
        == 1
    )


def test_authorized_but_negative_score_is_removed_by_relevance_gate() -> None:

    hybrid = MagicMock()
    reranker = MagicMock()

    weak_public = _result(
        chunk_id="WEAK-PUBLIC",
        rank=1,
        score=-0.35,
    )

    hybrid.retrieve.return_value = [
        weak_public
    ]

    reranker.rerank.return_value = [
        weak_public
    ]

    output = _pipeline(
        hybrid,
        reranker,
        minimum_relevance_score=0.0,
    ).retrieve(
        query=(
            "How much ghee is required?"
        ),
        user=UserContext(
            UserRole.CUSTOMER
        ),
    )

    assert (
        output.final_results
        == []
    )

    assert (
        output.filtered_results_count
        == 0
    )

    assert (
        output.relevance_filtered_results_count
        == 1
    )

    assert (
        output.authorized_results_count
        == 0
    )


def test_positive_relevant_public_result_survives_gate() -> None:

    hybrid = MagicMock()
    reranker = MagicMock()

    shelf_life = _result(
        chunk_id="SHELF-LIFE",
        rank=1,
        score=6.88,
    )

    hybrid.retrieve.return_value = [
        shelf_life
    ]

    reranker.rerank.return_value = [
        shelf_life
    ]

    output = _pipeline(
        hybrid,
        reranker,
        minimum_relevance_score=0.0,
    ).retrieve(
        query=(
            "What is the shelf life?"
        ),
        user=UserContext(
            UserRole.CUSTOMER
        ),
    )

    assert len(
        output.final_results
    ) == 1

    assert (
        output.final_results[
            0
        ].chunk_id
        == "SHELF-LIFE"
    )

    assert (
        output.relevance_filtered_results_count
        == 0
    )


def test_production_employee_can_receive_confidential_relevant_result() -> None:

    hybrid = MagicMock()
    reranker = MagicMock()

    formulation = _result(
        chunk_id="FORMULATION",
        rank=1,
        score=7.5,
        access_level="confidential",
        audience=[
            "production_employee",
            "manager",
        ],
    )

    hybrid.retrieve.return_value = [
        formulation
    ]

    reranker.rerank.return_value = [
        formulation
    ]

    output = _pipeline(
        hybrid,
        reranker,
    ).retrieve(
        query=(
            "How much ghee is required?"
        ),
        user=UserContext(
            UserRole.PRODUCTION_EMPLOYEE
        ),
    )

    assert len(
        output.final_results
    ) == 1

    assert (
        output.final_results[
            0
        ].chunk_id
        == "FORMULATION"
    )


def test_security_filter_happens_before_relevance_gate() -> None:

    hybrid = MagicMock()
    reranker = MagicMock()

    confidential = _result(
        chunk_id="CONFIDENTIAL",
        rank=1,
        score=9.0,
        access_level="confidential",
        audience=[
            "production_employee",
            "manager",
        ],
    )

    weak_public = _result(
        chunk_id="WEAK-PUBLIC",
        rank=2,
        score=-5.0,
    )

    hybrid.retrieve.return_value = [
        confidential,
        weak_public,
    ]

    reranker.rerank.return_value = [
        confidential,
        weak_public,
    ]

    output = _pipeline(
        hybrid,
        reranker,
        minimum_relevance_score=0.0,
    ).retrieve(
        query="recipe",
        user=UserContext(
            UserRole.CUSTOMER
        ),
    )

    # Confidential result removed by security.
    assert (
        output.filtered_results_count
        == 1
    )

    # Public result removed by relevance.
    assert (
        output.relevance_filtered_results_count
        == 1
    )

    assert (
        output.final_results
        == []
    )


def test_threshold_is_configurable() -> None:

    hybrid = MagicMock()
    reranker = MagicMock()

    result = _result(
        chunk_id="MEDIUM",
        rank=1,
        score=0.5,
    )

    hybrid.retrieve.return_value = [
        result
    ]

    reranker.rerank.return_value = [
        result
    ]

    output = _pipeline(
        hybrid,
        reranker,
        minimum_relevance_score=1.0,
    ).retrieve(
        query="question",
        user=UserContext(
            UserRole.CUSTOMER
        ),
    )

    assert (
        output.final_results
        == []
    )

    assert (
        output.relevance_filtered_results_count
        == 1
    )
