"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_reranker.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Unit tests for CrossEncoder reranking without model downloads.
================================================================================
"""

from unittest.mock import MagicMock

import pytest

from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankingError,
)
from src.schemas.retrieval import (
    RetrievalResult,
)


def _result(
    chunk_id: str,
    content: str,
    rank: int,
    score: float,
) -> RetrievalResult:

    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        rank=rank,
        file_name="Test.txt",
        source="hybrid",
    )


def test_reranker_reorders_candidates() -> None:
    model = MagicMock()

    # Candidate 2 should become rank 1.
    model.predict.return_value = [
        0.2,
        0.9,
        0.1,
    ]

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        input_top_k=3,
        output_top_k=2,
        batch_size=8,
        model=model,
    )

    candidates = [
        _result(
            "C1",
            "General Thekua information",
            1,
            0.01,
        ),
        _result(
            "C2",
            "Classic Thekua Ghee Moin 75 g",
            2,
            0.009,
        ),
        _result(
            "C3",
            "Gujiya Ghee Moin 100 g",
            3,
            0.008,
        ),
    ]

    results = reranker.rerank(
        query=(
            "How much ghee is required "
            "for Classic Thekua?"
        ),
        candidates=candidates,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "C2",
        "C1",
    ]

    assert results[0].rank == 1

    assert results[0].metadata[
        "pre_rerank_rank"
    ] == 2

    assert results[0].metadata[
        "reranker_score"
    ] == 0.9


def test_reranker_passes_expected_pairs_to_model() -> None:
    model = MagicMock()
    model.predict.return_value = [
        0.5
    ]

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        input_top_k=5,
        output_top_k=1,
        batch_size=4,
        model=model,
    )

    candidate = _result(
        "C1",
        "Classic Thekua Ghee 75 g",
        1,
        0.02,
    )

    reranker.rerank(
        query="Classic Thekua ghee",
        candidates=[candidate],
    )

    model.predict.assert_called_once_with(
        [
            (
                "Classic Thekua ghee",
                "Classic Thekua Ghee 75 g",
            )
        ],
        batch_size=4,
        show_progress_bar=False,
    )


def test_empty_candidates_return_empty() -> None:
    model = MagicMock()

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        model=model,
    )

    assert reranker.rerank(
        "query",
        [],
    ) == []

    model.predict.assert_not_called()


def test_score_count_mismatch_raises_error() -> None:
    model = MagicMock()
    model.predict.return_value = [
        0.5
    ]

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        input_top_k=2,
        output_top_k=2,
        model=model,
    )

    candidates = [
        _result(
            "C1",
            "one",
            1,
            0.1,
        ),
        _result(
            "C2",
            "two",
            2,
            0.2,
        ),
    ]

    with pytest.raises(
        RerankingError
    ):
        reranker.rerank(
            "query",
            candidates,
        )


def test_output_top_k_cannot_exceed_input_top_k() -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(
            input_top_k=5,
            output_top_k=6,
        )
