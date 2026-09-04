"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_hybrid_retrieval.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Unit tests for weighted Reciprocal Rank Fusion hybrid retrieval.
================================================================================
"""

from unittest.mock import MagicMock

import pytest

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.schemas.retrieval import (
    RetrievalResult,
)


def _result(
    chunk_id: str,
    score: float,
    rank: int,
    source: str,
    file_name: str,
) -> RetrievalResult:

    return RetrievalResult(
        chunk_id=chunk_id,
        content=f"Content for {chunk_id}",
        score=score,
        rank=rank,
        file_name=file_name,
        source=source,
    )


def test_hybrid_promotes_chunk_seen_by_both_retrievers() -> None:
    vector = MagicMock()
    bm25 = MagicMock()

    vector_results = [
        _result(
            "C-GUJIYA",
            0.53,
            1,
            "vector",
            "Classic_Gujiya_Product_Guide.docx",
        ),
        _result(
            "C-THEKUA",
            0.51,
            3,
            "vector",
            "Classic_Thekua_Product_Guide.docx",
        ),
    ]

    bm25_results = [
        _result(
            "C-THEKUA",
            3.5,
            1,
            "bm25",
            "Classic_Thekua_Product_Guide.docx",
        ),
        _result(
            "C-GUJIYA",
            1.0,
            4,
            "bm25",
            "Classic_Gujiya_Product_Guide.docx",
        ),
    ]

    retriever = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        vector_weight=0.55,
        bm25_weight=0.45,
        rrf_k=60,
    )

    results = retriever.fuse(
        vector_results,
        bm25_results,
        top_k=2,
    )

    assert results[0].chunk_id == (
        "C-THEKUA"
    )

    assert results[0].source == (
        "hybrid"
    )

    assert results[0].metadata[
        "vector_rank"
    ] == 3

    assert results[0].metadata[
        "bm25_rank"
    ] == 1


def test_retrieve_calls_both_retrievers() -> None:
    vector = MagicMock()
    bm25 = MagicMock()

    vector.retrieve.return_value = []
    bm25.retrieve.return_value = []

    retriever = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        candidate_pool_size=20,
        output_top_k=5,
    )

    results = retriever.retrieve(
        "Classic Thekua"
    )

    assert results == []

    vector.retrieve.assert_called_once_with(
        query="Classic Thekua",
        top_k=20,
    )

    bm25.retrieve.assert_called_once_with(
        query="Classic Thekua",
        top_k=20,
    )


def test_empty_query_returns_empty_without_retrieval() -> None:
    vector = MagicMock()
    bm25 = MagicMock()

    retriever = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
    )

    assert retriever.retrieve(
        ""
    ) == []

    vector.retrieve.assert_not_called()
    bm25.retrieve.assert_not_called()


def test_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(
            vector_retriever=MagicMock(),
            bm25_retriever=MagicMock(),
            vector_weight=0,
            bm25_weight=0,
        )


def test_weighted_rrf_does_not_mix_raw_scores() -> None:
    vector = MagicMock()
    bm25 = MagicMock()

    vector_results = [
        _result(
            "C1",
            0.99,
            2,
            "vector",
            "A.txt",
        )
    ]

    bm25_results = [
        _result(
            "C2",
            1000.0,
            2,
            "bm25",
            "B.txt",
        )
    ]

    retriever = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        vector_weight=0.5,
        bm25_weight=0.5,
        rrf_k=60,
    )

    results = retriever.fuse(
        vector_results,
        bm25_results,
    )

    # Equal ranks + equal weights => equal RRF scores despite wildly
    # different raw score scales.
    assert (
        results[0].score
        == results[1].score
    )
