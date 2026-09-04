"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_vector_retrieval.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Unit tests for semantic retrieval orchestration.
================================================================================
"""

from unittest.mock import MagicMock

from src.retrieval.vector_retriever import (
    VectorRetriever,
)
from src.schemas.retrieval import (
    RetrievalResult,
)


def test_retriever_embeds_query_and_searches_store() -> None:
    embedder = MagicMock()
    vector_store = MagicMock()

    embedder.embed_query.return_value = [
        0.1,
        0.2,
    ]

    vector_store.search.return_value = [
        RetrievalResult(
            chunk_id="C001",
            content="Ghee | 75 g",
            score=0.91,
            rank=1,
            document_id="ON-PROD-0001",
            file_name=(
                "Classic_Thekua_Product_Guide.docx"
            ),
            access_level="confidential",
        )
    ]

    retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=5,
    )

    results = retriever.retrieve(
        "How much ghee is required?"
    )

    embedder.embed_query.assert_called_once_with(
        "How much ghee is required?"
    )

    vector_store.search.assert_called_once_with(
        query_embedding=[
            0.1,
            0.2,
        ],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "C001"


def test_score_threshold_filters_results() -> None:
    embedder = MagicMock()
    vector_store = MagicMock()

    embedder.embed_query.return_value = [
        0.1,
        0.2,
    ]

    vector_store.search.return_value = [
        RetrievalResult(
            chunk_id="C001",
            content="Strong",
            score=0.85,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="C002",
            content="Weak",
            score=0.20,
            rank=2,
        ),
    ]

    retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=5,
        score_threshold=0.50,
    )

    results = retriever.retrieve(
        "query"
    )

    assert [
        item.chunk_id
        for item in results
    ] == [
        "C001"
    ]

    assert results[0].rank == 1


def test_empty_query_returns_empty() -> None:
    embedder = MagicMock()
    vector_store = MagicMock()

    retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=embedder,
    )

    assert retriever.retrieve("") == []

    embedder.embed_query.assert_not_called()
    vector_store.search.assert_not_called()


def test_top_k_override_is_supported() -> None:
    embedder = MagicMock()
    vector_store = MagicMock()

    embedder.embed_query.return_value = [
        0.1
    ]

    vector_store.search.return_value = []

    retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=10,
    )

    retriever.retrieve(
        "test",
        top_k=3,
    )

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1],
        top_k=3,
    )
