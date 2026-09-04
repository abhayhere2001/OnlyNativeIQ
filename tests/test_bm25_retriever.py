"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_bm25_retriever.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Unit tests for persistent BM25 indexing and retrieval.
================================================================================
"""

from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever
from src.schemas.document import DocumentChunk


def _chunk(
    chunk_id: str,
    content: str,
    file_name: str,
    access_level: str = "public",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("-S")[0],
        content=content,
        file_name=file_name,
        domain="products",
        chunk_index=1,
        section="Test Section",
        access_level=access_level,
        audience=["customer"],
        language="en",
    )


def test_bm25_favors_exact_product_terms(
    tmp_path: Path,
) -> None:
    """
    Use at least three records so discriminative terms get meaningful
    positive IDF under BM25Okapi.
    """

    retriever = BM25Retriever(
        persist_directory=tmp_path / "bm25",
        top_k=5,
    )

    chunks = [
        _chunk(
            "ON-PROD-0001-S001-C001",
            "Classic Thekua Standard Batch Formulation "
            "Ghee Moin 75 g Wheat Flour 500 g",
            "Classic_Thekua_Product_Guide.docx",
        ),
        _chunk(
            "ON-PROD-0004-S001-C001",
            "Classic Gujiya Standard Batch Formulation "
            "Ghee Moin 100 g Refined Flour 1000 g",
            "Classic_Gujiya_Product_Guide.docx",
        ),
        _chunk(
            "ON-POL-0001-S001-C001",
            "Shipping delivery policy Bengaluru Porter charges",
            "Shipping_Delivery_Policy.txt",
        ),
    ]

    assert retriever.build_index(
        chunks
    ) == 3

    results = retriever.retrieve(
        "How much ghee is required for Classic Thekua?"
    )

    assert results
    assert results[0].file_name == (
        "Classic_Thekua_Product_Guide.docx"
    )
    assert results[0].source == "bm25"


def test_bm25_persists_and_reloads(
    tmp_path: Path,
) -> None:
    persist = tmp_path / "bm25"

    retriever = BM25Retriever(
        persist_directory=persist
    )

    retriever.build_index(
        [
            _chunk(
                "ON-PROD-0001-S001-C001",
                "Classic Thekua ghee 75 g",
                "Classic_Thekua_Product_Guide.docx",
            )
        ]
    )

    reloaded = BM25Retriever(
        persist_directory=persist
    )

    assert reloaded.count() == 1

    results = reloaded.retrieve(
        "Classic Thekua"
    )

    # One-record corpora may produce zero/negative BM25Okapi scores,
    # but the sole matching record is still a valid ranked result.
    assert len(results) == 1
    assert results[0].file_name == (
        "Classic_Thekua_Product_Guide.docx"
    )


def test_empty_query_returns_empty(
    tmp_path: Path,
) -> None:
    retriever = BM25Retriever(
        persist_directory=(
            tmp_path / "bm25"
        )
    )

    assert retriever.retrieve(
        ""
    ) == []


def test_numbers_are_preserved_in_tokens(
    tmp_path: Path,
) -> None:
    retriever = BM25Retriever(
        persist_directory=(
            tmp_path / "bm25"
        )
    )

    tokens = retriever.tokenize(
        "Ghee 75 g, pack 200 g"
    )

    assert "75" in tokens
    assert "200" in tokens


def test_zero_or_negative_bm25_score_is_not_discarded(
    tmp_path: Path,
) -> None:
    """
    BM25 scores are ranking signals, not probabilities.
    Tiny corpora may legitimately produce non-positive scores.
    """

    retriever = BM25Retriever(
        persist_directory=(
            tmp_path / "bm25"
        )
    )

    retriever.build_index(
        [
            _chunk(
                "ON-TEST-001-S001-C001",
                "OnlyNative Classic Thekua",
                "Test.txt",
            )
        ]
    )

    results = retriever.retrieve(
        "Classic Thekua"
    )

    assert len(results) == 1
