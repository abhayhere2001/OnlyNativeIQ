"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval Diagnostics
File         : compare_vector_bm25.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Compares semantic vector retrieval and BM25 lexical retrieval using the
    same OnlyNativeIQ query.

Purpose:
    - Compare vector-search ranking against BM25 ranking.
    - Inspect exact-product-name retrieval behavior.
    - Validate source/security metadata from both retrievers.
    - Establish a baseline before implementing hybrid retrieval.

Usage:
    python scripts/compare_vector_bm25.py

Optional custom query:
    python scripts/compare_vector_bm25.py \
        "How much ghee is required for one batch of Classic Thekua?"
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.utils.config_loader import ConfigLoader


DEFAULT_QUERY = (
    "How much ghee is required for one batch of Classic Thekua?"
)


def print_result(
    result,
    *,
    result_number: int,
) -> None:
    """Print one retrieval result."""

    print("-" * 80)
    print(f"Result       : {result_number}")
    print(f"Rank         : {result.rank}")
    print(f"Score        : {result.score:.4f}")
    print(f"Chunk ID     : {result.chunk_id}")
    print(f"Document ID  : {result.document_id}")
    print(f"File         : {result.file_name}")
    print(f"Domain       : {result.domain}")
    print(f"Section      : {result.section}")
    print(f"Page         : {result.page_number}")
    print(f"Access Level : {result.access_level}")
    print(f"Audience     : {result.audience}")
    print(f"Source       : {result.source}")
    print()
    print("CONTENT")
    print("-" * 80)
    print(result.content)
    print()


def print_result_set(
    title: str,
    results,
) -> None:
    """Print one retriever's result set."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results returned.")
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        print_result(
            result,
            result_number=index,
        )


def print_rank_comparison(
    vector_results,
    bm25_results,
) -> None:
    """
    Print compact side-by-side chunk ranking.

    This makes it easy to see whether the same chunk is promoted differently
    by semantic and lexical retrieval.
    """

    print()
    print("=" * 100)
    print("RANK COMPARISON")
    print("=" * 100)

    vector_rank = {
        result.chunk_id: result.rank
        for result in vector_results
    }

    bm25_rank = {
        result.chunk_id: result.rank
        for result in bm25_results
    }

    all_chunk_ids = []

    for result in vector_results:
        if result.chunk_id not in all_chunk_ids:
            all_chunk_ids.append(
                result.chunk_id
            )

    for result in bm25_results:
        if result.chunk_id not in all_chunk_ids:
            all_chunk_ids.append(
                result.chunk_id
            )

    print(
        f"{'Chunk ID':40}"
        f"{'Vector Rank':>15}"
        f"{'BM25 Rank':>15}"
    )

    print("-" * 100)

    for chunk_id in all_chunk_ids:
        vector_value = vector_rank.get(
            chunk_id,
            "-"
        )

        bm25_value = bm25_rank.get(
            chunk_id,
            "-"
        )

        print(
            f"{chunk_id:40}"
            f"{str(vector_value):>15}"
            f"{str(bm25_value):>15}"
        )

    print("=" * 100)


def main() -> None:
    """Run vector and BM25 retrieval for the same query."""

    config = ConfigLoader(
        settings_path="config/settings.yaml",
        project_root=PROJECT_ROOT,
    )

    vector_retriever = (
        VectorRetriever.from_config(
            config
        )
    )

    bm25_retriever = (
        BM25Retriever.from_config(
            config
        )
    )

    if len(sys.argv) > 1:
        query = " ".join(
            sys.argv[1:]
        ).strip()
    else:
        query = DEFAULT_QUERY

    vector_top_k = config.get(
        "vector_retrieval.top_k",
        default=10,
    )

    bm25_top_k = config.get(
        "bm25.top_k",
        default=10,
    )

    print()
    print("=" * 100)
    print("OnlyNativeIQ - Vector vs BM25 Retrieval Comparison")
    print("=" * 100)
    print(f"Query        : {query}")
    print(
        "Vector Store : "
        f"{config.get('vector_store.provider', required=True)}"
    )
    print(
        "Embedding    : "
        f"{config.get('embeddings.model_name', required=True)}"
    )
    print(f"Vector Top K : {vector_top_k}")
    print(f"BM25 Top K   : {bm25_top_k}")
    print("=" * 100)

    vector_results = (
        vector_retriever.retrieve(
            query=query,
            top_k=vector_top_k,
        )
    )

    bm25_results = (
        bm25_retriever.retrieve(
            query=query,
            top_k=bm25_top_k,
        )
    )

    print_result_set(
        "VECTOR RETRIEVAL RESULTS",
        vector_results,
    )

    print_result_set(
        "BM25 RETRIEVAL RESULTS",
        bm25_results,
    )

    print_rank_comparison(
        vector_results,
        bm25_results,
    )

    print()
    print("=" * 100)
    print("Comparison completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()
