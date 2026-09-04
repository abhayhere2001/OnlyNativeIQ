"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval Diagnostics
File         : test_vector_search.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Runs real semantic-search queries against the configured OnlyNativeIQ
    vector store and prints ranked retrieval results for manual inspection.

Purpose:
    - Validate real vector retrieval after ingestion.
    - Check whether the correct document/section appears in Top-K.
    - Inspect similarity scores and source/security metadata.
    - Validate Chroma or FAISS without changing this script.

Important:
    Run the ingestion pipeline first so the configured vector store contains
    the current corpus.

Usage:
    python scripts/test_vector_search.py

    Optional custom query:
    python scripts/test_vector_search.py "What is the shelf life of Thekua?"
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.retrieval.vector_retriever import VectorRetriever
from src.utils.config_loader import ConfigLoader


DEFAULT_QUERIES = [
    "How much ghee is required for one batch of Classic Thekua?",
    "What is the shelf life of Jaggery Thekua?",
    "What is the price of Garlic Blended Nimki?",
    "How can a customer cancel an order?",
    "What protective equipment must production employees wear?",
]


def print_result(
    result,
) -> None:
    """Print one ranked retrieval result."""

    print("-" * 80)
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


def run_query(
    retriever: VectorRetriever,
    query: str,
    top_k: int,
) -> None:
    """Execute and print one real semantic query."""

    print()
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)
    print()

    results = retriever.retrieve(
        query=query,
        top_k=top_k,
    )

    if not results:
        print("No vector-search results returned.")
        return

    print(
        f"Returned {len(results)} result(s)."
    )
    print()

    for result in results:
        print_result(result)


def main() -> None:
    """Run the real-corpus semantic retrieval diagnostic."""

    config = ConfigLoader(
        settings_path="config/settings.yaml",
        project_root=PROJECT_ROOT,
    )

    provider = config.get(
        "vector_store.provider",
        required=True,
    )

    top_k = config.get(
        "vector_retrieval.top_k",
        default=5,
    )

    retriever = VectorRetriever.from_config(
        config
    )

    print()
    print("=" * 80)
    print("OnlyNativeIQ - Real Vector Retrieval Test")
    print("=" * 80)
    print(f"Vector Store : {provider}")
    print(
        "Embedding    : "
        f"{config.get('embeddings.model_name', required=True)}"
    )
    print(f"Top K        : {top_k}")
    print("=" * 80)

    # If a query is supplied on the command line, test only that query.
    if len(sys.argv) > 1:
        queries = [
            " ".join(sys.argv[1:]).strip()
        ]
    else:
        queries = DEFAULT_QUERIES

    for query in queries:
        run_query(
            retriever=retriever,
            query=query,
            top_k=top_k,
        )

    print()
    print("=" * 80)
    print("Vector retrieval diagnostic completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
