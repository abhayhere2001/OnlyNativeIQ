"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval Diagnostics
File         : compare_vector_bm25_hybrid.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Compares Vector, BM25 and Hybrid retrieval for the same query.
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


from src.retrieval.bm25_retriever import (
    BM25Retriever,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.retrieval.vector_retriever import (
    VectorRetriever,
)
from src.utils.config_loader import (
    ConfigLoader,
)


DEFAULT_QUERY = (
    "How much ghee is required for one batch of Classic Thekua?"
)


def print_results(
    title: str,
    results,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if not results:
        print("No results.")
        return

    for result in results:
        print(
            f"Rank={result.rank:<3} "
            f"Score={result.score:.6f} "
            f"File={result.file_name} "
            f"Section={result.section} "
            f"Chunk={result.chunk_id}"
        )

        if result.source == "hybrid":
            print(
                "    "
                f"VectorRank="
                f"{result.metadata.get('vector_rank')} "
                f"BM25Rank="
                f"{result.metadata.get('bm25_rank')}"
            )


def main() -> None:
    config = ConfigLoader(
        settings_path="config/settings.yaml",
        project_root=PROJECT_ROOT,
    )

    query = (
        " ".join(sys.argv[1:]).strip()
        if len(sys.argv) > 1
        else DEFAULT_QUERY
    )

    vector = VectorRetriever.from_config(
        config
    )

    bm25 = BM25Retriever.from_config(
        config
    )

    hybrid = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        vector_weight=config.get(
            "hybrid_retrieval.vector_weight",
            default=0.6,
        ),
        bm25_weight=config.get(
            "hybrid_retrieval.bm25_weight",
            default=0.4,
        ),
        candidate_pool_size=config.get(
            "hybrid_retrieval.candidate_pool_size",
            default=20,
        ),
        output_top_k=config.get(
            "hybrid_retrieval.output_top_k",
            default=10,
        ),
        rrf_k=config.get(
            "hybrid_retrieval.rrf_k",
            default=60,
        ),
    )

    print()
    print("=" * 100)
    print("OnlyNativeIQ - Vector vs BM25 vs Hybrid")
    print("=" * 100)
    print(f"Query: {query}")
    print("=" * 100)

    vector_results = vector.retrieve(
        query,
        top_k=10,
    )

    bm25_results = bm25.retrieve(
        query,
        top_k=10,
    )

    hybrid_results = hybrid.retrieve(
        query,
        top_k=10,
    )

    print_results(
        "VECTOR",
        vector_results,
    )

    print_results(
        "BM25",
        bm25_results,
    )

    print_results(
        "HYBRID (RRF)",
        hybrid_results,
    )


if __name__ == "__main__":
    main()
