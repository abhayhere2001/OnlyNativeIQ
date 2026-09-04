"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval Diagnostics
File         : compare_hybrid_reranked.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Compares Hybrid-RRF candidates against CrossEncoder-reranked results.
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


from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline,
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
    print("=" * 110)
    print(title)
    print("=" * 110)

    for result in results:
        print(
            f"Rank={result.rank:<3} "
            f"Score={result.score:.6f} "
            f"File={result.file_name} "
            f"Section={result.section} "
            f"Chunk={result.chunk_id}"
        )

        if (
            "pre_rerank_rank"
            in result.metadata
        ):
            print(
                "    "
                f"BeforeRerank="
                f"{result.metadata.get('pre_rerank_rank')} "
                f"HybridScore="
                f"{result.metadata.get('pre_rerank_score')} "
                f"RerankerScore="
                f"{result.metadata.get('reranker_score')}"
            )


def main() -> None:
    config = ConfigLoader(
        settings_path="config/settings.yaml",
        project_root=PROJECT_ROOT,
    )

    query = (
        " ".join(
            sys.argv[1:]
        ).strip()
        if len(sys.argv) > 1
        else DEFAULT_QUERY
    )

    pipeline = (
        RetrievalPipeline.from_config(
            config
        )
    )

    result = pipeline.retrieve(
        query
    )

    print()
    print("=" * 110)
    print("OnlyNativeIQ - Hybrid vs Reranked Retrieval")
    print("=" * 110)
    print(f"Query: {query}")
    print(
        f"Hybrid candidates : "
        f"{result.hybrid_candidates_count}"
    )
    print(
        f"Final results     : "
        f"{result.final_results_count}"
    )
    print(
        f"Reranking enabled : "
        f"{result.reranking_enabled}"
    )

    print_results(
        "HYBRID RRF CANDIDATES",
        result.hybrid_results,
    )

    print_results(
        "FINAL RERANKED RESULTS",
        result.final_results,
    )


if __name__ == "__main__":
    main()
