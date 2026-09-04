"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval
File         : hybrid_retriever.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Combines semantic vector retrieval and BM25 lexical retrieval using
    Reciprocal Rank Fusion (RRF).

Responsibilities:
    - Retrieve candidates from vector and BM25 retrievers.
    - Fuse rankings without directly mixing incompatible raw scores.
    - Preserve source, citation and access-control metadata.
    - Return one deduplicated ranked candidate list.

Design Principles:
    - RRF is used because vector similarity scores and BM25 scores are not on
      directly comparable scales.
    - Candidate identity is based on deterministic chunk_id.
    - Fusion weights and RRF constant are configurable.
    - Access control is preserved but enforced later by security filtering.
================================================================================
"""

from __future__ import annotations

from copy import deepcopy

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.schemas.retrieval import RetrievalResult
from src.utils.config_loader import ConfigLoader


class HybridRetrievalError(RuntimeError):
    """Raised when hybrid retrieval configuration is invalid."""


class HybridRetriever:
    """Combine vector and BM25 retrieval using weighted RRF."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        candidate_pool_size: int = 20,
        output_top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if vector_weight < 0:
            raise ValueError(
                "vector_weight cannot be negative."
            )

        if bm25_weight < 0:
            raise ValueError(
                "bm25_weight cannot be negative."
            )

        if (
            vector_weight == 0
            and bm25_weight == 0
        ):
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        if candidate_pool_size <= 0:
            raise ValueError(
                "candidate_pool_size must be greater than zero."
            )

        if output_top_k <= 0:
            raise ValueError(
                "output_top_k must be greater than zero."
            )

        if rrf_k < 0:
            raise ValueError(
                "rrf_k cannot be negative."
            )

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

        self.vector_weight = float(
            vector_weight
        )
        self.bm25_weight = float(
            bm25_weight
        )

        self.candidate_pool_size = (
            candidate_pool_size
        )

        self.output_top_k = output_top_k
        self.rrf_k = rrf_k

    @classmethod
    def from_config(
        cls,
        config: ConfigLoader,
    ) -> "HybridRetriever":
        """Create the hybrid retriever from central configuration."""

        return cls(
            vector_retriever=(
                VectorRetriever.from_config(
                    config
                )
            ),
            bm25_retriever=(
                BM25Retriever.from_config(
                    config
                )
            ),
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

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve and fuse vector + BM25 candidates.

        Returns:
            Deduplicated RetrievalResult list ordered by hybrid RRF score.
        """

        if not query or not query.strip():
            return []

        effective_top_k = (
            top_k
            if top_k is not None
            else self.output_top_k
        )

        if effective_top_k <= 0:
            return []

        vector_results = (
            self.vector_retriever.retrieve(
                query=query,
                top_k=self.candidate_pool_size,
            )
        )

        bm25_results = (
            self.bm25_retriever.retrieve(
                query=query,
                top_k=self.candidate_pool_size,
            )
        )

        return self.fuse(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=effective_top_k,
        )

    def fuse(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Fuse already-retrieved candidate lists using weighted RRF.
        """

        effective_top_k = (
            top_k
            if top_k is not None
            else self.output_top_k
        )

        if effective_top_k <= 0:
            return []

        candidates: dict[
            str,
            dict,
        ] = {}

        self._accumulate(
            candidates=candidates,
            results=vector_results,
            weight=self.vector_weight,
            source_name="vector",
        )

        self._accumulate(
            candidates=candidates,
            results=bm25_results,
            weight=self.bm25_weight,
            source_name="bm25",
        )

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                item["rrf_score"],
                item["source_count"],
            ),
            reverse=True,
        )

        output: list[
            RetrievalResult
        ] = []

        for rank, item in enumerate(
            ranked[:effective_top_k],
            start=1,
        ):
            result = deepcopy(
                item["result"]
            )

            result.rank = rank
            result.score = float(
                item["rrf_score"]
            )
            result.source = "hybrid"

            metadata = dict(
                result.metadata
            )

            metadata.update(
                {
                    "vector_rank":
                        item.get(
                            "vector_rank"
                        ),
                    "bm25_rank":
                        item.get(
                            "bm25_rank"
                        ),
                    "vector_score":
                        item.get(
                            "vector_score"
                        ),
                    "bm25_score":
                        item.get(
                            "bm25_score"
                        ),
                    "hybrid_rrf_score":
                        float(
                            item["rrf_score"]
                        ),
                    "retrieval_sources":
                        ",".join(
                            sorted(
                                item[
                                    "sources"
                                ]
                            )
                        ),
                }
            )

            result.metadata = metadata
            output.append(
                result
            )

        return output

    def _accumulate(
        self,
        candidates: dict[str, dict],
        results: list[RetrievalResult],
        weight: float,
        source_name: str,
    ) -> None:
        """Accumulate one retriever's rank contribution."""

        if weight == 0:
            return

        for position, result in enumerate(
            results,
            start=1,
        ):
            # Prefer actual result.rank where it is valid, but fall back to
            # sequential position for robustness.
            rank = (
                result.rank
                if result.rank > 0
                else position
            )

            contribution = (
                weight
                / (
                    self.rrf_k
                    + rank
                )
            )

            if result.chunk_id not in candidates:
                candidates[
                    result.chunk_id
                ] = {
                    "result":
                        deepcopy(
                            result
                        ),
                    "rrf_score":
                        0.0,
                    "source_count":
                        0,
                    "sources":
                        set(),
                    "vector_rank":
                        None,
                    "bm25_rank":
                        None,
                    "vector_score":
                        None,
                    "bm25_score":
                        None,
                }

            item = candidates[
                result.chunk_id
            ]

            item["rrf_score"] += (
                contribution
            )

            if (
                source_name
                not in item["sources"]
            ):
                item["source_count"] += 1
                item["sources"].add(
                    source_name
                )

            if source_name == "vector":
                item["vector_rank"] = (
                    rank
                )
                item["vector_score"] = (
                    float(
                        result.score
                    )
                )

            elif source_name == "bm25":
                item["bm25_rank"] = (
                    rank
                )
                item["bm25_score"] = (
                    float(
                        result.score
                    )
                )
