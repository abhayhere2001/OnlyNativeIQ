"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval
File         : retrieval_pipeline.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Updated On   : 04-Sep-2026
Description  :
    Orchestrates OnlyNativeIQ retrieval, reranking, authorization and
    post-reranker relevance filtering.

Current Retrieval Flow:
    Query
      -> Vector Retrieval
      -> BM25 Retrieval
      -> Hybrid RRF Fusion
      -> Optional CrossEncoder Reranking
      -> Access-Control Filtering
      -> Minimum Relevance Gate
      -> Authorized + Relevant Top-N Results

Future Extensions:
      -> Context compression
      -> Grounding
      -> Answer generation
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
)
from src.schemas.retrieval import (
    RetrievalResult,
)
from src.security.access_control import (
    AccessController,
    UserContext,
)
from src.security.document_filter import (
    DocumentFilter,
)
from src.utils.config_loader import (
    ConfigLoader,
)


@dataclass(slots=True)
class RetrievalPipelineResult:
    """Diagnostics and final authorized/relevant results for one query."""

    query: str
    user_role: str

    hybrid_candidates_count: int
    reranked_results_count: int
    authorized_results_count: int
    filtered_results_count: int

    reranking_enabled: bool
    access_control_enabled: bool

    # New diagnostic: results removed only because they did not meet the
    # configured minimum relevance score after security filtering.
    relevance_filtered_results_count: int = 0

    # Records the active threshold for diagnostics.
    minimum_relevance_score: float | None = None

    hybrid_results: list[
        RetrievalResult
    ] = field(
        default_factory=list
    )

    reranked_results: list[
        RetrievalResult
    ] = field(
        default_factory=list
    )

    # IMPORTANT:
    # final_results are BOTH authorized and relevant.
    final_results: list[
        RetrievalResult
    ] = field(
        default_factory=list
    )


class RetrievalPipeline:
    """
    Hybrid retrieval -> reranking -> authorization -> relevance gate.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        document_filter: DocumentFilter | None = None,
        reranking_enabled: bool = True,
        access_control_enabled: bool = True,
        hybrid_candidate_top_k: int = 20,
        rerank_output_top_k: int = 20,
        final_top_k: int = 5,
        minimum_relevance_score: float = 0.0,
    ) -> None:

        if hybrid_candidate_top_k <= 0:
            raise ValueError(
                "hybrid_candidate_top_k must be greater than zero."
            )

        if rerank_output_top_k <= 0:
            raise ValueError(
                "rerank_output_top_k must be greater than zero."
            )

        if final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than zero."
            )

        try:
            minimum_relevance_score = float(
                minimum_relevance_score
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "minimum_relevance_score must be numeric."
            ) from exc

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.reranker = reranker

        self.document_filter = (
            document_filter
        )

        self.reranking_enabled = (
            reranking_enabled
        )

        self.access_control_enabled = (
            access_control_enabled
        )

        self.hybrid_candidate_top_k = (
            hybrid_candidate_top_k
        )

        self.rerank_output_top_k = (
            rerank_output_top_k
        )

        self.final_top_k = (
            final_top_k
        )

        self.minimum_relevance_score = (
            minimum_relevance_score
        )

    @classmethod
    def from_config(
        cls,
        config: ConfigLoader,
    ) -> "RetrievalPipeline":
        """Create the complete retrieval pipeline from configuration."""

        hybrid = (
            HybridRetriever.from_config(
                config
            )
        )

        reranking_enabled = config.get(
            "reranking.enabled",
            default=True,
        )

        reranker = None

        if reranking_enabled:
            reranker = (
                CrossEncoderReranker.from_config(
                    config
                )
            )

        access_controller = (
            AccessController.from_config(
                config
            )
        )

        document_filter = (
            DocumentFilter(
                access_controller=(
                    access_controller
                )
            )
        )

        return cls(
            hybrid_retriever=hybrid,
            reranker=reranker,
            document_filter=document_filter,
            reranking_enabled=(
                reranking_enabled
            ),
            access_control_enabled=(
                access_controller
                .enforce_access_control
            ),
            hybrid_candidate_top_k=(
                config.get(
                    "reranking.input_top_k",
                    default=20,
                )
            ),
            rerank_output_top_k=(
                config.get(
                    "security.pre_filter_top_k",
                    default=20,
                )
            ),
            final_top_k=(
                config.get(
                    "security.final_top_k",
                    default=5,
                )
            ),
            minimum_relevance_score=(
                config.get(
                    "retrieval.minimum_relevance_score",
                    default=0.0,
                )
            ),
        )

    def retrieve(
        self,
        query: str,
        user: UserContext,
        top_k: int | None = None,
    ) -> RetrievalPipelineResult:
        """
        Run retrieval and return only results that are:
            1. relevant enough,
            2. authorized for the caller.

        The relevance gate is deliberately applied AFTER authorization.
        This gives diagnostics that clearly distinguish:
            - security-filtered results, and
            - authorized but low-relevance results.
        """

        effective_final_top_k = (
            top_k
            if top_k is not None
            else self.final_top_k
        )

        if (
            not query
            or not query.strip()
            or effective_final_top_k <= 0
        ):
            return RetrievalPipelineResult(
                query=query,
                user_role=(
                    user.role.value
                ),
                hybrid_candidates_count=0,
                reranked_results_count=0,
                authorized_results_count=0,
                filtered_results_count=0,
                reranking_enabled=(
                    self.reranking_enabled
                ),
                access_control_enabled=(
                    self.access_control_enabled
                ),
                relevance_filtered_results_count=0,
                minimum_relevance_score=(
                    self.minimum_relevance_score
                ),
            )

        # ---------------------------------------------------------------------
        # 1. Hybrid candidate retrieval
        # ---------------------------------------------------------------------

        hybrid_results = (
            self.hybrid_retriever.retrieve(
                query=query,
                top_k=(
                    self.hybrid_candidate_top_k
                ),
            )
        )

        # ---------------------------------------------------------------------
        # 2. CrossEncoder reranking
        # ---------------------------------------------------------------------

        if (
            self.reranking_enabled
            and self.reranker is not None
        ):
            reranked_results = (
                self.reranker.rerank(
                    query=query,
                    candidates=(
                        hybrid_results
                    ),
                    output_top_k=(
                        self.rerank_output_top_k
                    ),
                )
            )

        else:
            reranked_results = [
                result
                for result
                in hybrid_results[
                    :self.rerank_output_top_k
                ]
            ]

            for rank, result in enumerate(
                reranked_results,
                start=1,
            ):
                result.rank = rank

        # ---------------------------------------------------------------------
        # 3. Security / audience filtering
        #
        # IMPORTANT:
        # Do not truncate to final_top_k here. We want all authorized
        # pre-filter candidates to continue into the relevance gate.
        # ---------------------------------------------------------------------

        if (
            self.access_control_enabled
            and self.document_filter
            is not None
        ):
            filter_result = (
                self.document_filter
                .filter_results(
                    results=(
                        reranked_results
                    ),
                    user=user,
                    top_k=None,
                )
            )

            authorized_results = (
                filter_result
                .authorized_results
            )

            security_filtered_count = (
                filter_result
                .filtered_count
            )

        else:
            authorized_results = [
                result
                for result
                in reranked_results
            ]

            security_filtered_count = 0

        # ---------------------------------------------------------------------
        # 4. Post-reranker relevance gate
        #
        # CrossEncoder scores are used here because result.score is replaced
        # by reranker score in reranker.py.
        #
        # Example:
        #   shelf-life relevant chunk: +6.88  -> retained
        #   unrelated FAQ chunk:      -11.44  -> removed
        # ---------------------------------------------------------------------

        relevant_results = [
            result
            for result
            in authorized_results
            if float(
                result.score
            ) >= self.minimum_relevance_score
        ]

        relevance_filtered_count = (
            len(authorized_results)
            - len(relevant_results)
        )

        # ---------------------------------------------------------------------
        # 5. Final Top-K
        # ---------------------------------------------------------------------

        final_results = (
            relevant_results[
                :effective_final_top_k
            ]
        )

        for rank, result in enumerate(
            final_results,
            start=1,
        ):
            result.rank = rank

            metadata = dict(
                result.metadata
            )

            metadata.update(
                {
                    "relevance_gate_applied":
                        True,
                    "minimum_relevance_score":
                        self.minimum_relevance_score,
                }
            )

            result.metadata = (
                metadata
            )

        return RetrievalPipelineResult(
            query=query,
            user_role=(
                user.role.value
            ),
            hybrid_candidates_count=(
                len(
                    hybrid_results
                )
            ),
            reranked_results_count=(
                len(
                    reranked_results
                )
            ),
            authorized_results_count=(
                len(
                    final_results
                )
            ),
            filtered_results_count=(
                security_filtered_count
            ),
            reranking_enabled=(
                self.reranking_enabled
            ),
            access_control_enabled=(
                self.access_control_enabled
            ),
            relevance_filtered_results_count=(
                relevance_filtered_count
            ),
            minimum_relevance_score=(
                self.minimum_relevance_score
            ),
            hybrid_results=(
                hybrid_results
            ),
            reranked_results=(
                reranked_results
            ),
            final_results=(
                final_results
            ),
        )
