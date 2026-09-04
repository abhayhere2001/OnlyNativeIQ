"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval
File         : reranker.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Reranks retrieved OnlyNativeIQ candidates using a CrossEncoder model.

Responsibilities:
    - Score (query, candidate_text) pairs jointly.
    - Reorder Hybrid-RRF candidates by semantic relevance.
    - Preserve chunk identity, citation metadata and access metadata.
    - Support configurable model, candidate count, output count, batch size,
      device and enable/disable behavior.

Design Principles:
    - Reranking is performed after candidate retrieval/fusion.
    - Raw retrieval scores are retained in metadata for diagnostics.
    - Unit tests inject a fake CrossEncoder model; no model download is needed.
    - Model loading is lazy.
================================================================================
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from sentence_transformers import CrossEncoder

from src.schemas.retrieval import RetrievalResult
from src.utils.config_loader import ConfigLoader


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_INPUT_TOP_K = 20
DEFAULT_OUTPUT_TOP_K = 5
DEFAULT_BATCH_SIZE = 16


class RerankingError(RuntimeError):
    """Raised when candidate reranking fails."""


class CrossEncoderReranker:
    """Rerank retrieval candidates using a CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        input_top_k: int = DEFAULT_INPUT_TOP_K,
        output_top_k: int = DEFAULT_OUTPUT_TOP_K,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if input_top_k <= 0:
            raise ValueError(
                "input_top_k must be greater than zero."
            )

        if output_top_k <= 0:
            raise ValueError(
                "output_top_k must be greater than zero."
            )

        if output_top_k > input_top_k:
            raise ValueError(
                "output_top_k cannot be greater than input_top_k."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name.strip()
        self.input_top_k = input_top_k
        self.output_top_k = output_top_k
        self.batch_size = batch_size
        self.device = device
        self._model = model

    @classmethod
    def from_config(
        cls,
        config: ConfigLoader,
    ) -> "CrossEncoderReranker":
        """Create reranker from centralized configuration."""

        return cls(
            model_name=config.get(
                "reranking.model_name",
                required=True,
            ),
            input_top_k=config.get(
                "reranking.input_top_k",
                default=20,
            ),
            output_top_k=config.get(
                "reranking.output_top_k",
                default=5,
            ),
            batch_size=config.get(
                "reranking.batch_size",
                default=16,
            ),
            device=config.get(
                "reranking.device",
                default=None,
            ),
        )

    def rerank(
        self,
        query: str,
        candidates: Iterable[RetrievalResult],
        output_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Rerank candidate results using query-document pair scoring.
        """

        if not query or not query.strip():
            return []

        candidate_list = list(candidates)

        if not candidate_list:
            return []

        candidate_list = candidate_list[
            :self.input_top_k
        ]

        effective_output_top_k = (
            output_top_k
            if output_top_k is not None
            else self.output_top_k
        )

        if effective_output_top_k <= 0:
            return []

        effective_output_top_k = min(
            effective_output_top_k,
            len(candidate_list),
        )

        pairs = [
            (
                query.strip(),
                candidate.content,
            )
            for candidate in candidate_list
        ]

        model = self._get_model()

        try:
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RerankingError(
                f"Cross-encoder reranking failed using "
                f"'{self.model_name}': {exc}"
            ) from exc

        score_list = self._to_float_list(
            scores
        )

        if len(score_list) != len(
            candidate_list
        ):
            raise RerankingError(
                "Reranker score count does not match candidate count."
            )

        ranked_pairs = sorted(
            zip(
                candidate_list,
                score_list,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        output: list[
            RetrievalResult
        ] = []

        for new_rank, (
            candidate,
            reranker_score,
        ) in enumerate(
            ranked_pairs[
                :effective_output_top_k
            ],
            start=1,
        ):
            result = deepcopy(
                candidate
            )

            metadata = dict(
                result.metadata
            )

            metadata.update(
                {
                    "pre_rerank_rank":
                        candidate.rank,
                    "pre_rerank_score":
                        float(
                            candidate.score
                        ),
                    "reranker_model":
                        self.model_name,
                    "reranker_score":
                        float(
                            reranker_score
                        ),
                }
            )

            result.rank = new_rank
            result.score = float(
                reranker_score
            )
            result.source = (
                f"{candidate.source}+reranker"
            )
            result.metadata = metadata

            output.append(
                result
            )

        return output

    def _get_model(self) -> Any:
        """Lazily initialize CrossEncoder."""

        if self._model is not None:
            return self._model

        try:
            if self.device:
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
            else:
                self._model = CrossEncoder(
                    self.model_name
                )
        except Exception as exc:
            raise RerankingError(
                f"Unable to initialize reranker model "
                f"'{self.model_name}': {exc}"
            ) from exc

        return self._model

    @staticmethod
    def _to_float_list(
        values: Any,
    ) -> list[float]:
        """Convert NumPy/Torch/list-like scores to Python floats."""

        if hasattr(values, "tolist"):
            values = values.tolist()

        try:
            return [
                float(value)
                for value in values
            ]
        except (TypeError, ValueError) as exc:
            raise RerankingError(
                "Reranker returned unsupported score structure."
            ) from exc
