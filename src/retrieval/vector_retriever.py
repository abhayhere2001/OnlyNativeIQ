"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval
File         : vector_retriever.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Performs semantic retrieval from the configured Chroma or FAISS vector
    store.

Responsibilities:
    - Embed the user query using the same embedding model used for ingestion.
    - Search the configured vector store.
    - Apply configurable top_k and optional score threshold.
    - Return content together with source/security metadata.

Notes:
    - Access-control filtering will be added in document_filter.py.
    - BM25/hybrid retrieval is handled by separate components.
================================================================================
"""

from __future__ import annotations

from src.ingestion.embedder import DocumentEmbedder
from src.retrieval.vector_store_factory import (
    BaseVectorStore,
    VectorStoreFactory,
)
from src.schemas.retrieval import RetrievalResult
from src.utils.config_loader import ConfigLoader


class VectorRetriever:
    """Semantic retriever for OnlyNativeIQ."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: DocumentEmbedder,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold

    @classmethod
    def from_config(
        cls,
        config: ConfigLoader,
    ) -> "VectorRetriever":

        vector_store = (
            VectorStoreFactory.from_config(
                config
            )
        )

        embedder = DocumentEmbedder(
            model_name=config.get(
                "embeddings.model_name",
                required=True,
            ),
            batch_size=config.get(
                "embeddings.batch_size",
                default=32,
            ),
            normalize_embeddings=config.get(
                "embeddings.normalize_embeddings",
                default=True,
            ),
            device=config.get(
                "embeddings.device",
                default=None,
            ),
            show_progress_bar=False,
        )

        return cls(
            vector_store=vector_store,
            embedder=embedder,
            top_k=config.get(
                "vector_retrieval.top_k",
                default=10,
            ),
            score_threshold=config.get(
                "vector_retrieval.score_threshold",
                default=None,
            ),
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve semantically similar chunks.
        """

        if not query or not query.strip():
            return []

        effective_top_k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        if effective_top_k <= 0:
            return []

        query_embedding = (
            self.embedder.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
        )

        if self.score_threshold is None:
            return results

        filtered = [
            result
            for result in results
            if result.score
            >= self.score_threshold
        ]

        # Re-rank sequentially after filtering.
        for rank, result in enumerate(
            filtered,
            start=1,
        ):
            result.rank = rank

        return filtered
