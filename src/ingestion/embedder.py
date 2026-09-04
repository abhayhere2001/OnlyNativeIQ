"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Document Ingestion
File         : embedder.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Generates dense embeddings for document chunks and retrieval queries.
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from sentence_transformers import SentenceTransformer

from src.schemas.document import DocumentChunk, EmbeddedChunk


logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class DocumentEmbedder:
    """Generate dense embeddings for chunks and user queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
        device: str | None = None,
        show_progress_bar: bool = False,
        model: Any | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = model_name.strip()
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.show_progress_bar = show_progress_bar
        self._model = model

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate one vector for a retrieval query.
        """

        if not query or not query.strip():
            raise EmbeddingError(
                "Query cannot be empty."
            )

        model = self._get_model()

        try:
            vectors = model.encode(
                [query.strip()],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Query embedding failed using model "
                f"'{self.model_name}': {exc}"
            ) from exc

        vector_list = self._to_python_vectors(vectors)

        if (
            len(vector_list) != 1
            or not vector_list[0]
        ):
            raise EmbeddingError(
                "Expected exactly one non-empty query embedding."
            )

        return vector_list[0]

    def embed_chunk(
        self,
        chunk: DocumentChunk,
    ) -> EmbeddedChunk:
        results = self.embed_chunks([chunk])

        if len(results) != 1:
            raise EmbeddingError(
                "Expected exactly one embedding result."
            )

        return results[0]

    def embed_chunks(
        self,
        chunks: Iterable[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        chunk_list = list(chunks)

        if not chunk_list:
            return []

        self._validate_chunks(chunk_list)

        texts = [
            chunk.content
            for chunk in chunk_list
        ]

        model = self._get_model()

        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                normalize_embeddings=self.normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed using model "
                f"'{self.model_name}': {exc}"
            ) from exc

        vector_list = self._to_python_vectors(vectors)

        if len(vector_list) != len(chunk_list):
            raise EmbeddingError(
                "Embedding count does not match chunk count."
            )

        dimensions = {
            len(vector)
            for vector in vector_list
        }

        if not dimensions or 0 in dimensions:
            raise EmbeddingError(
                "Embedding model returned an empty vector."
            )

        if len(dimensions) != 1:
            raise EmbeddingError(
                "Embedding vectors have inconsistent dimensions."
            )

        dimension = next(iter(dimensions))

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=vector,
                model_name=self.model_name,
                embedding_dimension=dimension,
            )
            for chunk, vector in zip(
                chunk_list,
                vector_list,
            )
        ]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            if self.device:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
            else:
                self._model = SentenceTransformer(
                    self.model_name
                )
        except Exception as exc:
            raise EmbeddingError(
                f"Unable to initialize embedding model "
                f"'{self.model_name}': {exc}"
            ) from exc

        return self._model

    @staticmethod
    def _validate_chunks(
        chunks: list[DocumentChunk],
    ) -> None:
        for chunk in chunks:
            if not chunk.chunk_id:
                raise EmbeddingError(
                    "Cannot embed a chunk without chunk_id."
                )

            if not chunk.content or not chunk.content.strip():
                raise EmbeddingError(
                    f"Chunk '{chunk.chunk_id}' has empty content."
                )

    @staticmethod
    def _to_python_vectors(
        vectors: Any,
    ) -> list[list[float]]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()

        try:
            return [
                [
                    float(value)
                    for value in vector
                ]
                for vector in vectors
            ]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(
                "Embedding model returned an unsupported vector structure."
            ) from exc
