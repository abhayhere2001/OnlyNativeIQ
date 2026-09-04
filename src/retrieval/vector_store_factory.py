"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval / Vector Store
File         : vector_store_factory.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Configurable vector-store abstraction supporting Chroma and FAISS.
================================================================================
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from src.schemas.document import EmbeddedChunk
from src.schemas.retrieval import RetrievalResult


class VectorStoreError(RuntimeError):
    """Raised when vector-store persistence or retrieval fails."""


class UnsupportedVectorStoreError(VectorStoreError):
    """Raised when an unsupported vector-store provider is configured."""


class BaseVectorStore(ABC):
    """Common persistence/retrieval contract."""

    @abstractmethod
    def add_embeddings(
        self,
        embedded_chunks: Iterable[EmbeddedChunk],
    ) -> int:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass


def _sanitize_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:

    result: dict[str, str | int | float | bool] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, (list, tuple, set)):
            result[key] = ",".join(
                str(item)
                for item in value
            )
        elif isinstance(value, dict):
            result[key] = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            result[key] = str(value)

    return result


def _audience_from_metadata(
    metadata: dict[str, Any],
) -> list[str]:
    value = metadata.get("audience")

    if not value:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        distance_metric: str = "cosine",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is required for Chroma."
            ) from exc

        self.persist_directory = Path(
            persist_directory
        ).resolve()

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = collection_name
        self.distance_metric = distance_metric

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self._collection = (
            self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space":
                        self.distance_metric
                },
            )
        )

    def add_embeddings(
        self,
        embedded_chunks: Iterable[EmbeddedChunk],
    ) -> int:

        records = list(embedded_chunks)

        if not records:
            return 0

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for record in records:
            chunk = record.chunk

            ids.append(chunk.chunk_id)
            embeddings.append(record.embedding)
            documents.append(chunk.content)

            metadata = chunk.to_metadata_dict()
            metadata.update(
                {
                    "embedding_model":
                        record.model_name,
                    "embedding_dimension":
                        record.embedding_dimension,
                }
            )

            metadatas.append(
                _sanitize_metadata(metadata)
            )

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:

        if top_k <= 0:
            return []

        result = self._collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get(
            "documents",
            [[]],
        )[0]
        metadatas = result.get(
            "metadatas",
            [[]],
        )[0]
        distances = result.get(
            "distances",
            [[]],
        )[0]

        output: list[RetrievalResult] = []

        for rank, (
            chunk_id,
            content,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):
            metadata = metadata or {}

            # Convert cosine distance to an intuitive higher-is-better score.
            score = (
                1.0 - float(distance)
                if self.distance_metric == "cosine"
                else -float(distance)
            )

            page_number = metadata.get(
                "page_number"
            )

            if isinstance(page_number, str):
                try:
                    page_number = int(
                        page_number
                    )
                except ValueError:
                    page_number = None

            output.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    content=content or "",
                    score=score,
                    rank=rank,
                    document_id=metadata.get(
                        "document_id"
                    ),
                    file_name=metadata.get(
                        "file_name"
                    ),
                    domain=metadata.get(
                        "domain"
                    ),
                    section=metadata.get(
                        "section"
                    ),
                    page_number=page_number,
                    access_level=metadata.get(
                        "access_level"
                    ),
                    audience=_audience_from_metadata(
                        metadata
                    ),
                    source="vector",
                    metadata=dict(metadata),
                )
            )

        return output

    def count(self) -> int:
        return int(
            self._collection.count()
        )


class FaissVectorStore(BaseVectorStore):
    def __init__(
        self,
        persist_directory: str | Path,
        index_name: str = "onlynativeiq",
        distance_metric: str = "cosine",
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "faiss-cpu is required for FAISS."
            ) from exc

        self._faiss = faiss

        self.persist_directory = Path(
            persist_directory
        ).resolve()

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_name = index_name
        self.distance_metric = (
            distance_metric.lower()
        )

        self.index_path = (
            self.persist_directory
            / f"{self.index_name}.faiss"
        )

        self.records_path = (
            self.persist_directory
            / f"{self.index_name}_records.json"
        )

        self._index = None
        self._records: list[
            dict[str, Any]
        ] = []

        self._load_existing()

    def add_embeddings(
        self,
        embedded_chunks: Iterable[EmbeddedChunk],
    ) -> int:

        records = list(embedded_chunks)

        if not records:
            return 0

        import numpy as np

        dimension = records[
            0
        ].embedding_dimension

        existing_by_id = {
            item["chunk_id"]: item
            for item in self._records
        }

        new_by_id = {}

        for record in records:
            chunk = record.chunk

            new_by_id[
                chunk.chunk_id
            ] = {
                "chunk_id":
                    chunk.chunk_id,
                "content":
                    chunk.content,
                "metadata":
                    _sanitize_metadata(
                        chunk.to_metadata_dict()
                    ),
                "embedding":
                    record.embedding,
            }

        existing_by_id.update(
            new_by_id
        )

        combined_records = list(
            existing_by_id.values()
        )

        vectors = np.asarray(
            [
                item["embedding"]
                for item in combined_records
            ],
            dtype="float32",
        )

        if self.distance_metric == "cosine":
            self._faiss.normalize_L2(
                vectors
            )
            index = self._faiss.IndexFlatIP(
                dimension
            )
        else:
            index = self._faiss.IndexFlatL2(
                dimension
            )

        index.add(vectors)

        self._faiss.write_index(
            index,
            str(self.index_path),
        )

        serializable_records = [
            {
                "chunk_id":
                    item["chunk_id"],
                "content":
                    item["content"],
                "metadata":
                    item["metadata"],
            }
            for item in combined_records
        ]

        self.records_path.write_text(
            json.dumps(
                serializable_records,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self._index = index
        self._records = (
            serializable_records
        )

        return len(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:

        if (
            top_k <= 0
            or self._index is None
            or not self._records
        ):
            return []

        import numpy as np

        query = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        if self.distance_metric == "cosine":
            self._faiss.normalize_L2(
                query
            )

        limit = min(
            top_k,
            len(self._records),
        )

        distances, indices = (
            self._index.search(
                query,
                limit,
            )
        )

        output: list[
            RetrievalResult
        ] = []

        for rank, (
            raw_score,
            index_position,
        ) in enumerate(
            zip(
                distances[0],
                indices[0],
            ),
            start=1,
        ):
            if index_position < 0:
                continue

            record = self._records[
                int(index_position)
            ]

            metadata = record.get(
                "metadata",
                {},
            )

            score = (
                float(raw_score)
                if self.distance_metric
                == "cosine"
                else -float(raw_score)
            )

            page_number = metadata.get(
                "page_number"
            )

            output.append(
                RetrievalResult(
                    chunk_id=record[
                        "chunk_id"
                    ],
                    content=record.get(
                        "content",
                        "",
                    ),
                    score=score,
                    rank=rank,
                    document_id=metadata.get(
                        "document_id"
                    ),
                    file_name=metadata.get(
                        "file_name"
                    ),
                    domain=metadata.get(
                        "domain"
                    ),
                    section=metadata.get(
                        "section"
                    ),
                    page_number=(
                        int(page_number)
                        if page_number is not None
                        else None
                    ),
                    access_level=metadata.get(
                        "access_level"
                    ),
                    audience=_audience_from_metadata(
                        metadata
                    ),
                    source="vector",
                    metadata=dict(
                        metadata
                    ),
                )
            )

        return output

    def count(self) -> int:
        return len(
            self._records
        )

    def _load_existing(self) -> None:
        if self.records_path.exists():
            self._records = json.loads(
                self.records_path.read_text(
                    encoding="utf-8"
                )
            )

        if self.index_path.exists():
            self._index = (
                self._faiss.read_index(
                    str(self.index_path)
                )
            )


class VectorStoreFactory:
    @staticmethod
    def create(
        provider: str,
        *,
        chroma_persist_directory: str | Path | None = None,
        chroma_collection_name: str = "onlynativeiq",
        faiss_persist_directory: str | Path | None = None,
        faiss_index_name: str = "onlynativeiq",
        distance_metric: str = "cosine",
    ) -> BaseVectorStore:

        normalized_provider = (
            provider.strip().lower()
        )

        if normalized_provider == "chroma":
            if chroma_persist_directory is None:
                raise VectorStoreError(
                    "Chroma persist directory is required."
                )

            return ChromaVectorStore(
                persist_directory=(
                    chroma_persist_directory
                ),
                collection_name=(
                    chroma_collection_name
                ),
                distance_metric=distance_metric,
            )

        if normalized_provider == "faiss":
            if faiss_persist_directory is None:
                raise VectorStoreError(
                    "FAISS persist directory is required."
                )

            return FaissVectorStore(
                persist_directory=(
                    faiss_persist_directory
                ),
                index_name=faiss_index_name,
                distance_metric=distance_metric,
            )

        raise UnsupportedVectorStoreError(
            f"Unsupported vector-store provider "
            f"'{provider}'."
        )

    @staticmethod
    def from_config(
        config,
    ) -> BaseVectorStore:

        provider = config.get(
            "vector_store.provider",
            required=True,
        )

        distance_metric = config.get(
            "vector_store.distance_metric",
            default="cosine",
        )

        return VectorStoreFactory.create(
            provider=provider,
            chroma_persist_directory=(
                config.resolve_path(
                    "vector_store.chroma.persist_directory"
                )
            ),
            chroma_collection_name=config.get(
                "vector_store.chroma.collection_name",
                default="onlynativeiq",
            ),
            faiss_persist_directory=(
                config.resolve_path(
                    "vector_store.faiss.persist_directory"
                )
            ),
            faiss_index_name=config.get(
                "vector_store.faiss.index_name",
                default="onlynativeiq",
            ),
            distance_metric=distance_metric,
        )
