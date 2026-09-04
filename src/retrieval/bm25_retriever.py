"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval
File         : bm25_retriever.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Builds, persists and queries a BM25 lexical index over OnlyNativeIQ chunks.

Responsibilities:
    - Tokenize chunk/query text deterministically.
    - Build BM25 over retrieval-ready DocumentChunk objects.
    - Persist chunk content and metadata to disk.
    - Rebuild the in-memory BM25 model from persisted records on startup.
    - Return ranked RetrievalResult objects with source/security metadata.

Important:
    BM25Okapi scores are relative ranking scores. On very small corpora they
    may legitimately be zero or negative because of IDF behavior. Therefore,
    valid ranked records must not be discarded merely because score <= 0.
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from rank_bm25 import BM25Okapi

from src.schemas.document import DocumentChunk
from src.schemas.retrieval import RetrievalResult
from src.utils.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class BM25Error(RuntimeError):
    """Raised when BM25 indexing or retrieval fails."""


class BM25Retriever:
    """Persistent BM25 lexical retriever for OnlyNativeIQ."""

    def __init__(
        self,
        persist_directory: str | Path,
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
        min_token_length: int = 1,
    ) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if min_token_length <= 0:
            raise ValueError(
                "min_token_length must be greater than zero."
            )

        self.persist_directory = Path(
            persist_directory
        ).resolve()

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records_path = (
            self.persist_directory
            / "bm25_records.json"
        )

        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase
        self.min_token_length = min_token_length

        self._records: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        self._load_existing()

    @classmethod
    def from_config(
        cls,
        config: ConfigLoader,
    ) -> "BM25Retriever":
        """Create BM25Retriever from centralized configuration."""

        return cls(
            persist_directory=config.resolve_path(
                "bm25.persist_directory"
            ),
            top_k=config.get(
                "bm25.top_k",
                default=10,
            ),
            k1=config.get(
                "bm25.k1",
                default=1.5,
            ),
            b=config.get(
                "bm25.b",
                default=0.75,
            ),
            lowercase=config.get(
                "bm25.lowercase",
                default=True,
            ),
            min_token_length=config.get(
                "bm25.min_token_length",
                default=1,
            ),
        )

    # -------------------------------------------------------------------------
    # Index construction
    # -------------------------------------------------------------------------

    def build_index(
        self,
        chunks: Iterable[DocumentChunk],
    ) -> int:
        """
        Rebuild and persist BM25 using the supplied current corpus chunks.
        """

        chunk_list = list(chunks)

        if not chunk_list:
            self._records = []
            self._tokenized_corpus = []
            self._bm25 = None
            self._persist_records()
            return 0

        records: list[dict[str, Any]] = []
        tokenized_corpus: list[list[str]] = []
        seen_chunk_ids: set[str] = set()

        for chunk in chunk_list:
            self._validate_chunk(chunk)

            if chunk.chunk_id in seen_chunk_ids:
                raise BM25Error(
                    "Duplicate chunk_id encountered while "
                    f"building BM25: {chunk.chunk_id}"
                )

            seen_chunk_ids.add(chunk.chunk_id)

            tokens = self.tokenize(
                chunk.content
            )

            if not tokens:
                raise BM25Error(
                    f"Chunk '{chunk.chunk_id}' produced no BM25 tokens."
                )

            tokenized_corpus.append(tokens)

            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": self._serializable_metadata(
                        chunk.to_metadata_dict()
                    ),
                }
            )

        try:
            bm25 = BM25Okapi(
                tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )
        except Exception as exc:
            raise BM25Error(
                f"Unable to build BM25 model: {exc}"
            ) from exc

        self._records = records
        self._tokenized_corpus = tokenized_corpus
        self._bm25 = bm25

        self._persist_records()

        logger.info(
            "Built BM25 index with %d chunk(s).",
            len(records),
        )

        return len(records)

    def count(self) -> int:
        """Return number of indexed chunks."""
        return len(self._records)

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve lexically relevant chunks using BM25.

        Note:
            BM25Okapi scores are ranking signals, not probabilities.
            Zero/negative scores can occur on tiny corpora and are still valid
            for ordering, so they are intentionally retained.
        """

        if not query or not query.strip():
            return []

        if (
            self._bm25 is None
            or not self._records
        ):
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        effective_top_k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        if effective_top_k <= 0:
            return []

        try:
            scores = self._bm25.get_scores(
                query_tokens
            )
        except Exception as exc:
            raise BM25Error(
                f"BM25 retrieval failed: {exc}"
            ) from exc

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(
                scores[index]
            ),
            reverse=True,
        )[:effective_top_k]

        results: list[RetrievalResult] = []

        for index in ranked_indices:
            raw_score = float(
                scores[index]
            )

            record = self._records[
                index
            ]

            metadata = record.get(
                "metadata",
                {},
            )

            page_number = metadata.get(
                "page_number"
            )

            if isinstance(
                page_number,
                str,
            ):
                try:
                    page_number = int(
                        page_number
                    )
                except ValueError:
                    page_number = None

            results.append(
                RetrievalResult(
                    chunk_id=record[
                        "chunk_id"
                    ],
                    content=record.get(
                        "content",
                        "",
                    ),
                    score=raw_score,
                    rank=len(results) + 1,
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
                    audience=self._audience_from_metadata(
                        metadata
                    ),
                    source="bm25",
                    metadata=dict(
                        metadata
                    ),
                )
            )

        return results

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
    ) -> list[str]:
        """
        Tokenize text for BM25.

        Numbers are deliberately preserved because prices, quantities,
        pack sizes and dates are important lexical signals in OnlyNativeIQ.
        """

        if not text:
            return []

        value = (
            text.lower()
            if self.lowercase
            else text
        )

        tokens = re.findall(
            r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*",
            value,
        )

        return [
            token
            for token in tokens
            if len(token)
            >= self.min_token_length
        ]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _load_existing(self) -> None:
        """Load persisted records and reconstruct BM25."""

        if not self.records_path.exists():
            return

        try:
            payload = json.loads(
                self.records_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            raise BM25Error(
                f"Unable to load BM25 records: {exc}"
            ) from exc

        if not isinstance(
            payload,
            list,
        ):
            raise BM25Error(
                "BM25 records file must contain a JSON list."
            )

        self._records = payload

        if not self._records:
            self._tokenized_corpus = []
            self._bm25 = None
            return

        self._tokenized_corpus = [
            self.tokenize(
                record.get(
                    "content",
                    "",
                )
            )
            for record in self._records
        ]

        if any(
            not tokens
            for tokens in self._tokenized_corpus
        ):
            raise BM25Error(
                "Persisted BM25 record contains no retrievable tokens."
            )

        try:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )
        except Exception as exc:
            raise BM25Error(
                f"Unable to rebuild persisted BM25 model: {exc}"
            ) from exc

    def _persist_records(self) -> None:
        """Persist BM25 corpus records to JSON."""

        try:
            self.records_path.write_text(
                json.dumps(
                    self._records,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            raise BM25Error(
                f"Unable to persist BM25 records: {exc}"
            ) from exc

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_chunk(
        chunk: DocumentChunk,
    ) -> None:
        if not chunk.chunk_id:
            raise BM25Error(
                "Cannot index a chunk without chunk_id."
            )

        if not chunk.content or not chunk.content.strip():
            raise BM25Error(
                f"Chunk '{chunk.chunk_id}' has empty content."
            )

    @staticmethod
    def _serializable_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert metadata to JSON-safe primitive values."""

        result: dict[str, Any] = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                result[key] = value

            elif isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                result[key] = ",".join(
                    str(item)
                    for item in value
                )

            elif isinstance(
                value,
                dict,
            ):
                result[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            else:
                result[key] = str(value)

        return result

    @staticmethod
    def _audience_from_metadata(
        metadata: dict[str, Any],
    ) -> list[str]:
        value = metadata.get(
            "audience"
        )

        if not value:
            return []

        if isinstance(
            value,
            list,
        ):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        return [
            item.strip()
            for item in str(
                value
            ).split(",")
            if item.strip()
        ]
