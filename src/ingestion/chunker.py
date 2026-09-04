"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Document Ingestion
File         : chunker.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Converts cleaned ParsedSection objects into retrieval-ready DocumentChunk
    objects.

Key Improvement:
    Retrieval text is contextualized with document/product identity and section
    name before embedding/BM25 indexing.

Important:
    - Source content is preserved.
    - Security metadata is preserved.
    - No chunk merges content across logical sections.
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.schemas.document import (
    DocumentChunk,
    ParsedSection,
)


logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100


class ChunkingError(RuntimeError):
    """Raised when a parsed section cannot be chunked safely."""


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        include_section_title: bool = True,
        include_document_context: bool = True,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = (
            chunk_overlap
        )
        self.include_section_title = (
            include_section_title
        )
        self.include_document_context = (
            include_document_context
        )

        self._text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "; ",
                    ", ",
                    " ",
                    "",
                ],
                keep_separator=True,
            )
        )

    def chunk_sections(
        self,
        sections: Iterable[
            ParsedSection
        ],
    ) -> list[
        DocumentChunk
    ]:

        chunks: list[
            DocumentChunk
        ] = []

        for section in sections:
            chunks.extend(
                self.chunk_section(
                    section
                )
            )

        return chunks

    def chunk_section(
        self,
        section: ParsedSection,
    ) -> list[
        DocumentChunk
    ]:

        self._validate_section(
            section
        )

        if self._is_table_like(
            section
        ):
            pieces = (
                self._split_table_section(
                    section
                )
            )
            strategy = (
                "table_row_aware"
            )
        else:
            pieces = (
                self._split_text_section(
                    section
                )
            )
            strategy = (
                "recursive_character"
            )

        chunks: list[
            DocumentChunk
        ] = []

        for chunk_index, piece in enumerate(
            pieces,
            start=1,
        ):
            if not piece.strip():
                continue

            metadata = dict(
                section.metadata
            )

            metadata.update(
                {
                    "section_id":
                        section.section_id,
                    "section_index":
                        section.section_index,
                    "section_type":
                        section.section_type,
                    "chunk_size_config":
                        self.chunk_size,
                    "chunk_overlap_config":
                        self.chunk_overlap,
                    "chunking_strategy":
                        strategy,
                    "retrieval_context_added":
                        self.include_document_context,
                }
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{section.section_id}-C"
                        f"{chunk_index:03d}"
                    ),
                    document_id=(
                        section.document_id
                    ),
                    content=(
                        self._build_retrieval_content(
                            section,
                            piece.strip(),
                        )
                    ),
                    file_name=(
                        section.file_name
                    ),
                    domain=(
                        section.domain
                    ),
                    chunk_index=chunk_index,
                    page_number=(
                        section.page_number
                    ),
                    section=(
                        section.title
                    ),
                    document_type=(
                        metadata.get(
                            "document_type"
                        )
                    ),
                    version=(
                        metadata.get(
                            "version"
                        )
                    ),
                    status=(
                        metadata.get(
                            "status"
                        )
                    ),
                    access_level=(
                        section.access_level
                    ),
                    audience=list(
                        section.audience
                    ),
                    language=(
                        section.language
                    ),
                    metadata=metadata,
                )
            )

        return chunks

    # ---------------------------------------------------------------------
    # Retrieval-context construction
    # ---------------------------------------------------------------------

    def _build_retrieval_content(
        self,
        section: ParsedSection,
        content: str,
    ) -> str:
        """
        Prefix structural identity to the text used for BM25/embeddings.

        Example:
            Document: Classic Thekua Product Guide
            Section: Standard Batch Formulation

            Ingredient | Quantity ...
        """

        prefix: list[str] = []

        if self.include_document_context:
            document_label = (
                Path(
                    section.file_name
                )
                .stem
                .replace("_", " ")
            )

            prefix.append(
                f"Document: {document_label}"
            )

        if (
            self.include_section_title
            and section.title
        ):
            prefix.append(
                f"Section: {section.title}"
            )

        if not prefix:
            return content

        return (
            "\n".join(prefix)
            + "\n\n"
            + content
        )

    # ---------------------------------------------------------------------
    # Splitting
    # ---------------------------------------------------------------------

    def _split_text_section(
        self,
        section: ParsedSection,
    ) -> list[str]:

        content = (
            section.content.strip()
        )

        if (
            len(content)
            <= self.chunk_size
        ):
            return [
                content
            ]

        return [
            piece.strip()
            for piece
            in self._text_splitter.split_text(
                content
            )
            if piece.strip()
        ]

    def _split_table_section(
        self,
        section: ParsedSection,
    ) -> list[str]:

        rows = [
            line.strip()
            for line
            in section.content.splitlines()
            if line.strip()
        ]

        if not rows:
            return []

        if (
            len(section.content)
            <= self.chunk_size
        ):
            return [
                section.content.strip()
            ]

        header = (
            rows[0]
            if "|" in rows[0]
            else None
        )

        data_rows = (
            rows[1:]
            if header
            else rows
        )

        chunks: list[str] = []
        current_rows: list[str] = []

        for row in data_rows:
            candidate_rows = (
                current_rows
                + [row]
            )

            candidate = (
                self._compose_table_chunk(
                    header,
                    candidate_rows,
                )
            )

            if (
                len(candidate)
                <= self.chunk_size
                or not current_rows
            ):
                current_rows.append(
                    row
                )
                continue

            chunks.append(
                self._compose_table_chunk(
                    header,
                    current_rows,
                )
            )

            current_rows = (
                [current_rows[-1], row]
                if (
                    self.chunk_overlap > 0
                    and current_rows
                )
                else [row]
            )

        if current_rows:
            chunks.append(
                self._compose_table_chunk(
                    header,
                    current_rows,
                )
            )

        return chunks

    @staticmethod
    def _compose_table_chunk(
        header: str | None,
        rows: list[str],
    ) -> str:

        lines = []

        if header:
            lines.append(
                header
            )

        lines.extend(
            rows
        )

        return "\n".join(
            lines
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_table_like(
        section: ParsedSection,
    ) -> bool:

        if (
            section.section_type
            in {
                "table_or_structured_data",
                "ingredients_or_formula",
            }
        ):
            return True

        lines = [
            line
            for line
            in section.content.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        pipe_rows = sum(
            1
            for line in lines
            if "|" in line
        )

        return (
            pipe_rows >= 2
            and pipe_rows
            >= len(lines) / 2
        )

    @staticmethod
    def _validate_section(
        section: ParsedSection,
    ) -> None:

        missing = []

        if not section.section_id:
            missing.append(
                "section_id"
            )

        if not section.document_id:
            missing.append(
                "document_id"
            )

        if not section.file_name:
            missing.append(
                "file_name"
            )

        if not section.access_level:
            missing.append(
                "access_level"
            )

        if section.is_empty:
            missing.append(
                "content"
            )

        if missing:
            raise ChunkingError(
                f"Section '{section.section_id}' "
                f"cannot be chunked. "
                f"Missing/empty: "
                f"{', '.join(missing)}"
            )
