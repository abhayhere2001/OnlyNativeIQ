"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Shared Document Schemas
File         : document.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Defines shared source-document, parsed-section, chunk and embedding
    structures used across ingestion, retrieval and evaluation.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SourceDocument:
    document_id: str | None
    file_name: str
    file_path: str
    file_extension: str
    domain: str

    content: str = ""
    document_type: str | None = None
    version: str | None = None
    status: str | None = None
    access_level: str | None = None
    audience: list[str] = field(default_factory=list)
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.file_path)

    @property
    def is_empty(self) -> bool:
        return not self.content or not self.content.strip()


@dataclass(slots=True)
class ParsedSection:
    section_id: str
    document_id: str
    title: str
    content: str
    section_index: int
    file_name: str
    domain: str

    page_number: int | None = None
    section_type: str | None = None
    access_level: str | None = None
    audience: list[str] = field(default_factory=list)
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.content or not self.content.strip()


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    file_name: str
    domain: str
    chunk_index: int

    page_number: int | None = None
    section: str | None = None
    subsection: str | None = None
    document_type: str | None = None
    version: str | None = None
    status: str | None = None
    access_level: str | None = None
    audience: list[str] = field(default_factory=list)
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "domain": self.domain,
            "chunk_index": self.chunk_index,
            "language": self.language,
        }

        optional_values = {
            "page_number": self.page_number,
            "section": self.section,
            "subsection": self.subsection,
            "document_type": self.document_type,
            "version": self.version,
            "status": self.status,
            "access_level": self.access_level,
        }

        for key, value in optional_values.items():
            if value is not None:
                result[key] = value

        if self.audience:
            result["audience"] = ",".join(self.audience)

        result.update(self.metadata)
        return result


@dataclass(slots=True)
class EmbeddedChunk:
    """
    Represents one retrieval chunk plus its dense embedding vector.
    """

    chunk: DocumentChunk
    embedding: list[float]
    model_name: str
    embedding_dimension: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id
