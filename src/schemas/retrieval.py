"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Shared Retrieval Schemas
File         : retrieval.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Defines retrieval result structures shared by vector, BM25, hybrid and
    reranking components.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    """
    Represents one retrieved knowledge chunk.
    """

    chunk_id: str
    content: str
    score: float
    rank: int

    document_id: str | None = None
    file_name: str | None = None
    domain: str | None = None
    section: str | None = None
    page_number: int | None = None

    access_level: str | None = None
    audience: list[str] = field(default_factory=list)

    source: str = "vector"
    metadata: dict[str, Any] = field(default_factory=dict)
