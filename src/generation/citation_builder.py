"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Generation
File         : citation_builder.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Builds trusted source citations from grounded OnlyNativeIQ evidence.

Responsibilities:
    - Convert GroundedEvidence into structured citations.
    - Preserve evidence, document, section and chunk identity.
    - Generate human-readable source references.
    - Deduplicate repeated citations.
    - Never allow the LLM to invent source metadata.

Design Principle:
    Citations are created from retrieval metadata, not from generated text.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.grounding import (
    GroundedEvidence,
    GroundingContext,
)


# =============================================================================
# Citation model
# =============================================================================

@dataclass(frozen=True, slots=True)
class Citation:
    """
    Trusted citation derived from grounded retrieval evidence.
    """

    citation_number: int

    evidence_id: str

    document_id: str | None

    file_name: str

    section: str | None

    chunk_id: str

    display_text: str


# =============================================================================
# Citation builder
# =============================================================================

class CitationBuilder:
    """
    Build trusted citations from GroundingContext.

    The LLM must never provide the filename, document ID, section name,
    or chunk ID used here.
    """

    def build(
        self,
        grounding_context: GroundingContext,
    ) -> list[Citation]:
        """
        Build citations from grounded evidence.

        Duplicate source/section/chunk combinations are removed while
        preserving retrieval order.
        """

        if not grounding_context.has_evidence:
            return []

        citations: list[Citation] = []

        seen: set[
            tuple[
                str | None,
                str,
                str | None,
                str,
            ]
        ] = set()

        for evidence in grounding_context.evidence:

            file_name = (
                evidence.file_name
                or "Unknown Source"
            )

            key = (
                evidence.document_id,
                file_name,
                evidence.section,
                evidence.chunk_id,
            )

            if key in seen:
                continue

            seen.add(key)

            citation_number = (
                len(citations) + 1
            )

            display_text = (
                self._build_display_text(
                    citation_number=(
                        citation_number
                    ),
                    evidence=evidence,
                )
            )

            citations.append(
                Citation(
                    citation_number=(
                        citation_number
                    ),
                    evidence_id=(
                        evidence.evidence_id
                    ),
                    document_id=(
                        evidence.document_id
                    ),
                    file_name=file_name,
                    section=(
                        evidence.section
                    ),
                    chunk_id=(
                        evidence.chunk_id
                    ),
                    display_text=(
                        display_text
                    ),
                )
            )

        return citations

    # =========================================================================
    # Human-readable citation
    # =========================================================================

    @staticmethod
    def _build_display_text(
        citation_number: int,
        evidence: GroundedEvidence,
    ) -> str:
        """
        Build a human-readable citation from trusted metadata.
        """

        file_name = (
            evidence.file_name
            or "Unknown Source"
        )

        parts = [
            f"[{citation_number}]",
            file_name,
        ]

        if evidence.section:
            parts.append(
                f"Section: {evidence.section}"
            )

        parts.append(
            f"Chunk: {evidence.chunk_id}"
        )

        return " | ".join(parts)

    # =========================================================================
    # Sources section
    # =========================================================================

    @staticmethod
    def format_sources(
        citations: list[Citation],
    ) -> str:
        """
        Build a user-readable Sources section.

        Returns an empty string when there are no citations.
        """

        if not citations:
            return ""

        lines = [
            "Sources:"
        ]

        for citation in citations:
            lines.append(
                citation.display_text
            )

        return "\n".join(lines)