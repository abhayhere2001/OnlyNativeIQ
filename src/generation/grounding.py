"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Generation
File         : grounding.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Builds and validates grounded generation context from authorized retrieval
    results.

Responsibilities:
    - Accept only retrieval results already authorized by the security layer.
    - Remove empty or unusable evidence.
    - Preserve source/chunk identity.
    - Build structured LLM context.
    - Detect when generation should not be attempted.
    - Prevent accidental mixing of retrieval metadata and instructions.

Important:
    This module does NOT perform authorization.
    Authorization belongs to the security/retrieval layer.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas.retrieval import (
    RetrievalResult,
)


# =============================================================================
# Exceptions
# =============================================================================

class GroundingError(RuntimeError):
    """
    Raised when retrieved evidence cannot safely be prepared for generation.
    """


# =============================================================================
# Grounded evidence
# =============================================================================

@dataclass(frozen=True, slots=True)
class GroundedEvidence:
    """
    One retrieval result transformed into generation evidence.
    """

    evidence_id: str

    chunk_id: str

    document_id: str | None

    file_name: str | None

    section: str | None

    content: str

    access_level: str | None

    score: float

    rank: int


# =============================================================================
# Grounding context
# =============================================================================

@dataclass(slots=True)
class GroundingContext:
    """
    Complete evidence package supplied to the generation layer.
    """

    evidence: list[
        GroundedEvidence
    ] = field(
        default_factory=list
    )

    context_text: str = ""

    has_evidence: bool = False

    evidence_count: int = 0


# =============================================================================
# Grounding builder
# =============================================================================

class GroundingBuilder:
    """
    Convert authorized retrieval results into controlled LLM context.
    """

    def __init__(
        self,
        max_context_characters: int = 12000,
    ) -> None:

        if max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than zero."
            )

        self.max_context_characters = (
            max_context_characters
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def build(
        self,
        results: list[
            RetrievalResult
        ],
    ) -> GroundingContext:
        """
        Build grounded context from authorized retrieval results.

        The caller must pass ONLY results returned by the retrieval pipeline's
        final authorized result set.
        """

        if not results:

            return GroundingContext(
                evidence=[],
                context_text="",
                has_evidence=False,
                evidence_count=0,
            )

        evidence_items: list[
            GroundedEvidence
        ] = []

        context_blocks: list[
            str
        ] = []

        current_length = 0

        for result in results:

            content = (
                result.content or ""
            ).strip()

            if not content:
                continue

            evidence_number = (
                len(evidence_items) + 1
            )

            evidence_id = (
                f"SOURCE-{evidence_number}"
            )

            evidence = GroundedEvidence(
                evidence_id=evidence_id,
                chunk_id=result.chunk_id,
                document_id=(
                    result.document_id
                ),
                file_name=(
                    result.file_name
                ),
                section=(
                    result.section
                ),
                content=content,
                access_level=(
                    result.access_level
                ),
                score=float(
                    result.score
                ),
                rank=int(
                    result.rank
                ),
            )

            block = (
                self._format_evidence(
                    evidence
                )
            )

            projected_length = (
                current_length
                + len(block)
            )

            if (
                projected_length
                > self.max_context_characters
            ):
                break

            evidence_items.append(
                evidence
            )

            context_blocks.append(
                block
            )

            current_length = (
                projected_length
            )

        if not evidence_items:

            return GroundingContext(
                evidence=[],
                context_text="",
                has_evidence=False,
                evidence_count=0,
            )

        context_text = (
            "\n\n".join(
                context_blocks
            )
        )

        return GroundingContext(
            evidence=evidence_items,
            context_text=context_text,
            has_evidence=True,
            evidence_count=len(
                evidence_items
            ),
        )

    # =========================================================================
    # Evidence formatting
    # =========================================================================

    @staticmethod
    def _format_evidence(
        evidence: GroundedEvidence,
    ) -> str:
        """
        Convert one evidence object into a clearly delimited context block.
        """

        file_name = (
            evidence.file_name
            or "Unknown Source"
        )

        section = (
            evidence.section
            or "Unknown Section"
        )

        document_id = (
            evidence.document_id
            or "Unknown"
        )

        return (
            f"[{evidence.evidence_id}]\n"
            f"Document ID: {document_id}\n"
            f"File: {file_name}\n"
            f"Section: {section}\n"
            f"Chunk ID: {evidence.chunk_id}\n"
            f"Content:\n"
            f"{evidence.content}\n"
            f"[END {evidence.evidence_id}]"
        )