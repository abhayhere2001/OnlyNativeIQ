"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_citation_builder.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Unit tests for trusted citation construction from grounded evidence.
================================================================================
"""

from src.generation.citation_builder import (
    CitationBuilder,
)

from src.generation.grounding import (
    GroundedEvidence,
    GroundingContext,
)


# =============================================================================
# Helper
# =============================================================================

def _evidence(
    evidence_id: str,
    chunk_id: str,
    file_name: str | None = (
        "Classic_Thekua_Product_Guide.docx"
    ),
    section: str | None = (
        "Standard Batch Formulation"
    ),
    document_id: str | None = (
        "ON-PROD-0001"
    ),
    content: str = (
        "Ghee (Moin) | 75 | 15.0%"
    ),
    rank: int = 1,
) -> GroundedEvidence:

    return GroundedEvidence(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=file_name,
        section=section,
        content=content,
        access_level="confidential",
        score=5.0,
        rank=rank,
    )


# =============================================================================
# Empty context
# =============================================================================

def test_empty_grounding_context_returns_no_citations() -> None:

    builder = CitationBuilder()

    context = GroundingContext(
        evidence=[],
        context_text="",
        has_evidence=False,
        evidence_count=0,
    )

    citations = builder.build(
        context
    )

    assert citations == []


# =============================================================================
# Single citation
# =============================================================================

def test_single_evidence_creates_single_citation() -> None:

    builder = CitationBuilder()

    evidence = _evidence(
        "SOURCE-1",
        "ON-PROD-0001-S007-C001",
    )

    context = GroundingContext(
        evidence=[evidence],
        context_text="context",
        has_evidence=True,
        evidence_count=1,
    )

    citations = builder.build(
        context
    )

    assert len(citations) == 1

    citation = citations[0]

    assert (
        citation.citation_number
        == 1
    )

    assert (
        citation.evidence_id
        == "SOURCE-1"
    )

    assert (
        citation.document_id
        == "ON-PROD-0001"
    )

    assert (
        citation.file_name
        == (
            "Classic_Thekua_"
            "Product_Guide.docx"
        )
    )

    assert (
        citation.section
        == "Standard Batch Formulation"
    )

    assert (
        citation.chunk_id
        == "ON-PROD-0001-S007-C001"
    )


# =============================================================================
# Numbering
# =============================================================================

def test_multiple_citations_are_numbered_sequentially() -> None:

    builder = CitationBuilder()

    evidence = [
        _evidence(
            "SOURCE-1",
            "C1",
            rank=1,
        ),
        _evidence(
            "SOURCE-2",
            "C2",
            rank=2,
        ),
        _evidence(
            "SOURCE-3",
            "C3",
            rank=3,
        ),
    ]

    context = GroundingContext(
        evidence=evidence,
        context_text="context",
        has_evidence=True,
        evidence_count=3,
    )

    citations = builder.build(
        context
    )

    assert [
        citation.citation_number
        for citation in citations
    ] == [
        1,
        2,
        3,
    ]


# =============================================================================
# Deduplication
# =============================================================================

def test_duplicate_evidence_is_not_cited_twice() -> None:

    builder = CitationBuilder()

    evidence = [
        _evidence(
            "SOURCE-1",
            "C1",
        ),
        _evidence(
            "SOURCE-2",
            "C1",
        ),
    ]

    context = GroundingContext(
        evidence=evidence,
        context_text="context",
        has_evidence=True,
        evidence_count=2,
    )

    citations = builder.build(
        context
    )

    assert len(citations) == 1


# =============================================================================
# Display text
# =============================================================================

def test_display_text_contains_trusted_source_metadata() -> None:

    builder = CitationBuilder()

    evidence = _evidence(
        "SOURCE-1",
        "ON-PROD-0001-S007-C001",
    )

    context = GroundingContext(
        evidence=[evidence],
        context_text="context",
        has_evidence=True,
        evidence_count=1,
    )

    citation = builder.build(
        context
    )[0]

    assert (
        "[1]"
        in citation.display_text
    )

    assert (
        "Classic_Thekua_Product_Guide.docx"
        in citation.display_text
    )

    assert (
        "Standard Batch Formulation"
        in citation.display_text
    )

    assert (
        "ON-PROD-0001-S007-C001"
        in citation.display_text
    )


# =============================================================================
# Missing optional metadata
# =============================================================================

def test_missing_filename_uses_safe_fallback() -> None:

    builder = CitationBuilder()

    evidence = _evidence(
        "SOURCE-1",
        "C1",
        file_name=None,
    )

    context = GroundingContext(
        evidence=[evidence],
        context_text="context",
        has_evidence=True,
        evidence_count=1,
    )

    citation = builder.build(
        context
    )[0]

    assert (
        citation.file_name
        == "Unknown Source"
    )


def test_missing_section_does_not_break_citation() -> None:

    builder = CitationBuilder()

    evidence = _evidence(
        "SOURCE-1",
        "C1",
        section=None,
    )

    context = GroundingContext(
        evidence=[evidence],
        context_text="context",
        has_evidence=True,
        evidence_count=1,
    )

    citation = builder.build(
        context
    )[0]

    assert (
        citation.section
        is None
    )

    assert (
        "C1"
        in citation.display_text
    )


# =============================================================================
# Sources formatting
# =============================================================================

def test_format_sources_builds_sources_section() -> None:

    builder = CitationBuilder()

    evidence = [
        _evidence(
            "SOURCE-1",
            "C1",
        ),
        _evidence(
            "SOURCE-2",
            "C2",
        ),
    ]

    context = GroundingContext(
        evidence=evidence,
        context_text="context",
        has_evidence=True,
        evidence_count=2,
    )

    citations = builder.build(
        context
    )

    sources = (
        builder.format_sources(
            citations
        )
    )

    assert (
        sources.startswith(
            "Sources:"
        )
    )

    assert (
        "[1]"
        in sources
    )

    assert (
        "[2]"
        in sources
    )


def test_format_sources_with_no_citations_returns_empty_string() -> None:

    assert (
        CitationBuilder.format_sources(
            []
        )
        == ""
    )