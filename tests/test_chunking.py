"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_chunking.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Focused tests for contextualized retrieval chunk text.
================================================================================
"""

from src.ingestion.chunker import (
    DocumentChunker,
)
from src.schemas.document import (
    ParsedSection,
)


def test_chunk_contains_document_and_section_context() -> None:
    section = ParsedSection(
        section_id="ON-PROD-0001-S003",
        document_id="ON-PROD-0001",
        title="Standard Batch Formulation",
        content=(
            "Ingredient | Quantity\n"
            "Ghee (Moin) | 75"
        ),
        section_index=3,
        file_name=(
            "Classic_Thekua_Product_Guide.docx"
        ),
        domain="products",
        section_type=(
            "ingredients_or_formula"
        ),
        access_level="confidential",
        audience=[
            "production_employee",
            "manager",
        ],
        language="en",
    )

    chunk = (
        DocumentChunker(
            include_document_context=True,
            include_section_title=True,
        )
        .chunk_section(
            section
        )[0]
    )

    assert chunk.content.startswith(
        "Document: Classic Thekua Product Guide\n"
        "Section: Standard Batch Formulation"
    )

    assert (
        "Ghee (Moin) | 75"
        in chunk.content
    )

    assert chunk.access_level == (
        "confidential"
    )
