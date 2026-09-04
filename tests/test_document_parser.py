"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_document_parser.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Focused tests for product-guide structure and access classification.
================================================================================
"""

from src.ingestion.document_parser import (
    DocumentParser,
)
from src.schemas.document import (
    SourceDocument,
)


def _product(
    content: str,
) -> SourceDocument:

    return SourceDocument(
        document_id="ON-PROD-0001",
        file_name=(
            "Classic_Thekua_Product_Guide.docx"
        ),
        file_path=(
            "knowledge_base/products/"
            "Classic_Thekua_Product_Guide.docx"
        ),
        file_extension=".docx",
        domain="products",
        content=content,
        document_type="product_guide",
        version="2.0",
        status="approved",
        access_level="mixed",
        audience=[
            "customer",
            "employee",
            "production_employee",
            "manager",
        ],
        language="en",
    )


def test_formulation_table_starts_confidential_section() -> None:
    document = _product(
        """
Description
Classic Thekua.

Price
INR 160

Shelf Life
30 days

Ingredient | Quantity (g) | Ratio vs. Base Flour
Wheat Flour | 500 | 100.0%
Sugar | 250 | 50.0%
Ghee (Moin) | 75 | 15.0%

Preparation / Cooking Process
Mix wheat flour with ghee as moin.
"""
    )

    sections = (
        DocumentParser().parse_document(
            document
        )
    )

    titles = [
        section.title
        for section in sections
    ]

    assert (
        "Standard Batch Formulation"
        in titles
    )

    formula = next(
        section
        for section in sections
        if section.title
        == "Standard Batch Formulation"
    )

    assert formula.access_level == (
        "confidential"
    )

    assert (
        "Ghee (Moin) | 75 | 15.0%"
        in formula.content
    )


def test_public_product_fact_stays_public() -> None:
    document = _product(
        """
Shelf Life
30 days
"""
    )

    section = (
        DocumentParser()
        .parse_document(
            document
        )[0]
    )

    assert section.title == (
        "Shelf Life"
    )

    assert section.access_level == (
        "public"
    )


def test_preparation_is_confidential() -> None:
    document = _product(
        """
Preparation / Cooking Process
Mix wheat flour with ghee.
"""
    )

    section = (
        DocumentParser()
        .parse_document(
            document
        )[0]
    )

    assert section.access_level == (
        "confidential"
    )
