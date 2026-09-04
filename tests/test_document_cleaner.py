"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_document_cleaner.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Unit tests for conservative ParsedSection text normalization.
================================================================================
"""

from src.ingestion.document_cleaner import DocumentCleaner
from src.schemas.document import ParsedSection


def _section(
    content: str,
    title: str = "Standard Batch Formulation",
) -> ParsedSection:
    return ParsedSection(
        section_id="ON-PROD-0001-S001",
        document_id="ON-PROD-0001",
        title=title,
        content=content,
        section_index=1,
        file_name="Classic_Thekua_Product_Guide.docx",
        domain="products",
        access_level="confidential",
        audience=[
            "production_employee",
            "manager",
        ],
        language="en",
        metadata={
            "version": "2.0",
        },
    )


def test_repeated_spaces_are_normalized() -> None:
    section = _section(
        "Wheat   Flour     500 g"
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert cleaned.content == (
        "Wheat Flour 500 g"
    )


def test_non_breaking_spaces_are_normalized() -> None:
    section = _section(
        "Ghee\u00a0(Moin)\u00a075 g"
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert cleaned.content == (
        "Ghee (Moin) 75 g"
    )


def test_zero_width_characters_are_removed() -> None:
    section = _section(
        "Classic\u200b Thekua"
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert cleaned.content == (
        "Classic Thekua"
    )


def test_table_delimiters_and_values_are_preserved() -> None:
    section = _section(
        "Ingredient   |   Quantity\n"
        "Wheat Flour  |  500 g\n"
        "Ghee (Moin)  |  75 g"
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert cleaned.content == (
        "Ingredient | Quantity\n"
        "Wheat Flour | 500 g\n"
        "Ghee (Moin) | 75 g"
    )


def test_numeric_values_are_not_changed() -> None:
    section = _section(
        "Wheat Flour | 500 g | 100%\n"
        "Sugar | 250 g | 50%\n"
        "Ghee | 75 g | 15%"
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert "500 g | 100%" in cleaned.content
    assert "250 g | 50%" in cleaned.content
    assert "75 g | 15%" in cleaned.content


def test_duplicate_blank_lines_are_collapsed() -> None:
    section = _section(
        "Line one\n\n\n\nLine two"
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert cleaned.content == (
        "Line one\n\nLine two"
    )


def test_access_metadata_is_preserved() -> None:
    section = _section(
        "Wheat Flour | 500 g"
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert cleaned.access_level == (
        "confidential"
    )

    assert cleaned.audience == [
        "production_employee",
        "manager",
    ]

    assert cleaned.document_id == (
        "ON-PROD-0001"
    )

    assert cleaned.section_id == (
        "ON-PROD-0001-S001"
    )


def test_cleaning_metadata_is_added() -> None:
    original = _section(
        "Wheat   Flour"
    )

    cleaned = DocumentCleaner().clean_section(
        original
    )

    assert cleaned.metadata["cleaned"] is True

    assert (
        cleaned.metadata[
            "original_character_count"
        ]
        == len(original.content)
    )

    assert (
        cleaned.metadata[
            "cleaned_character_count"
        ]
        == len(cleaned.content)
    )


def test_original_section_is_not_modified() -> None:
    original = _section(
        "Wheat   Flour"
    )

    cleaned = DocumentCleaner().clean_section(
        original
    )

    assert original.content == (
        "Wheat   Flour"
    )

    assert cleaned.content == (
        "Wheat Flour"
    )


def test_title_is_cleaned_without_changing_meaning() -> None:
    section = _section(
        "Content",
        title=(
            "Preparation\u00a0 /   "
            "Cooking Process"
        ),
    )

    cleaned = DocumentCleaner().clean_section(
        section
    )

    assert cleaned.title == (
        "Preparation / Cooking Process"
    )
