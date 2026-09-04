"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Corpus Validation
File         : validate_all_product_guides.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Validates the structure and section-level security classification of all
    product DOCX files in the OnlyNativeIQ knowledge base.

Validation Goals:
    - Description must be public.
    - Product Facts must be public.
    - Standard Batch Formulation must be confidential.
    - Preparation / Cooking Process must be confidential.
    - Production Resource must be confidential.
    - Unstandardized Process Parameters must be internal.
    - Public product facts such as Shelf Life, Price, Storage and Allergens
      must remain inside Product Facts.
    - Ingredient/formulation data must never appear inside public sections.

Usage:
    python scripts/validate_all_product_guides.py
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# OnlyNativeIQ imports
# =============================================================================

from src.ingestion.document_loader import (
    DocumentLoader,
)

from src.ingestion.metadata_manager import (
    MetadataManager,
)

from src.ingestion.document_parser import (
    DocumentParser,
)

from src.ingestion.document_cleaner import (
    DocumentCleaner,
)

from src.ingestion.chunker import (
    DocumentChunker,
)

from src.utils.config_loader import (
    ConfigLoader,
)


# =============================================================================
# Expected section security
# =============================================================================

EXPECTED_SECTION_ACCESS = {
    "Description":
        "public",

    "Product Facts":
        "public",

    "Standard Batch Formulation":
        "confidential",

    "Preparation / Cooking Process":
        "confidential",

    "Production Resource":
        "confidential",

    "Unstandardized Process Parameters":
        "internal",
}


# =============================================================================
# Required public product facts
# =============================================================================

REQUIRED_PRODUCT_FACTS = [
    "Document ID",
    "Category",
    "Pack Size",
    "Price",
    "Shelf Life",
    "Storage",
    "Allergens",
]


# =============================================================================
# Sensitive formulation indicators
# =============================================================================

SENSITIVE_FORMULATION_TERMS = [
    "ingredient | quantity",
    "ratio vs. base flour",
    "ghee (moin)",
]


# =============================================================================
# Helpers
# =============================================================================

def separator(
    character: str = "=",
    width: int = 110,
) -> None:

    print(
        character * width
    )


def heading(
    text: str,
) -> None:

    print()

    separator()

    print(
        text
    )

    separator()


def normalize(
    value: str,
) -> str:

    return (
        value
        .strip()
        .lower()
    )


# =============================================================================
# Validation
# =============================================================================

def validate_document(
    document,
    parser: DocumentParser,
    cleaner: DocumentCleaner,
    chunker: DocumentChunker,
) -> list[str]:
    """
    Validate one product document.

    Returns
    -------
    list[str]
        Validation errors. Empty means PASS.
    """

    errors: list[str] = []

    sections = parser.parse_document(
        document
    )

    cleaned_sections = (
        cleaner.clean_sections(
            sections
        )
    )

    chunks = (
        chunker.chunk_sections(
            cleaned_sections
        )
    )

    # -------------------------------------------------------------------------
    # Build section lookup
    # -------------------------------------------------------------------------

    section_lookup = {
        section.title:
            section
        for section
        in sections
    }

    # -------------------------------------------------------------------------
    # Expected sections
    # -------------------------------------------------------------------------

    for (
        section_name,
        expected_access,
    ) in (
        EXPECTED_SECTION_ACCESS.items()
    ):

        section = section_lookup.get(
            section_name
        )

        if section is None:

            errors.append(
                f"Missing section: "
                f"{section_name}"
            )

            continue

        actual_access = (
            section.access_level
            or ""
        ).lower()

        if (
            actual_access
            != expected_access
        ):

            errors.append(
                f"Section '{section_name}' "
                f"has access '{actual_access}', "
                f"expected '{expected_access}'."
            )

    # -------------------------------------------------------------------------
    # Product Facts validation
    # -------------------------------------------------------------------------

    product_facts = (
        section_lookup.get(
            "Product Facts"
        )
    )

    if product_facts is not None:

        facts_content = (
            product_facts.content
            or ""
        )

        facts_lower = (
            facts_content.lower()
        )

        for required_fact in (
            REQUIRED_PRODUCT_FACTS
        ):

            if (
                required_fact.lower()
                not in facts_lower
            ):

                errors.append(
                    f"Product Facts missing "
                    f"'{required_fact}'."
                )

        # ---------------------------------------------------------------------
        # Confidential formulation must not leak into Product Facts
        # ---------------------------------------------------------------------

        for sensitive_term in (
            SENSITIVE_FORMULATION_TERMS
        ):

            if (
                sensitive_term
                in facts_lower
            ):

                errors.append(
                    "Sensitive formulation data "
                    f"found inside Product Facts: "
                    f"'{sensitive_term}'."
                )

    # -------------------------------------------------------------------------
    # Standard Batch Formulation validation
    # -------------------------------------------------------------------------

    formulation = (
        section_lookup.get(
            "Standard Batch Formulation"
        )
    )

    if formulation is not None:

        formulation_content = (
            formulation.content
            or ""
        ).lower()

        if (
            "ingredient"
            not in formulation_content
        ):

            errors.append(
                "Standard Batch Formulation "
                "does not contain an ingredient table."
            )

    # -------------------------------------------------------------------------
    # Chunk-level security validation
    # -------------------------------------------------------------------------

    for chunk in chunks:

        section_name = (
            chunk.section
            or ""
        )

        chunk_access = (
            chunk.access_level
            or ""
        ).lower()

        expected_access = (
            EXPECTED_SECTION_ACCESS.get(
                section_name
            )
        )

        if (
            expected_access
            and chunk_access
            != expected_access
        ):

            errors.append(
                f"Chunk '{chunk.chunk_id}' "
                f"for section '{section_name}' "
                f"has access '{chunk_access}', "
                f"expected '{expected_access}'."
            )

        # ---------------------------------------------------------------------
        # No formulation data in public chunks
        # ---------------------------------------------------------------------

        if (
            chunk_access
            == "public"
        ):

            content_lower = (
                chunk.content
                or ""
            ).lower()

            if (
                "ingredient | quantity"
                in content_lower
            ):

                errors.append(
                    f"Public chunk "
                    f"'{chunk.chunk_id}' "
                    "contains formulation table."
                )

    return errors


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    config = ConfigLoader(
        settings_path=(
            "config/settings.yaml"
        ),
        project_root=(
            PROJECT_ROOT
        ),
    )

    # -------------------------------------------------------------------------
    # Components
    # -------------------------------------------------------------------------

    loader = DocumentLoader(
        config.resolve_path(
            "knowledge_base.root"
        )
    )

    metadata_manager = (
        MetadataManager(
            registry_path=(
                config.resolve_path(
                    "metadata.registry_path"
                )
            ),
            schema_path=(
                config.resolve_path(
                    "metadata.schema_path"
                )
            ),
            strict=(
                config.get(
                    "metadata.strict_validation",
                    default=True,
                )
            ),
        )
    )

    parser = DocumentParser()

    cleaner = DocumentCleaner()

    chunker = DocumentChunker(
        chunk_size=(
            config.get(
                "chunking.chunk_size",
                required=True,
            )
        ),
        chunk_overlap=(
            config.get(
                "chunking.chunk_overlap",
                required=True,
            )
        ),
        include_section_title=(
            config.get(
                "chunking.include_section_title",
                default=True,
            )
        ),
        include_document_context=True,
    )

    # -------------------------------------------------------------------------
    # Load all documents
    # -------------------------------------------------------------------------

    documents = (
        loader.load_all()
    )

    # -------------------------------------------------------------------------
    # Only product DOCX files
    # -------------------------------------------------------------------------

    product_documents = [
        document
        for document
        in documents
        if (
            document.domain
            == "products"
            and document.file_extension
            == ".docx"
        )
    ]

    if not product_documents:

        print(
            "No product DOCX files found."
        )

        return

    # -------------------------------------------------------------------------
    # Metadata enrichment
    # -------------------------------------------------------------------------

    product_documents = (
        metadata_manager
        .enrich_documents(
            product_documents
        )
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    heading(
        "OnlyNativeIQ - Product Guide Validation"
    )

    print(
        f"Product guides found: "
        f"{len(product_documents)}"
    )

    passed = 0

    failed = 0

    all_errors: dict[
        str,
        list[str],
    ] = {}

    for document in (
        product_documents
    ):

        errors = validate_document(
            document=document,
            parser=parser,
            cleaner=cleaner,
            chunker=chunker,
        )

        separator(
            "-",
            110,
        )

        print(
            f"File: "
            f"{document.file_name}"
        )

        if not errors:

            passed += 1

            print(
                "Status: PASS"
            )

        else:

            failed += 1

            all_errors[
                document.file_name
            ] = errors

            print(
                "Status: FAIL"
            )

            for error in errors:

                print(
                    f"  - {error}"
                )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    heading(
        "VALIDATION SUMMARY"
    )

    print(
        f"Total product guides : "
        f"{len(product_documents)}"
    )

    print(
        f"Passed               : "
        f"{passed}"
    )

    print(
        f"Failed               : "
        f"{failed}"
    )

    print()

    if failed == 0:

        separator()

        print(
            "ALL PRODUCT GUIDES PASSED VALIDATION"
        )

        separator()

        return

    separator(
        "!",
        110,
    )

    print(
        "PRODUCT GUIDE VALIDATION FAILED"
    )

    separator(
        "!",
        110,
    )

    print()

    for (
        file_name,
        errors,
    ) in all_errors.items():

        print(
            file_name
        )

        for error in errors:

            print(
                f"  - {error}"
            )

        print()

    raise RuntimeError(
        f"{failed} product guide(s) "
        "failed structural/security validation."
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()