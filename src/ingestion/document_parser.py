"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Document Ingestion
File         : document_parser.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Parses metadata-enriched SourceDocument objects into logical sections.

Key Goals:
    - Explicit headings always take priority over heuristic table detection.
    - Product Facts is recognized as a public section.
    - Standard Batch Formulation is confidential.
    - Preparation / Cooking Process is confidential.
    - Production Resource is confidential.
    - Unstandardized Process Parameters is internal.
    - FAQ question headings remain supported.
    - Page provenance is preserved.
    - Ingredient-table heuristics are used only when an explicit suitable
      section is not already active.

Important:
    This parser does not perform chunk-size splitting.
    Chunking remains the responsibility of chunker.py.
================================================================================
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from src.schemas.document import (
    ParsedSection,
    SourceDocument,
)


logger = logging.getLogger(__name__)


# =============================================================================
# PAGE MARKERS
# =============================================================================

PAGE_MARKER_PATTERN = re.compile(
    r"(?m)^\[PAGE\s+(\d+)\]\s*$"
)


# =============================================================================
# SECTION DEFINITIONS
# =============================================================================

KNOWN_SECTION_HEADINGS = {
    # -------------------------------------------------------------------------
    # Public product information
    # -------------------------------------------------------------------------
    "description",
    "product description",
    "product facts",
    "product information",
    "product details",
    "category",
    "pack size",
    "price",
    "shelf life",
    "storage",
    "storage instructions",
    "allergens",
    "allergen information",
    "dietary status",
    "dietary classification",
    "preservatives",
    "claim",
    "approved claim",

    # -------------------------------------------------------------------------
    # Restricted product information
    # -------------------------------------------------------------------------
    "ingredients",
    "ingredient",
    "ingredient formula",
    "ingredient formula / ratio",
    "standard batch formulation",
    "batch formulation",
    "preparation",
    "preparation process",
    "preparation / cooking process",
    "cooking process",
    "production resource",
    "production resources",

    # -------------------------------------------------------------------------
    # Internal product information
    # -------------------------------------------------------------------------
    "unstandardized process parameters",
    "knowledge status",

    # -------------------------------------------------------------------------
    # Organization / HR
    # -------------------------------------------------------------------------
    "founder story",
    "culinary focus",
    "official tagline",
    "market and expansion",
    "mission",
    "meaning of the name",
    "working schedule",
    "leave and attendance",
    "production supervision",
    "workplace requirements",
    "confidentiality",
    "monitoring and discipline",
    "helper compensation",

    # -------------------------------------------------------------------------
    # SOP / policy
    # -------------------------------------------------------------------------
    "food hygiene",
    "quality control",
    "packaging",
    "scope",
    "purpose",
}


PUBLIC_PRODUCT_HEADINGS = {
    "description",
    "product description",
    "product facts",
    "product information",
    "product details",
    "category",
    "pack size",
    "price",
    "shelf life",
    "storage",
    "storage instructions",
    "allergens",
    "allergen information",
    "dietary status",
    "dietary classification",
    "preservatives",
    "claim",
    "approved claim",
}


CONFIDENTIAL_PRODUCT_HEADINGS = {
    "ingredients",
    "ingredient",
    "ingredient formula",
    "ingredient formula / ratio",
    "standard batch formulation",
    "batch formulation",
    "preparation",
    "preparation process",
    "preparation / cooking process",
    "cooking process",
    "production resource",
    "production resources",
}


INTERNAL_PRODUCT_HEADINGS = {
    "unstandardized process parameters",
    "knowledge status",
}


# =============================================================================
# AUDIENCE DEFINITIONS
# =============================================================================

PUBLIC_PRODUCT_AUDIENCE = [
    "customer",
    "employee",
    "production_employee",
    "manager",
]


INTERNAL_PRODUCT_AUDIENCE = [
    "employee",
    "production_employee",
    "manager",
]


CONFIDENTIAL_PRODUCT_AUDIENCE = [
    "production_employee",
    "manager",
]


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DocumentParseError(
    RuntimeError
):
    """
    Raised when a source document cannot be parsed safely.
    """


# =============================================================================
# DOCUMENT PARSER
# =============================================================================

class DocumentParser:
    """
    Convert SourceDocument objects into logical ParsedSection objects.
    """

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def parse_document(
        self,
        document: SourceDocument,
    ) -> list[ParsedSection]:
        """
        Parse one source document into logical sections.
        """

        self._validate_source_document(
            document
        )

        page_blocks = (
            self._split_pages(
                document.content
            )
        )

        parsed_sections: list[
            ParsedSection
        ] = []

        section_index = 0

        for (
            page_number,
            page_content,
        ) in page_blocks:

            raw_sections = (
                self._split_logical_sections(
                    page_content
                )
            )

            for (
                title,
                content,
            ) in raw_sections:

                title = (
                    title or ""
                ).strip()

                content = (
                    content or ""
                ).strip()

                if (
                    not title
                    and not content
                ):
                    continue

                if not title:
                    title = (
                        "Document Content"
                    )

                section_index += 1

                (
                    access_level,
                    audience,
                ) = (
                    self._resolve_section_access(
                        document=document,
                        section_title=title,
                    )
                )

                section_type = (
                    self._infer_section_type(
                        title=title,
                        content=content,
                    )
                )

                metadata = dict(
                    document.metadata
                )

                metadata.update(
                    {
                        "document_type":
                            document.document_type,

                        "version":
                            document.version,

                        "status":
                            document.status,

                        "source_section_title":
                            title,
                    }
                )

                parsed_sections.append(
                    ParsedSection(
                        section_id=(
                            self._create_section_id(
                                document=document,
                                section_index=(
                                    section_index
                                ),
                            )
                        ),

                        document_id=(
                            document.document_id
                            or ""
                        ),

                        title=title,

                        content=content,

                        section_index=(
                            section_index
                        ),

                        file_name=(
                            document.file_name
                        ),

                        domain=(
                            document.domain
                        ),

                        page_number=(
                            page_number
                        ),

                        section_type=(
                            section_type
                        ),

                        access_level=(
                            access_level
                        ),

                        audience=(
                            audience
                        ),

                        language=(
                            document.language
                        ),

                        metadata=metadata,
                    )
                )

        logger.info(
            "Parsed document '%s' into %d section(s).",
            document.file_name,
            len(parsed_sections),
        )

        return parsed_sections

    def parse_documents(
        self,
        documents: Iterable[
            SourceDocument
        ],
    ) -> list[ParsedSection]:
        """
        Parse multiple source documents.
        """

        results: list[
            ParsedSection
        ] = []

        for document in documents:

            results.extend(
                self.parse_document(
                    document
                )
            )

        return results

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_source_document(
        document: SourceDocument,
    ) -> None:
        """
        Validate minimum metadata required for parsing.
        """

        missing: list[
            str
        ] = []

        if not document.document_id:
            missing.append(
                "document_id"
            )

        if not document.file_name:
            missing.append(
                "file_name"
            )

        if not document.domain:
            missing.append(
                "domain"
            )

        if not document.access_level:
            missing.append(
                "access_level"
            )

        if document.is_empty:
            missing.append(
                "content"
            )

        if missing:

            raise DocumentParseError(
                f"Document "
                f"'{document.file_name}' "
                f"cannot be parsed. "
                f"Missing/empty: "
                f"{', '.join(missing)}"
            )

    # =========================================================================
    # PAGE SPLITTING
    # =========================================================================

    @staticmethod
    def _split_pages(
        content: str,
    ) -> list[
        tuple[
            int | None,
            str,
        ]
    ]:
        """
        Split content by optional [PAGE N] markers.
        """

        matches = list(
            PAGE_MARKER_PATTERN.finditer(
                content
            )
        )

        if not matches:

            return [
                (
                    None,
                    content.strip(),
                )
            ]

        pages: list[
            tuple[
                int | None,
                str,
            ]
        ] = []

        prefix = content[
            :matches[0].start()
        ].strip()

        if prefix:
            pages.append(
                (
                    None,
                    prefix,
                )
            )

        for index, match in enumerate(
            matches
        ):

            start = (
                match.end()
            )

            if (
                index + 1
                < len(matches)
            ):
                end = (
                    matches[
                        index + 1
                    ].start()
                )
            else:
                end = len(
                    content
                )

            page_content = (
                content[
                    start:end
                ].strip()
            )

            if page_content:

                pages.append(
                    (
                        int(
                            match.group(
                                1
                            )
                        ),
                        page_content,
                    )
                )

        return pages

    # =========================================================================
    # LOGICAL SECTION SPLITTING
    # =========================================================================

    def _split_logical_sections(
        self,
        content: str,
    ) -> list[
        tuple[
            str,
            str,
        ]
    ]:
        """
        Split document text into logical sections.

        Rules:
            1. Explicit headings have highest priority.
            2. FAQ questions are treated as headings.
            3. Ingredient-table heuristics may infer Standard Batch Formulation
               only when the current section is not already explicitly suitable.
            4. Product Facts is never overridden by a table heuristic.
        """

        lines = [
            line.rstrip()
            for line
            in content.splitlines()
        ]

        sections: list[
            tuple[
                str,
                str,
            ]
        ] = []

        current_title = (
            "Document Content"
        )

        current_lines: list[
            str
        ] = []

        current_title_is_explicit = (
            False
        )

        for raw_line in lines:

            stripped = (
                raw_line.strip()
            )

            # -----------------------------------------------------------------
            # Blank line
            # -----------------------------------------------------------------

            if not stripped:

                if (
                    current_lines
                    and current_lines[
                        -1
                    ] != ""
                ):
                    current_lines.append(
                        ""
                    )

                continue

            # -----------------------------------------------------------------
            # Explicit heading detection
            # -----------------------------------------------------------------

            explicit_heading = (
                self._detect_explicit_heading(
                    stripped
                )
            )

            if explicit_heading:

                self._append_section(
                    sections=sections,
                    title=current_title,
                    lines=current_lines,
                )

                current_title = (
                    explicit_heading
                )

                current_lines = []

                current_title_is_explicit = (
                    True
                )

                continue

            # -----------------------------------------------------------------
            # FAQ-style question heading
            # -----------------------------------------------------------------

            if self._is_question_heading(
                stripped
            ):

                self._append_section(
                    sections=sections,
                    title=current_title,
                    lines=current_lines,
                )

                current_title = (
                    stripped
                )

                current_lines = []

                current_title_is_explicit = (
                    True
                )

                continue

            # -----------------------------------------------------------------
            # Structural table heuristic
            # -----------------------------------------------------------------

            inferred_heading = (
                self._infer_table_heading(
                    stripped
                )
            )

            if inferred_heading:

                normalized_current = (
                    self._normalize_heading(
                        current_title
                    )
                )

                # -------------------------------------------------------------
                # Never override Product Facts.
                # -------------------------------------------------------------

                if (
                    normalized_current
                    in {
                        "product facts",
                        "product information",
                        "product details",
                    }
                ):

                    current_lines.append(
                        stripped
                    )

                    continue

                # -------------------------------------------------------------
                # If already inside an explicit formulation section,
                # keep the ingredient table in that same section.
                # -------------------------------------------------------------

                if (
                    normalized_current
                    in {
                        "standard batch formulation",
                        "batch formulation",
                        "ingredients",
                        "ingredient",
                        "ingredient formula",
                        "ingredient formula / ratio",
                    }
                ):

                    current_lines.append(
                        stripped
                    )

                    continue

                # -------------------------------------------------------------
                # Only infer a new section when no appropriate explicit
                # section is currently active.
                # -------------------------------------------------------------

                self._append_section(
                    sections=sections,
                    title=current_title,
                    lines=current_lines,
                )

                current_title = (
                    inferred_heading
                )

                current_lines = [
                    stripped
                ]

                current_title_is_explicit = (
                    False
                )

                continue

            # -----------------------------------------------------------------
            # Normal content line
            # -----------------------------------------------------------------

            current_lines.append(
                stripped
            )

        # ---------------------------------------------------------------------
        # Flush last section
        # ---------------------------------------------------------------------

        self._append_section(
            sections=sections,
            title=current_title,
            lines=current_lines,
        )

        return sections

    # =========================================================================
    # HEADING DETECTION
    # =========================================================================

    def _detect_explicit_heading(
        self,
        line: str,
    ) -> str | None:
        """
        Detect a known explicit section heading.
        """

        normalized = (
            self._normalize_heading(
                line
            )
        )

        if (
            normalized
            in KNOWN_SECTION_HEADINGS
        ):
            return (
                self._clean_heading(
                    line
                )
            )

        return None

    @staticmethod
    def _is_question_heading(
        line: str,
    ) -> bool:
        """
        Treat short standalone questions as FAQ section headings.
        """

        return (
            line.endswith("?")
            and len(line) <= 180
            and "|" not in line
        )

    @staticmethod
    def _infer_table_heading(
        line: str,
    ) -> str | None:
        """
        Infer Standard Batch Formulation from a table header.

        Example:
            Ingredient | Quantity (g) | Ratio vs. Base Flour
        """

        if "|" not in line:
            return None

        cells = [
            cell.strip().lower()
            for cell
            in line.split("|")
        ]

        joined = " ".join(
            cells
        )

        if (
            "ingredient"
            in joined
            and "quantity"
            in joined
        ):
            return (
                "Standard Batch Formulation"
            )

        return None

    # =========================================================================
    # SECTION APPENDING
    # =========================================================================

    @staticmethod
    def _append_section(
        sections: list[
            tuple[
                str,
                str,
            ]
        ],
        title: str,
        lines: list[str],
    ) -> None:
        """
        Append a completed logical section.
        """

        if not lines:
            return

        content = "\n".join(
            lines
        )

        content = re.sub(
            r"\n{3,}",
            "\n\n",
            content,
        ).strip()

        if not content:
            return

        sections.append(
            (
                title.strip(),
                content,
            )
        )

    # =========================================================================
    # HEADING NORMALIZATION
    # =========================================================================

    @staticmethod
    def _clean_heading(
        value: str,
    ) -> str:
        """
        Remove markdown markers/colon noise.
        """

        cleaned = (
            value.strip()
        )

        cleaned = re.sub(
            r"^#{1,6}\s*",
            "",
            cleaned,
        )

        cleaned = (
            cleaned.rstrip(
                ":"
            ).strip()
        )

        return cleaned

    @staticmethod
    def _normalize_heading(
        value: str,
    ) -> str:
        """
        Normalize heading for classification.
        """

        cleaned = (
            DocumentParser
            ._clean_heading(
                value
            )
        )

        cleaned = (
            cleaned.lower()
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        )

        return (
            cleaned.strip()
        )

    # =========================================================================
    # ACCESS CLASSIFICATION
    # =========================================================================

    def _resolve_section_access(
        self,
        document: SourceDocument,
        section_title: str,
    ) -> tuple[
        str,
        list[str],
    ]:
        """
        Resolve section-level security for mixed-access product guides.
        """

        document_access = (
            document.access_level
            or "unclassified"
        ).lower()

        # ---------------------------------------------------------------------
        # Non-mixed document: inherit document metadata.
        # ---------------------------------------------------------------------

        if (
            document_access
            != "mixed"
        ):

            return (
                document.access_level
                or "unclassified",

                list(
                    document.audience
                ),
            )

        normalized_title = (
            self._normalize_heading(
                section_title
            )
        )

        # ---------------------------------------------------------------------
        # Public
        # ---------------------------------------------------------------------

        if (
            normalized_title
            in PUBLIC_PRODUCT_HEADINGS
        ):

            return (
                "public",
                list(
                    PUBLIC_PRODUCT_AUDIENCE
                ),
            )

        # ---------------------------------------------------------------------
        # Confidential
        # ---------------------------------------------------------------------

        if (
            normalized_title
            in CONFIDENTIAL_PRODUCT_HEADINGS
        ):

            return (
                "confidential",
                list(
                    CONFIDENTIAL_PRODUCT_AUDIENCE
                ),
            )

        # ---------------------------------------------------------------------
        # Internal
        # ---------------------------------------------------------------------

        if (
            normalized_title
            in INTERNAL_PRODUCT_HEADINGS
        ):

            return (
                "internal",
                list(
                    INTERNAL_PRODUCT_AUDIENCE
                ),
            )

        # ---------------------------------------------------------------------
        # Generic document content in mixed product documents
        # defaults to internal.
        # ---------------------------------------------------------------------

        if (
            normalized_title
            == "document content"
        ):

            return (
                "internal",
                list(
                    INTERNAL_PRODUCT_AUDIENCE
                ),
            )

        # ---------------------------------------------------------------------
        # Fail conservatively.
        # ---------------------------------------------------------------------

        return (
            "internal",
            list(
                INTERNAL_PRODUCT_AUDIENCE
            ),
        )

    # =========================================================================
    # SECTION TYPE
    # =========================================================================

    def _infer_section_type(
        self,
        title: str,
        content: str,
    ) -> str:
        """
        Infer semantic section type.
        """

        normalized = (
            self._normalize_heading(
                title
            )
        )

        # ---------------------------------------------------------------------
        # FAQ
        # ---------------------------------------------------------------------

        if title.endswith("?"):
            return (
                "faq_question"
            )

        # ---------------------------------------------------------------------
        # Formula / ingredients
        # ---------------------------------------------------------------------

        if (
            normalized
            in {
                "ingredients",
                "ingredient",
                "ingredient formula",
                "ingredient formula / ratio",
                "standard batch formulation",
                "batch formulation",
            }
        ):

            return (
                "ingredients_or_formula"
            )

        # ---------------------------------------------------------------------
        # Procedure
        # ---------------------------------------------------------------------

        if (
            normalized
            in {
                "preparation",
                "preparation process",
                "preparation / cooking process",
                "cooking process",
            }
        ):

            return (
                "procedure"
            )

        # ---------------------------------------------------------------------
        # Product facts
        # ---------------------------------------------------------------------

        if (
            normalized
            in {
                "product facts",
                "product information",
                "product details",
                "price",
                "shelf life",
                "storage",
                "storage instructions",
                "allergens",
                "allergen information",
                "pack size",
                "category",
                "dietary status",
                "dietary classification",
                "preservatives",
                "claim",
                "approved claim",
            }
        ):

            return (
                "product_fact"
            )

        # ---------------------------------------------------------------------
        # Description
        # ---------------------------------------------------------------------

        if (
            normalized
            in {
                "description",
                "product description",
            }
        ):

            return (
                "description"
            )

        # ---------------------------------------------------------------------
        # Structured table fallback
        # ---------------------------------------------------------------------

        if "|" in content:

            return (
                "table_or_structured_data"
            )

        return (
            "general_section"
        )

    # =========================================================================
    # SECTION ID
    # =========================================================================

    @staticmethod
    def _create_section_id(
        document: SourceDocument,
        section_index: int,
    ) -> str:
        """
        Create stable sequential section ID.
        """

        return (
            f"{document.document_id}"
            f"-S"
            f"{section_index:03d}"
        )