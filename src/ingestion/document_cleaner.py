"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Document Ingestion
File         : document_cleaner.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Cleans and normalizes ParsedSection content before chunking.

Responsibilities:
    - Normalize Unicode whitespace and line endings.
    - Remove zero-width/control characters that add retrieval noise.
    - Collapse accidental duplicate blank lines.
    - Normalize repeated spaces without damaging table structure.
    - Preserve section titles, quantities, ratios, page numbers, metadata and
      access-control classifications.

Design Principles:
    - Cleaning must be loss-minimizing and deterministic.
    - The cleaner must not rewrite business facts.
    - The cleaner must not infer missing information.
    - The cleaner must not alter numerical values, units or ingredient ratios.
    - Table rows represented with "|" delimiters must remain intact.
================================================================================
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import replace
from typing import Iterable

from src.schemas.document import ParsedSection


logger = logging.getLogger(__name__)


class DocumentCleaner:
    """
    Normalize ParsedSection text while preserving semantic meaning.
    """

    # Common invisible characters that can appear after copy/paste or document
    # extraction and reduce BM25/vector retrieval quality.
    ZERO_WIDTH_CHARACTERS = {
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\ufeff",  # byte-order mark / zero-width no-break space
    }

    # Unicode spaces that should behave like normal ASCII spaces.
    SPACE_TRANSLATION = {
        ord("\u00a0"): " ",  # non-breaking space
        ord("\u2007"): " ",  # figure space
        ord("\u202f"): " ",  # narrow no-break space
    }

    def clean_section(
        self,
        section: ParsedSection,
    ) -> ParsedSection:
        """
        Return a cleaned copy of one ParsedSection.

        The original ParsedSection remains unchanged.
        """

        cleaned_content = self.clean_text(
            section.content
        )

        cleaned_title = self.clean_title(
            section.title
        )

        metadata = dict(section.metadata)
        metadata["cleaned"] = True
        metadata["original_character_count"] = len(
            section.content
        )
        metadata["cleaned_character_count"] = len(
            cleaned_content
        )

        return replace(
            section,
            title=cleaned_title,
            content=cleaned_content,
            metadata=metadata,
        )

    def clean_sections(
        self,
        sections: Iterable[ParsedSection],
    ) -> list[ParsedSection]:
        """
        Clean multiple parsed sections while preserving their order.
        """

        cleaned_sections = [
            self.clean_section(section)
            for section in sections
        ]

        logger.info(
            "Cleaned %d parsed section(s).",
            len(cleaned_sections),
        )

        return cleaned_sections

    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Clean body text without rewriting semantic content.
        """

        if not text:
            return ""

        value = text

        # Normalize CRLF/CR line endings to LF.
        value = value.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        # Convert equivalent Unicode forms consistently while preserving
        # readable characters.
        value = unicodedata.normalize(
            "NFKC",
            value,
        )

        value = value.translate(
            self.SPACE_TRANSLATION
        )

        for character in self.ZERO_WIDTH_CHARACTERS:
            value = value.replace(
                character,
                "",
            )

        lines = value.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            cleaned_lines.append(
                self._clean_line(line)
            )

        value = "\n".join(
            cleaned_lines
        )

        # Keep at most one empty line between blocks.
        value = re.sub(
            r"\n[ \t]*\n(?:[ \t]*\n)+",
            "\n\n",
            value,
        )

        return value.strip()

    def clean_title(
        self,
        title: str,
    ) -> str:
        """
        Normalize a section title conservatively.
        """

        if not title:
            return "Document Content"

        value = unicodedata.normalize(
            "NFKC",
            title,
        )

        value = value.translate(
            self.SPACE_TRANSLATION
        )

        for character in self.ZERO_WIDTH_CHARACTERS:
            value = value.replace(
                character,
                "",
            )

        value = re.sub(
            r"[ \t]+",
            " ",
            value,
        )

        return value.strip()

    @staticmethod
    def _clean_line(
        line: str,
    ) -> str:
        """
        Clean one line while preserving structured-table delimiters.
        """

        if not line:
            return ""

        value = line.strip()

        if not value:
            return ""

        # Normalize tabs to spaces.
        value = value.replace(
            "\t",
            " ",
        )

        # Collapse repeated spaces.
        value = re.sub(
            r" {2,}",
            " ",
            value,
        )

        # For rows created by document_loader.py, normalize spacing around the
        # pipe separator but never remove the separator.
        if "|" in value:
            cells = [
                re.sub(
                    r"\s+",
                    " ",
                    cell.strip(),
                )
                for cell in value.split("|")
            ]

            value = " | ".join(
                cells
            )

        return value
