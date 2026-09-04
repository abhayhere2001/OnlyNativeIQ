"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Memory
File         : conversation_summarizer.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026
Updated On   : 04-Sep-2026

Description  :
    Builds compact, security-conscious conversation summaries for OnlyNativeIQ.

Purpose:
    Hybrid memory keeps:
        1. Recent messages verbatim.
        2. Older conversation context in compact summary form.

Security Principle:
    Summary memory must NOT become an alternate knowledge store.

    Therefore this summarizer intentionally preserves:
        - products/entities discussed
        - topics discussed
        - conversational intent
        - current conversational focus

    and intentionally avoids preserving:
        - recipe/formulation quantities
        - exact prices
        - exact shelf-life values
        - production parameters
        - internal/confidential factual values
        - citations/source content

    Any factual answer must still be retrieved again through the normal:
        retrieval
        -> reranking
        -> access control
        -> relevance gate
        -> grounding
        -> generation

Notes:
    - This implementation is deterministic and does not require an LLM.
    - Product names are matched against the current authoritative product set.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from src.memory.conversation_memory import (
    ConversationRole,
    ConversationTurn,
)


# =============================================================================
# Summary result
# =============================================================================

@dataclass(frozen=True, slots=True)
class ConversationSummaryResult:
    """Result of one conversation-summary operation."""

    summary: str
    turns_processed: int
    entities: tuple[str, ...]
    topics: tuple[str, ...]
    used_existing_summary: bool


# =============================================================================
# Conversation summarizer
# =============================================================================

class ConversationSummarizer:
    """
    Build a compact context-only summary from older conversation turns.

    Assistant factual answers are deliberately excluded from semantic
    extraction so summary memory does not become a secondary knowledge store.
    """

    KNOWN_PRODUCTS: tuple[str, ...] = (
        "Classic Thekua",
        "Danedar Sugar Thekua",
        "Jaggery Thekua",
        "Classic Gujiya",
        "Classic Nimki",
        "Garlic Blended Nimki",
        "Maida Spicy Mathree",
        "Whole Wheat Spicy Mathree",
    )

    TOPIC_PATTERNS: tuple[
        tuple[str, tuple[str, ...]],
        ...
    ] = (
        (
            "shelf life",
            (
                "shelf life",
                "expiry",
                "expire",
                "best before",
            ),
        ),
        (
            "price",
            (
                "price",
                "cost",
                "how much does",
            ),
        ),
        (
            "storage",
            (
                "storage",
                "store",
                "keep it",
                "keep this",
            ),
        ),
        (
            "allergens",
            (
                "allergen",
                "gluten",
                "dairy",
                "milk",
            ),
        ),
        (
            "ingredients/formulation",
            (
                "ingredient",
                "formulation",
                "ghee",
                "sugar",
                "jaggery",
                "flour",
                "oil",
                "quantity",
                "batch",
            ),
        ),
        (
            "preparation process",
            (
                "prepare",
                "preparation",
                "process",
                "cook",
                "cooking",
                "fry",
                "shape",
            ),
        ),
        (
            "pack size",
            (
                "pack size",
                "weight",
                "grams",
                "gm",
            ),
        ),
        (
            "product description",
            (
                "describe",
                "description",
                "tell me about",
            ),
        ),
    )

    def summarize(
        self,
        turns: Iterable[
            ConversationTurn
        ],
        *,
        existing_summary: str = "",
        last_referenced_entity: str | None = None,
    ) -> ConversationSummaryResult:
        """
        Summarize conversation turns into compact context memory.
        """

        turn_list = list(
            turns
        )

        normalized_existing = (
            existing_summary
            or ""
        ).strip()

        if not turn_list:
            return ConversationSummaryResult(
                summary=normalized_existing,
                turns_processed=0,
                entities=(
                    (last_referenced_entity,)
                    if last_referenced_entity
                    else ()
                ),
                topics=(),
                used_existing_summary=bool(
                    normalized_existing
                ),
            )

        entities: list[str] = []
        topics: list[str] = []

        if last_referenced_entity:
            self._append_unique(
                entities,
                last_referenced_entity.strip(),
            )

        # ---------------------------------------------------------------------
        # Analyze USER turns only.
        # ---------------------------------------------------------------------

        for turn in turn_list:

            if (
                turn.role
                != ConversationRole.USER
            ):
                continue

            text = (
                turn.content
                or ""
            ).strip()

            if not text:
                continue

            normalized = (
                text.lower()
            )

            for topic_name, patterns in (
                self.TOPIC_PATTERNS
            ):

                if any(
                    pattern
                    in normalized
                    for pattern
                    in patterns
                ):
                    self._append_unique(
                        topics,
                        topic_name,
                    )

            for entity in (
                self._extract_product_entities(
                    text
                )
            ):
                self._append_unique(
                    entities,
                    entity,
                )

        # ---------------------------------------------------------------------
        # Build context-only summary.
        # ---------------------------------------------------------------------

        summary_parts: list[str] = []

        if normalized_existing:
            summary_parts.append(
                normalized_existing
            )

        current_parts: list[str] = []

        if entities:
            current_parts.append(
                "Products/entities discussed: "
                + ", ".join(
                    entities
                )
                + "."
            )

        if topics:
            current_parts.append(
                "Topics discussed: "
                + ", ".join(
                    topics
                )
                + "."
            )

        if last_referenced_entity:
            current_parts.append(
                "Current conversational focus: "
                f"{last_referenced_entity.strip()}."
            )

        if current_parts:
            summary_parts.append(
                " ".join(
                    current_parts
                )
            )

        summary = self._deduplicate_summary(
            "\n".join(
                summary_parts
            )
        )

        return ConversationSummaryResult(
            summary=summary,
            turns_processed=len(
                turn_list
            ),
            entities=tuple(
                entities
            ),
            topics=tuple(
                topics
            ),
            used_existing_summary=bool(
                normalized_existing
            ),
        )

    # =========================================================================
    # Security helpers
    # =========================================================================

    @staticmethod
    def sanitize_summary(
        summary: str,
    ) -> str:
        """
        Remove obvious factual numeric values from externally supplied summary.

        Examples removed:
            75 g
            30 days
            INR 160
            ₹175
            15.0%
        """

        value = (
            summary
            or ""
        ).strip()

        if not value:
            return ""

        patterns = (
            r"\b\d+(?:\.\d+)?\s*(?:g|gm|kg|days?|hours?|minutes?)\b",
            r"\bINR\s*\d+(?:\.\d+)?\b",
            r"₹\s*\d+(?:\.\d+)?",
            r"\b\d+(?:\.\d+)?\s*%",
        )

        for pattern in patterns:
            value = re.sub(
                pattern,
                "[fact omitted]",
                value,
                flags=re.IGNORECASE,
            )

        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

    # =========================================================================
    # Entity extraction
    # =========================================================================

    @classmethod
    def _extract_product_entities(
        cls,
        text: str,
    ) -> tuple[str, ...]:
        """
        Extract known OnlyNative product names conservatively.

        Examples:
            Classic Thekua
            Jaggery Thekua
            Danedar Sugar Thekua
            Classic Gujiya
            Classic Nimki
            Garlic Blended Nimki
            Maida Spicy Mathree
            Whole Wheat Spicy Mathree
        """

        normalized_text = (
            text
            or ""
        ).lower()

        found: list[str] = []

        # Match longer names first.
        for product in sorted(
            cls.KNOWN_PRODUCTS,
            key=len,
            reverse=True,
        ):

            if (
                product.lower()
                in normalized_text
            ):
                found.append(
                    product
                )

        return tuple(
            dict.fromkeys(
                found
            )
        )

    # =========================================================================
    # Generic helpers
    # =========================================================================

    @staticmethod
    def _append_unique(
        values: list[str],
        value: str,
    ) -> None:

        normalized = (
            value
            or ""
        ).strip()

        if (
            normalized
            and normalized
            not in values
        ):
            values.append(
                normalized
            )

    @staticmethod
    def _deduplicate_summary(
        summary: str,
    ) -> str:
        """
        Remove duplicate summary lines while preserving order.
        """

        lines = [
            line.strip()
            for line
            in summary.splitlines()
            if line.strip()
        ]

        unique_lines = list(
            dict.fromkeys(
                lines
            )
        )

        return "\n".join(
            unique_lines
        )
