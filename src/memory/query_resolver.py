"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Memory
File         : query_resolver.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Resolves short conversational follow-up questions using session memory.

Responsibilities:
    - Detect when a question is likely a follow-up.
    - Use the last referenced entity/product when appropriate.
    - Reconstruct a more explicit standalone retrieval query.
    - Preserve the user's original intent.
    - Never make authorization decisions.

Security Principle:
    Query resolution changes only the wording of the retrieval query.
    The resolved query must still pass through the normal:
        retrieval
        -> reranking
        -> access control
        -> relevance gate
        -> grounding
        -> generation

Important:
    This component does NOT grant access to remembered restricted facts.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from src.memory.conversation_memory import (
    ConversationMemory,
)


# =============================================================================
# Result model
# =============================================================================

@dataclass(frozen=True, slots=True)
class QueryResolutionResult:
    """
    Result returned by QueryResolver.
    """

    original_query: str

    resolved_query: str

    was_resolved: bool

    referenced_entity: str | None

    reason: str


# =============================================================================
# Resolver
# =============================================================================

class QueryResolver:
    """
    Resolve conversational follow-up questions into standalone queries.

    Initial implementation is deliberately conservative and rule-based.
    """

    # -------------------------------------------------------------------------
    # Common conversational references
    # -------------------------------------------------------------------------

    REFERENCE_TERMS = {
        "it",
        "its",
        "this",
        "that",
        "this one",
        "that one",
        "one",
        "same",
        "same one",
        "the same",
    }

    FOLLOW_UP_PREFIXES = (
        "what about",
        "how about",
        "and what about",
        "and how about",
        "what is its",
        "what's its",
        "tell me its",
        "how much does it",
        "how much is its",
        "does it",
        "is it",
        "can it",
    )

    def resolve(
        self,
        query: str,
        memory: ConversationMemory | None,
    ) -> QueryResolutionResult:
        """
        Resolve one user query using conversation memory.

        Parameters
        ----------
        query:
            Current user question.

        memory:
            Current session memory. May be None.

        Returns
        -------
        QueryResolutionResult
            Original and resolved query plus diagnostic information.
        """

        original_query = (
            query or ""
        ).strip()

        if not original_query:
            raise ValueError(
                "query cannot be empty."
            )

        # ---------------------------------------------------------------------
        # No memory available
        # ---------------------------------------------------------------------

        if memory is None:

            return QueryResolutionResult(
                original_query=original_query,
                resolved_query=original_query,
                was_resolved=False,
                referenced_entity=None,
                reason="No conversation memory available.",
            )

        entity = (
            memory.last_referenced_entity
        )

        # ---------------------------------------------------------------------
        # No remembered entity
        # ---------------------------------------------------------------------

        if not entity:

            return QueryResolutionResult(
                original_query=original_query,
                resolved_query=original_query,
                was_resolved=False,
                referenced_entity=None,
                reason=(
                    "No last referenced entity "
                    "is available in memory."
                ),
            )

        normalized = (
            original_query.lower()
        )

        # ---------------------------------------------------------------------
        # Explicit entity already present
        #
        # If the current question already mentions the remembered entity,
        # no rewriting is necessary.
        # ---------------------------------------------------------------------

        if (
            entity.lower()
            in normalized
        ):

            return QueryResolutionResult(
                original_query=original_query,
                resolved_query=original_query,
                was_resolved=False,
                referenced_entity=entity,
                reason=(
                    "Current query already contains "
                    "the remembered entity."
                ),
            )

        # ---------------------------------------------------------------------
        # Follow-up detection
        # ---------------------------------------------------------------------

        if not self._looks_like_follow_up(
            normalized
        ):

            return QueryResolutionResult(
                original_query=original_query,
                resolved_query=original_query,
                was_resolved=False,
                referenced_entity=entity,
                reason=(
                    "Query does not appear to require "
                    "conversation-based resolution."
                ),
            )

        # ---------------------------------------------------------------------
        # Resolve
        # ---------------------------------------------------------------------

        resolved = (
            self._resolve_with_entity(
                original_query,
                entity,
            )
        )

        return QueryResolutionResult(
            original_query=original_query,
            resolved_query=resolved,
            was_resolved=(
                resolved
                != original_query
            ),
            referenced_entity=entity,
            reason=(
                "Resolved conversational reference "
                "using the last referenced entity."
            ),
        )

    # =========================================================================
    # Follow-up detection
    # =========================================================================

    def _looks_like_follow_up(
        self,
        normalized_query: str,
    ) -> bool:
        """
        Return True when a query likely depends on earlier context.
        """

        query = (
            normalized_query
            .strip()
            .rstrip("?")
        )

        # ---------------------------------------------------------------------
        # Prefix-based follow-ups
        # ---------------------------------------------------------------------

        if any(
            query.startswith(
                prefix
            )
            for prefix
            in self.FOLLOW_UP_PREFIXES
        ):
            return True

        # ---------------------------------------------------------------------
        # Pronoun/reference detection
        # ---------------------------------------------------------------------

        padded = (
            f" {query} "
        )

        for reference in (
            self.REFERENCE_TERMS
        ):

            if (
                f" {reference} "
                in padded
            ):
                return True

        # ---------------------------------------------------------------------
        # Very short conversational questions
        #
        # Examples:
        #     Shelf life?
        #     Price?
        #     Ingredients?
        #
        # With an active remembered entity, these are likely follow-ups.
        # ---------------------------------------------------------------------

        words = query.split()

        if (
            len(words)
            <= 4
        ):
            return True

        return False

    # =========================================================================
    # Resolution
    # =========================================================================

    def _resolve_with_entity(
        self,
        query: str,
        entity: str,
    ) -> str:
        """
        Build a standalone retrieval query using the remembered entity.
        """

        stripped = (
            query.strip()
        )

        normalized = (
            stripped.lower()
        )

        # ---------------------------------------------------------------------
        # "What about Jaggery one?"
        #
        # If the user provides a modifier followed by "one", preserve that
        # modifier while adding the product family from remembered entity.
        #
        # Example:
        #   Previous entity: Classic Thekua
        #   Current query  : What about Jaggery one?
        #   Resolved       : What about Jaggery Thekua?
        # ---------------------------------------------------------------------

        if (
            normalized.startswith(
                "what about "
            )
            or normalized.startswith(
                "how about "
            )
        ):

            candidate = (
                stripped
                .rstrip("?")
            )

            if (
                candidate.lower()
                .endswith(" one")
            ):

                prefix_length = (
                    len("what about ")
                    if normalized.startswith(
                        "what about "
                    )
                    else len(
                        "how about "
                    )
                )

                modifier = (
                    candidate[
                        prefix_length:
                    ]
                    .strip()
                )

                modifier = (
                    modifier[
                        :-len(" one")
                    ]
                    .strip()
                )

                family = (
                    self._infer_entity_family(
                        entity
                    )
                )

                if (
                    modifier
                    and family
                ):

                    prefix = (
                        "What about"
                        if normalized.startswith(
                            "what about "
                        )
                        else "How about"
                    )

                    return (
                        f"{prefix} "
                        f"{modifier} "
                        f"{family}?"
                    )

        # ---------------------------------------------------------------------
        # Short fact request
        #
        # Example:
        #   "Shelf life?"
        #       ->
        #   "Shelf life of Classic Thekua?"
        # ---------------------------------------------------------------------

        words = (
            stripped
            .rstrip("?")
            .split()
        )

        if (
            len(words)
            <= 4
            and not self._contains_reference_term(
                normalized
            )
        ):

            return (
                f"{stripped.rstrip('?')} "
                f"of {entity}?"
            )

        # ---------------------------------------------------------------------
        # Replace simple references with the explicit entity.
        # ---------------------------------------------------------------------

        replacements = (
            ("this one", entity),
            ("that one", entity),
            ("same one", entity),
            ("the same", entity),
            (" its ", f" {entity}'s "),
            (" it ", f" {entity} "),
        )

        rewritten = (
            f" {stripped} "
        )

        for (
            source,
            target,
        ) in replacements:

            rewritten = (
                rewritten.replace(
                    source,
                    target,
                )
            )

            rewritten = (
                rewritten.replace(
                    source.title(),
                    target,
                )
            )

        rewritten = (
            rewritten.strip()
        )

        # ---------------------------------------------------------------------
        # If the query still lacks the entity, append it explicitly.
        #
        # Example:
        #   "What about the shelf life?"
        #       ->
        #   "What about the shelf life of Classic Thekua?"
        # ---------------------------------------------------------------------

        if (
            entity.lower()
            not in rewritten.lower()
        ):

            rewritten = (
                f"{rewritten.rstrip('?')} "
                f"for {entity}?"
            )

        return rewritten

    # =========================================================================
    # Helpers
    # =========================================================================

    def _contains_reference_term(
        self,
        normalized_query: str,
    ) -> bool:
        """
        Return True when known conversational reference terms are present.
        """

        padded = (
            f" "
            f"{normalized_query.strip().rstrip('?')}"
            f" "
        )

        return any(
            f" {reference} "
            in padded
            for reference
            in self.REFERENCE_TERMS
        )

    @staticmethod
    def _infer_entity_family(
        entity: str,
    ) -> str | None:
        """
        Infer the product-family portion of a remembered entity.

        Examples
        --------
        Classic Thekua -> Thekua
        Jaggery Thekua -> Thekua
        Classic Gujiya -> Gujiya
        Garlic Blended Nimki -> Nimki

        This is intentionally simple for the initial implementation.
        """

        words = [
            word
            for word
            in entity.split()
            if word
        ]

        if not words:
            return None

        return words[
            -1
        ]