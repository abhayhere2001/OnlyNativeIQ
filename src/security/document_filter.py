"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Security
File         : document_filter.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Filters retrieval results according to OnlyNativeIQ access-control rules.

Responsibilities:
    - Remove unauthorized retrieval results.
    - Preserve original relevance order for authorized results.
    - Re-rank authorized results sequentially.
    - Return diagnostics describing how many results were filtered.
================================================================================
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from src.schemas.retrieval import RetrievalResult
from src.security.access_control import (
    AccessController,
    UserContext,
)


@dataclass(slots=True)
class FilterResult:
    """Result of one access-control filtering operation."""

    input_count: int
    authorized_count: int
    filtered_count: int

    authorized_results: list[
        RetrievalResult
    ] = field(
        default_factory=list
    )


class DocumentFilter:
    """Filter retrieval candidates before they can reach generation."""

    def __init__(
        self,
        access_controller: AccessController,
    ) -> None:
        self.access_controller = (
            access_controller
        )

    def filter_results(
        self,
        results: list[RetrievalResult],
        user: UserContext,
        top_k: int | None = None,
    ) -> FilterResult:
        """
        Return only results authorized for the supplied user context.
        """

        authorized: list[
            RetrievalResult
        ] = []

        for result in results:
            if self.access_controller.is_authorized(
                result,
                user,
            ):
                authorized.append(
                    deepcopy(
                        result
                    )
                )

        if top_k is not None:
            if top_k <= 0:
                authorized = []
            else:
                authorized = authorized[
                    :top_k
                ]

        for rank, result in enumerate(
            authorized,
            start=1,
        ):
            result.rank = rank

            metadata = dict(
                result.metadata
            )

            metadata[
                "access_filter_applied"
            ] = True

            metadata[
                "authorized_role"
            ] = user.role.value

            result.metadata = metadata

        return FilterResult(
            input_count=len(results),
            authorized_count=len(
                authorized
            ),
            filtered_count=(
                len(results)
                - len(authorized)
            ),
            authorized_results=(
                authorized
            ),
        )
