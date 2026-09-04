"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Retrieval / Security Validation
File         : test_role_based_retrieval.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026
Description  :
    Runs the same real-corpus query using different OnlyNativeIQ roles.

Purpose:
    Demonstrate that retrieval relevance and security work together.

    The script compares:
        - Customer
        - Employee
        - Production Employee
        - Manager

    Confidential formulation information must never appear in customer or
    ordinary employee final retrieval context.

    Production employees and managers may receive confidential formulation
    content when explicitly authorized by document audience metadata.

Retrieval Flow:
    Query
      -> Vector Retrieval
      -> BM25 Retrieval
      -> Hybrid RRF
      -> CrossEncoder Reranking
      -> Access-Control Filtering
      -> Authorized Results
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys


# =============================================================================
# Project path setup
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# OnlyNativeIQ imports
# =============================================================================

from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline,
)

from src.security.access_control import (
    UserContext,
    UserRole,
)

from src.utils.config_loader import (
    ConfigLoader,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_QUERY = (
    "How much ghee is required for one batch of Classic Thekua?"
)

ROLES_TO_TEST = [
    UserRole.CUSTOMER,
    UserRole.EMPLOYEE,
    UserRole.PRODUCTION_EMPLOYEE,
    UserRole.MANAGER,
]


# =============================================================================
# Display helpers
# =============================================================================

def print_separator(
    character: str = "=",
    width: int = 110,
) -> None:

    print(
        character * width
    )


def print_result(
    result,
) -> None:
    """
    Print one authorized retrieval result.

    Content is intentionally printed so we can verify that confidential
    formulation values are not leaking into unauthorized contexts.
    """

    print(
        f"Rank={result.rank:<3} "
        f"Score={result.score:.6f}"
    )

    print(
        f"File        : "
        f"{result.file_name}"
    )

    print(
        f"Section     : "
        f"{result.section}"
    )

    print(
        f"Chunk       : "
        f"{result.chunk_id}"
    )

    print(
        f"Access      : "
        f"{result.access_level}"
    )

    print(
        f"Audience    : "
        f"{result.audience}"
    )

    print(
        "Content:"
    )

    print(
        result.content
    )

    print_separator(
        "-",
        110,
    )


# =============================================================================
# Role test
# =============================================================================

def run_for_role(
    pipeline: RetrievalPipeline,
    query: str,
    role: UserRole,
) -> None:

    print()
    print_separator()

    print(
        f"ROLE: {role.value.upper()}"
    )

    print_separator()

    user = UserContext(
        role=role
    )

    result = pipeline.retrieve(
        query=query,
        user=user,
    )

    print(
        f"Hybrid candidates     : "
        f"{result.hybrid_candidates_count}"
    )

    print(
        f"Reranked candidates   : "
        f"{result.reranked_results_count}"
    )

    print(
        f"Filtered by security  : "
        f"{result.filtered_results_count}"
    )

    print(
        f"Authorized final      : "
        f"{result.authorized_results_count}"
    )

    print(
        f"Access control enabled: "
        f"{result.access_control_enabled}"
    )

    print()

    if not result.final_results:

        print(
            "No authorized retrieval context "
            "is available for this role."
        )

        return

    print(
        "AUTHORIZED RETRIEVAL RESULTS"
    )

    print_separator(
        "-",
        110,
    )

    for retrieval_result in (
        result.final_results
    ):
        print_result(
            retrieval_result
        )


# =============================================================================
# Security validation
# =============================================================================

def validate_security(
    pipeline: RetrievalPipeline,
    query: str,
) -> None:
    """
    Programmatically verify the critical formulation-access rule.
    """

    print()
    print_separator()

    print(
        "SECURITY VALIDATION"
    )

    print_separator()

    unauthorized_roles = [
        UserRole.CUSTOMER,
        UserRole.EMPLOYEE,
    ]

    authorized_roles = [
        UserRole.PRODUCTION_EMPLOYEE,
        UserRole.MANAGER,
    ]

    failures: list[str] = []

    # ---------------------------------------------------------------------
    # Customer + Employee must NOT receive confidential chunks
    # ---------------------------------------------------------------------

    for role in unauthorized_roles:

        output = pipeline.retrieve(
            query=query,
            user=UserContext(
                role=role
            ),
        )

        confidential_results = [
            result
            for result
            in output.final_results
            if (
                result.access_level
                or ""
            ).lower()
            == "confidential"
        ]

        if confidential_results:

            failures.append(
                f"{role.value}: "
                f"received confidential content."
            )

            print(
                f"[FAIL] {role.value:<20} "
                f"Confidential content leaked."
            )

        else:

            print(
                f"[PASS] {role.value:<20} "
                f"No confidential content returned."
            )

    # ---------------------------------------------------------------------
    # Production Employee + Manager may receive formulation
    # ---------------------------------------------------------------------

    for role in authorized_roles:

        output = pipeline.retrieve(
            query=query,
            user=UserContext(
                role=role
            ),
        )

        confidential_results = [
            result
            for result
            in output.final_results
            if (
                result.access_level
                or ""
            ).lower()
            == "confidential"
        ]

        if confidential_results:

            print(
                f"[PASS] {role.value:<20} "
                f"Authorized confidential content available."
            )

        else:

            print(
                f"[WARN] {role.value:<20} "
                f"No confidential result appeared in final Top-K."
            )

    print()

    if failures:

        print_separator(
            "!",
            110,
        )

        print(
            "SECURITY VALIDATION FAILED"
        )

        for failure in failures:
            print(
                f" - {failure}"
            )

        print_separator(
            "!",
            110,
        )

        raise RuntimeError(
            "Role-based retrieval security validation failed."
        )

    print_separator(
        "=",
        110,
    )

    print(
        "SECURITY VALIDATION PASSED"
    )

    print_separator(
        "=",
        110,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    config = ConfigLoader(
        settings_path=(
            "config/settings.yaml"
        ),
        project_root=PROJECT_ROOT,
    )

    pipeline = (
        RetrievalPipeline.from_config(
            config
        )
    )

    query = (
        " ".join(
            sys.argv[1:]
        ).strip()
        if len(sys.argv) > 1
        else DEFAULT_QUERY
    )

    print()
    print_separator()

    print(
        "OnlyNativeIQ - Real Corpus Role-Based Retrieval Test"
    )

    print_separator()

    print(
        f"Query: {query}"
    )

    print(
        "Roles: "
        "customer, employee, "
        "production_employee, manager"
    )

    # ---------------------------------------------------------------------
    # Run retrieval separately for every role
    # ---------------------------------------------------------------------

    for role in ROLES_TO_TEST:

        run_for_role(
            pipeline=pipeline,
            query=query,
            role=role,
        )

    # ---------------------------------------------------------------------
    # Automated security validation
    # ---------------------------------------------------------------------

    validate_security(
        pipeline=pipeline,
        query=query,
    )


if __name__ == "__main__":
    main()