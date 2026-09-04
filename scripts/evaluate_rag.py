"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Evaluation
File         : evaluate_rag.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Runs repeatable end-to-end RAG evaluation against OnlyNativeIQ.

Evaluation Coverage:
    - Public fact correctness
    - Role-based access control
    - Restricted formulation/process access
    - Hallucination / unsupported-query handling
    - Expected source file validation
    - Expected source section validation
    - LLM invocation behavior
    - Forbidden-content leakage checks

Pipeline Under Test:
    Question + Role
        -> Vector Retrieval
        -> BM25 Retrieval
        -> Hybrid RRF
        -> CrossEncoder Reranking
        -> Access Control
        -> Relevance Gate
        -> Grounding
        -> LLM Generation
        -> Trusted Citations
        -> Evaluation Assertions

Usage:
    python scripts/evaluate_rag.py

Optional custom dataset:
    python scripts/evaluate_rag.py evaluation/rag_test_cases.jsonl
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# Environment
# =============================================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)


# =============================================================================
# OnlyNativeIQ imports
# =============================================================================

from src.generation.answer_generator import (
    AnswerGenerator,
)

from src.generation.grounding import (
    GroundingBuilder,
)

from src.generation.rag_chain import (
    RAGChain,
)

from src.llm.llm_factory import (
    LLMFactory,
)

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
# Defaults
# =============================================================================

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "rag_test_cases.jsonl"
)


# =============================================================================
# Evaluation models
# =============================================================================

@dataclass(slots=True)
class EvaluationFailure:
    """
    One failed assertion for a test case.
    """

    check: str
    message: str


@dataclass(slots=True)
class EvaluationResult:
    """
    Result of one RAG evaluation case.
    """

    test_id: str
    name: str
    category: str
    role: str
    question: str

    passed: bool

    answer: str

    used_llm: bool

    authorized_results: int

    evidence_count: int

    citation_files: list[str] = field(
        default_factory=list
    )

    citation_sections: list[str] = field(
        default_factory=list
    )

    failures: list[
        EvaluationFailure
    ] = field(
        default_factory=list
    )


# =============================================================================
# Display helpers
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
    print(text)
    separator()


# =============================================================================
# Dataset loading
# =============================================================================

def load_test_cases(
    dataset_path: Path,
) -> list[dict[str, Any]]:
    """
    Load JSONL RAG evaluation cases.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{dataset_path}"
        )

    test_cases: list[
        dict[str, Any]
    ] = []

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):

            line = (
                raw_line.strip()
            )

            if not line:
                continue

            try:
                case = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in "
                    f"{dataset_path} "
                    f"at line {line_number}: "
                    f"{exc}"
                ) from exc

            _validate_test_case(
                case=case,
                line_number=line_number,
            )

            test_cases.append(
                case
            )

    if not test_cases:
        raise ValueError(
            "Evaluation dataset contains no test cases."
        )

    return test_cases


def _validate_test_case(
    case: dict[str, Any],
    line_number: int,
) -> None:
    """
    Validate required evaluation-case fields.
    """

    required_fields = [
        "test_id",
        "name",
        "category",
        "role",
        "question",
        "expected_behavior",
        "expected_contains",
        "forbidden_contains",
        "expect_llm_invoked",
    ]

    missing = [
        field_name
        for field_name
        in required_fields
        if field_name not in case
    ]

    if missing:
        raise ValueError(
            f"Evaluation case at line "
            f"{line_number} is missing: "
            f"{', '.join(missing)}"
        )


# =============================================================================
# Role handling
# =============================================================================

def parse_role(
    value: str,
) -> UserRole:
    """
    Convert configured role text to UserRole.
    """

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    try:
        return UserRole(
            normalized
        )

    except ValueError as exc:
        valid_roles = ", ".join(
            role.value
            for role in UserRole
        )

        raise ValueError(
            f"Unsupported evaluation role "
            f"'{value}'. "
            f"Valid roles: {valid_roles}"
        ) from exc


# =============================================================================
# Evaluation assertions
# =============================================================================

def evaluate_response(
    case: dict[str, Any],
    response,
) -> EvaluationResult:
    """
    Evaluate one RAGResponse against one configured test case.
    """

    failures: list[
        EvaluationFailure
    ] = []

    answer = (
        response.answer
        or ""
    ).strip()

    answer_lower = (
        answer.lower()
    )

    expected_behavior = (
        str(
            case[
                "expected_behavior"
            ]
        )
        .strip()
        .lower()
    )

    expected_contains = [
        str(value)
        for value in case.get(
            "expected_contains",
            [],
        )
    ]

    forbidden_contains = [
        str(value)
        for value in case.get(
            "forbidden_contains",
            [],
        )
    ]

    # -------------------------------------------------------------------------
    # Expected answer content
    # -------------------------------------------------------------------------

    for expected in (
        expected_contains
    ):

        if (
            expected.lower()
            not in answer_lower
        ):

            failures.append(
                EvaluationFailure(
                    check="expected_contains",
                    message=(
                        f"Answer does not contain "
                        f"expected value: "
                        f"'{expected}'."
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # Forbidden leakage
    # -------------------------------------------------------------------------

    for forbidden in (
        forbidden_contains
    ):

        if (
            forbidden.lower()
            in answer_lower
        ):

            failures.append(
                EvaluationFailure(
                    check="forbidden_contains",
                    message=(
                        f"Answer contains forbidden "
                        f"value: '{forbidden}'."
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # LLM invocation expectation
    # -------------------------------------------------------------------------

    expected_llm = bool(
        case[
            "expect_llm_invoked"
        ]
    )

    if (
        bool(
            response.used_llm
        )
        != expected_llm
    ):

        failures.append(
            EvaluationFailure(
                check="llm_invocation",
                message=(
                    f"Expected LLM invoked="
                    f"{expected_llm}, "
                    f"actual="
                    f"{response.used_llm}."
                ),
            )
        )

    # -------------------------------------------------------------------------
    # No-authorized-answer behavior
    # -------------------------------------------------------------------------

    if (
        expected_behavior
        == "no_authorized_answer"
    ):

        if response.has_evidence:

            failures.append(
                EvaluationFailure(
                    check="no_authorized_answer",
                    message=(
                        "Expected no grounded evidence, "
                        "but evidence was available."
                    ),
                )
            )

        if (
            response.authorized_results
            != 0
        ):

            failures.append(
                EvaluationFailure(
                    check="authorized_results",
                    message=(
                        "Expected zero final authorized/"
                        "relevant results, but got "
                        f"{response.authorized_results}."
                    ),
                )
            )

        if response.citations:

            failures.append(
                EvaluationFailure(
                    check="citations",
                    message=(
                        "Expected no citations for "
                        "no-authorized-answer case."
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # Positive-answer behavior
    # -------------------------------------------------------------------------

    elif (
        expected_behavior
        == "answer"
    ):

        if not response.has_evidence:

            failures.append(
                EvaluationFailure(
                    check="answer_evidence",
                    message=(
                        "Expected grounded evidence "
                        "for answer case."
                    ),
                )
            )

        if (
            response.authorized_results
            <= 0
        ):

            failures.append(
                EvaluationFailure(
                    check="answer_authorized_results",
                    message=(
                        "Expected at least one authorized/"
                        "relevant result."
                    ),
                )
            )

    else:

        failures.append(
            EvaluationFailure(
                check="expected_behavior",
                message=(
                    f"Unsupported expected_behavior: "
                    f"'{expected_behavior}'."
                ),
            )
        )

    # -------------------------------------------------------------------------
    # Citation provenance
    # -------------------------------------------------------------------------

    citation_files = [
        citation.file_name
        for citation in response.citations
    ]

    citation_sections = [
        citation.section
        or ""
        for citation in response.citations
    ]

    expected_source_file = (
        case.get(
            "expected_source_file"
        )
    )

    if expected_source_file:

        if (
            expected_source_file
            not in citation_files
        ):

            failures.append(
                EvaluationFailure(
                    check="source_file",
                    message=(
                        f"Expected source file "
                        f"'{expected_source_file}' "
                        f"not found in citations."
                    ),
                )
            )

    expected_source_section = (
        case.get(
            "expected_source_section"
        )
    )

    if expected_source_section:

        if (
            expected_source_section
            not in citation_sections
        ):

            failures.append(
                EvaluationFailure(
                    check="source_section",
                    message=(
                        f"Expected source section "
                        f"'{expected_source_section}' "
                        f"not found in citations."
                    ),
                )
            )

    return EvaluationResult(
        test_id=(
            case[
                "test_id"
            ]
        ),
        name=(
            case[
                "name"
            ]
        ),
        category=(
            case[
                "category"
            ]
        ),
        role=(
            case[
                "role"
            ]
        ),
        question=(
            case[
                "question"
            ]
        ),
        passed=(
            len(failures)
            == 0
        ),
        answer=answer,
        used_llm=(
            response.used_llm
        ),
        authorized_results=(
            response.authorized_results
        ),
        evidence_count=(
            response.evidence_count
        ),
        citation_files=(
            citation_files
        ),
        citation_sections=(
            citation_sections
        ),
        failures=(
            failures
        ),
    )


# =============================================================================
# Pipeline construction
# =============================================================================

def build_rag_chain(
    config: ConfigLoader,
) -> RAGChain:
    """
    Build the real OnlyNativeIQ RAG chain used by evaluation.
    """

    retrieval_pipeline = (
        RetrievalPipeline.from_config(
            config
        )
    )

    llm = (
        LLMFactory(
            config
        ).create()
    )

    max_context_characters = int(
        config.get(
            "generation.grounding."
            "max_context_characters",
            default=12000,
        )
    )

    grounding_builder = (
        GroundingBuilder(
            max_context_characters=(
                max_context_characters
            )
        )
    )

    include_sources = bool(
        config.get(
            "generation.behavior."
            "include_sources",
            default=True,
        )
    )

    answer_generator = (
        AnswerGenerator(
            llm=llm,
            grounding_builder=(
                grounding_builder
            ),
            include_sources=(
                include_sources
            ),
        )
    )

    return RAGChain(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        answer_generator=(
            answer_generator
        ),
    )


# =============================================================================
# Result display
# =============================================================================

def print_case_result(
    result: EvaluationResult,
) -> None:
    """
    Print concise PASS/FAIL output for one case.
    """

    status = (
        "PASS"
        if result.passed
        else "FAIL"
    )

    print(
        f"[{status}] "
        f"{result.test_id} - "
        f"{result.name}"
    )

    if result.passed:
        return

    print(
        f"       Role     : "
        f"{result.role}"
    )

    print(
        f"       Question : "
        f"{result.question}"
    )

    print(
        f"       Answer   : "
        f"{result.answer}"
    )

    for failure in (
        result.failures
    ):

        print(
            f"       - "
            f"[{failure.check}] "
            f"{failure.message}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Execute all configured RAG evaluation cases.
    """

    dataset_path = (
        Path(
            sys.argv[1]
        )
        if len(sys.argv) >= 2
        else DEFAULT_DATASET_PATH
    )

    if not dataset_path.is_absolute():
        dataset_path = (
            PROJECT_ROOT
            / dataset_path
        )

    # -------------------------------------------------------------------------
    # Config
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
    # Dataset
    # -------------------------------------------------------------------------

    test_cases = (
        load_test_cases(
            dataset_path
        )
    )

    # -------------------------------------------------------------------------
    # RAG
    # -------------------------------------------------------------------------

    rag_chain = (
        build_rag_chain(
            config
        )
    )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    heading(
        "OnlyNativeIQ - RAG Evaluation"
    )

    print(
        f"Dataset    : "
        f"{dataset_path}"
    )

    print(
        f"Test cases : "
        f"{len(test_cases)}"
    )

    print()

    results: list[
        EvaluationResult
    ] = []

    for case in (
        test_cases
    ):

        role = parse_role(
            case[
                "role"
            ]
        )

        try:

            response = (
                rag_chain.invoke(
                    question=(
                        case[
                            "question"
                        ]
                    ),
                    user=(
                        UserContext(
                            role=role
                        )
                    ),
                )
            )

            result = (
                evaluate_response(
                    case=case,
                    response=response,
                )
            )

        except Exception as exc:

            result = (
                EvaluationResult(
                    test_id=(
                        case[
                            "test_id"
                        ]
                    ),
                    name=(
                        case[
                            "name"
                        ]
                    ),
                    category=(
                        case[
                            "category"
                        ]
                    ),
                    role=(
                        case[
                            "role"
                        ]
                    ),
                    question=(
                        case[
                            "question"
                        ]
                    ),
                    passed=False,
                    answer="",
                    used_llm=False,
                    authorized_results=0,
                    evidence_count=0,
                    failures=[
                        EvaluationFailure(
                            check="exception",
                            message=str(
                                exc
                            ),
                        )
                    ],
                )
            )

        results.append(
            result
        )

        print_case_result(
            result
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    total = len(
        results
    )

    passed = sum(
        1
        for result
        in results
        if result.passed
    )

    failed = (
        total
        - passed
    )

    pass_rate = (
        (
            passed
            / total
        )
        * 100.0
        if total
        else 0.0
    )

    heading(
        "EVALUATION SUMMARY"
    )

    print(
        f"Total Tests : "
        f"{total}"
    )

    print(
        f"Passed      : "
        f"{passed}"
    )

    print(
        f"Failed      : "
        f"{failed}"
    )

    print(
        f"Pass Rate   : "
        f"{pass_rate:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Category summary
    # -------------------------------------------------------------------------

    categories = sorted(
        {
            result.category
            for result in results
        }
    )

    if categories:

        print()

        print(
            "Category Results:"
        )

        for category in categories:

            category_results = [
                result
                for result in results
                if result.category
                == category
            ]

            category_passed = sum(
                1
                for result
                in category_results
                if result.passed
            )

            print(
                f"  {category:<22} "
                f"{category_passed}/"
                f"{len(category_results)}"
            )

    print()

    if failed == 0:

        separator()

        print(
            "ALL RAG EVALUATION CASES PASSED"
        )

        separator()

        return

    separator(
        "!",
        110,
    )

    print(
        "RAG EVALUATION COMPLETED WITH FAILURES"
    )

    separator(
        "!",
        110,
    )

    # Non-zero exit code helps CI/CD detect evaluation regressions.
    raise SystemExit(
        1
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
