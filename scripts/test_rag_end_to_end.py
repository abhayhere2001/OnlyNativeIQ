"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Integration / End-to-End RAG Test
File         : test_rag_end_to_end.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Runs a real end-to-end OnlyNativeIQ RAG query against the actual corpus.

Pipeline:
    Question
        -> Vector Retrieval
        -> BM25 Retrieval
        -> Hybrid RRF
        -> Cross-Encoder Reranking
        -> Role-Based Access Control
        -> Authorized Retrieval Results
        -> Grounding
        -> LLM Generation
        -> Trusted Citations
        -> Final Answer

Usage:
    python scripts/test_rag_end_to_end.py

Optional:
    python scripts/test_rag_end_to_end.py customer
    python scripts/test_rag_end_to_end.py production_employee

Custom question:
    python scripts/test_rag_end_to_end.py customer \
        "What is the shelf life of Classic Thekua?"
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys

from dotenv import load_dotenv


# =============================================================================
# PROJECT ROOT
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
# LOAD ENVIRONMENT VARIABLES
# =============================================================================

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)

load_dotenv(
    dotenv_path=ENV_FILE
)


# =============================================================================
# ONLYNATIVEIQ IMPORTS
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
# DEFAULTS
# =============================================================================

DEFAULT_QUERY = (
    "How much ghee is required "
    "for one batch of Classic Thekua?"
)

DEFAULT_ROLE = (
    UserRole.PRODUCTION_EMPLOYEE
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def separator(
    character: str = "=",
    width: int = 110,
) -> None:
    """
    Print a visual separator.
    """

    print(
        character * width
    )


def heading(
    text: str,
) -> None:
    """
    Print a formatted section heading.
    """

    print()
    separator()

    print(
        text
    )

    separator()


# =============================================================================
# ROLE PARSING
# =============================================================================

def parse_role(
    value: str,
) -> UserRole:
    """
    Convert command-line role text into UserRole.
    """

    normalized = (
        value.strip()
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    for role in UserRole:

        if (
            role.value
            == normalized
        ):
            return role

    valid_roles = ", ".join(
        role.value
        for role in UserRole
    )

    raise ValueError(
        f"Unknown role '{value}'. "
        f"Valid roles: {valid_roles}"
    )


# =============================================================================
# GROUNDED EVIDENCE DIAGNOSTICS
# =============================================================================

def print_grounded_evidence(
    retrieval_pipeline: RetrievalPipeline,
    query: str,
    user: UserContext,
) -> None:
    """
    Print the actual authorized retrieval chunks that are allowed to reach
    generation.

    This diagnostic is especially useful when:
        - retrieval succeeded,
        - authorization succeeded,
        - the LLM was invoked,
        - but the generated answer was not what we expected.

    Important:
        This prints only final authorized retrieval results.
    """

    heading(
        "GROUNDED EVIDENCE"
    )

    retrieval_debug = (
        retrieval_pipeline.retrieve(
            query=query,
            user=user,
        )
    )

    if not retrieval_debug.final_results:

        print(
            "No authorized retrieval results."
        )

        return

    print(
        f"Authorized evidence chunks: "
        f"{len(retrieval_debug.final_results)}"
    )

    print()

    for result in (
        retrieval_debug.final_results
    ):

        separator(
            "-",
            110,
        )

        print(
            f"Rank       : "
            f"{result.rank}"
        )

        print(
            f"Score      : "
            f"{result.score:.6f}"
        )

        print(
            f"File       : "
            f"{result.file_name}"
        )

        print(
            f"Section    : "
            f"{result.section}"
        )

        print(
            f"Chunk      : "
            f"{result.chunk_id}"
        )

        print(
            f"Document ID: "
            f"{result.document_id}"
        )

        print(
            f"Access     : "
            f"{result.access_level}"
        )

        print(
            f"Audience   : "
            f"{result.audience}"
        )

        print(
            f"Source     : "
            f"{result.source}"
        )

        print()

        print(
            "CONTENT"
        )

        separator(
            "-",
            110,
        )

        print(
            result.content
        )

        print()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Run the real end-to-end OnlyNativeIQ RAG test.
    """

    # -------------------------------------------------------------------------
    # Role
    # -------------------------------------------------------------------------

    role = DEFAULT_ROLE

    if len(sys.argv) >= 2:

        role = parse_role(
            sys.argv[1]
        )

    # -------------------------------------------------------------------------
    # Optional question
    #
    # Example:
    #
    # python scripts/test_rag_end_to_end.py customer \
    #     "What is OnlyNative?"
    #
    # -------------------------------------------------------------------------

    if len(sys.argv) >= 3:

        query = " ".join(
            sys.argv[2:]
        ).strip()

    else:

        query = DEFAULT_QUERY

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
    # Retrieval pipeline
    # -------------------------------------------------------------------------

    retrieval_pipeline = (
        RetrievalPipeline.from_config(
            config
        )
    )

    # -------------------------------------------------------------------------
    # LLM creation
    # -------------------------------------------------------------------------

    llm = (
        LLMFactory(
            config
        )
        .create()
    )

    # -------------------------------------------------------------------------
    # Grounding configuration
    # -------------------------------------------------------------------------

    max_context_characters = (
        config.get(
            "generation.grounding."
            "max_context_characters",
            default=12000,
        )
    )

    grounding_builder = (
        GroundingBuilder(
            max_context_characters=int(
                max_context_characters
            )
        )
    )

    # -------------------------------------------------------------------------
    # Answer generator
    # -------------------------------------------------------------------------

    include_sources = (
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
            include_sources=bool(
                include_sources
            ),
        )
    )

    # -------------------------------------------------------------------------
    # RAG chain
    # -------------------------------------------------------------------------

    rag_chain = RAGChain(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        answer_generator=(
            answer_generator
        ),
    )

    # -------------------------------------------------------------------------
    # User
    # -------------------------------------------------------------------------

    user = UserContext(
        role=role
    )

    # -------------------------------------------------------------------------
    # Start
    # -------------------------------------------------------------------------

    heading(
        "OnlyNativeIQ - End-to-End RAG Test"
    )

    print(
        f"Role     : "
        f"{role.value}"
    )

    print(
        f"Question : "
        f"{query}"
    )

    # -------------------------------------------------------------------------
    # Execute RAG chain
    # -------------------------------------------------------------------------

    response = (
        rag_chain.invoke(
            question=query,
            user=user,
        )
    )

    # -------------------------------------------------------------------------
    # Final answer
    # -------------------------------------------------------------------------

    heading(
        "FINAL ANSWER"
    )

    print(
        response.final_answer
    )

    # -------------------------------------------------------------------------
    # RAG diagnostics
    # -------------------------------------------------------------------------

    heading(
        "RAG DIAGNOSTICS"
    )

    print(
        f"Role                  : "
        f"{response.user_role}"
    )

    print(
        f"Hybrid candidates     : "
        f"{response.retrieval_candidates}"
    )

    print(
        f"Reranked candidates   : "
        f"{response.reranked_candidates}"
    )

    print(
        f"Filtered by security  : "
        f"{response.security_filtered}"
    )

    print(
        f"Authorized results    : "
        f"{response.authorized_results}"
    )

    print(
        f"Grounding evidence    : "
        f"{response.evidence_count}"
    )

    print(
        f"LLM invoked           : "
        f"{response.used_llm}"
    )

    print(
        f"Has evidence          : "
        f"{response.has_evidence}"
    )

    # -------------------------------------------------------------------------
    # Trusted citations
    # -------------------------------------------------------------------------

    heading(
        "TRUSTED CITATIONS"
    )

    if not response.citations:

        print(
            "No citations."
        )

    else:

        for citation in (
            response.citations
        ):

            print(
                citation.display_text
            )

    # -------------------------------------------------------------------------
    # Authorized grounded evidence diagnostics
    # -------------------------------------------------------------------------

    print_grounded_evidence(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        query=query,
        user=user,
    )

    # -------------------------------------------------------------------------
    # End
    # -------------------------------------------------------------------------

    print()

    separator()

    print(
        "End-to-end RAG test completed."
    )

    separator()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()