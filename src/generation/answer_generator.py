"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Generation
File         : answer_generator.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Generates grounded OnlyNativeIQ answers from authorized retrieval evidence.

Responsibilities:
    - Accept authorized retrieval results only.
    - Build controlled grounding context.
    - Avoid LLM invocation when no usable evidence exists.
    - Construct strict grounded prompts.
    - Invoke the configured chat model.
    - Build trusted citations from retrieval metadata.
    - Return a structured answer for downstream RAG orchestration/UI.

Security Principles:
    - Authorization is completed before this layer.
    - The LLM never decides document permissions.
    - The LLM never constructs trusted citation metadata.
    - No evidence means no LLM call.
    - Only authorized grounded context is supplied to the LLM.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from src.generation.citation_builder import (
    Citation,
    CitationBuilder,
)
from src.generation.grounding import (
    GroundingBuilder,
    GroundingContext,
)
from src.generation.prompts import (
    NO_CONTEXT_RESPONSE,
    SYSTEM_PROMPT,
    PromptInput,
    build_rag_prompt,
)
from src.schemas.retrieval import (
    RetrievalResult,
)


# =============================================================================
# Exceptions
# =============================================================================

class AnswerGenerationError(RuntimeError):
    """
    Raised when grounded answer generation fails.
    """


# =============================================================================
# Generated answer
# =============================================================================

@dataclass(slots=True)
class GeneratedAnswer:
    """
    Structured result returned by AnswerGenerator.
    """

    answer: str

    citations: list[
        Citation
    ] = field(
        default_factory=list
    )

    sources_text: str = ""

    grounding_context: (
        GroundingContext | None
    ) = None

    used_llm: bool = False

    has_evidence: bool = False

    evidence_count: int = 0


# =============================================================================
# Answer generator
# =============================================================================

class AnswerGenerator:
    """
    Generate an answer strictly from authorized grounded evidence.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        grounding_builder: (
            GroundingBuilder | None
        ) = None,
        citation_builder: (
            CitationBuilder | None
        ) = None,
        include_sources: bool = True,
    ) -> None:

        if llm is None:
            raise ValueError(
                "llm cannot be None."
            )

        self.llm = llm

        self.grounding_builder = (
            grounding_builder
            or GroundingBuilder()
        )

        self.citation_builder = (
            citation_builder
            or CitationBuilder()
        )

        self.include_sources = (
            include_sources
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def generate(
        self,
        question: str,
        user_role: str,
        retrieval_results: list[
            RetrievalResult
        ],
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer.

        Parameters
        ----------
        question:
            User question.

        user_role:
            Role already authenticated/validated by the security layer.

        retrieval_results:
            ONLY final authorized results returned by RetrievalPipeline.

        Returns
        -------
        GeneratedAnswer
            Structured answer including trusted citations.
        """

        question = (
            question or ""
        ).strip()

        user_role = (
            user_role or ""
        ).strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not user_role:
            raise ValueError(
                "User role cannot be empty."
            )

        # ---------------------------------------------------------------------
        # Build grounded evidence
        # ---------------------------------------------------------------------

        grounding_context = (
            self.grounding_builder.build(
                retrieval_results
            )
        )

        # ---------------------------------------------------------------------
        # Critical security/grounding rule:
        # Do NOT invoke the LLM when no authorized evidence exists.
        # ---------------------------------------------------------------------

        if not grounding_context.has_evidence:

            return GeneratedAnswer(
                answer=NO_CONTEXT_RESPONSE,
                citations=[],
                sources_text="",
                grounding_context=(
                    grounding_context
                ),
                used_llm=False,
                has_evidence=False,
                evidence_count=0,
            )

        # ---------------------------------------------------------------------
        # Build controlled user prompt
        # ---------------------------------------------------------------------

        prompt = build_rag_prompt(
            PromptInput(
                question=question,
                context=(
                    grounding_context.context_text
                ),
                user_role=user_role,
            )
        )

        if not prompt:

            return GeneratedAnswer(
                answer=NO_CONTEXT_RESPONSE,
                citations=[],
                sources_text="",
                grounding_context=(
                    grounding_context
                ),
                used_llm=False,
                has_evidence=False,
                evidence_count=0,
            )

        # ---------------------------------------------------------------------
        # Invoke LLM
        # ---------------------------------------------------------------------

        try:

            response = self.llm.invoke(
                [
                    SystemMessage(
                        content=SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=prompt
                    ),
                ]
            )

        except Exception as exc:

            raise AnswerGenerationError(
                "OnlyNativeIQ answer generation "
                f"failed: {exc}"
            ) from exc

        answer_text = (
            self._extract_response_text(
                response
            )
        )

        if not answer_text:

            raise AnswerGenerationError(
                "The configured LLM returned "
                "an empty answer."
            )

        # ---------------------------------------------------------------------
        # Trusted citations
        # ---------------------------------------------------------------------

        citations = (
            self.citation_builder.build(
                grounding_context
            )
        )

        sources_text = ""

        if self.include_sources:

            sources_text = (
                self.citation_builder
                .format_sources(
                    citations
                )
            )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            sources_text=sources_text,
            grounding_context=(
                grounding_context
            ),
            used_llm=True,
            has_evidence=True,
            evidence_count=(
                grounding_context
                .evidence_count
            ),
        )

    # =========================================================================
    # Final answer formatting
    # =========================================================================

    def format_final_answer(
        self,
        generated: GeneratedAnswer,
    ) -> str:
        """
        Combine generated answer with trusted sources for display.

        Citation metadata is appended programmatically rather than being
        trusted to the LLM.
        """

        answer = (
            generated.answer
            or ""
        ).strip()

        if (
            not self.include_sources
            or not generated.sources_text
        ):
            return answer

        return (
            f"{answer}\n\n"
            f"{generated.sources_text}"
        )

    # =========================================================================
    # LLM response normalization
    # =========================================================================

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:
        """
        Normalize common LangChain chat-model responses to plain text.
        """

        if response is None:
            return ""

        content = getattr(
            response,
            "content",
            response,
        )

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        # Some providers may return structured content blocks.
        if isinstance(
            content,
            list,
        ):

            text_parts: list[str] = []

            for item in content:

                if isinstance(
                    item,
                    str,
                ):
                    text_parts.append(
                        item
                    )

                elif isinstance(
                    item,
                    dict,
                ):

                    text = (
                        item.get("text")
                        or item.get("content")
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

            return "\n".join(
                text_parts
            ).strip()

        return str(
            content
        ).strip()