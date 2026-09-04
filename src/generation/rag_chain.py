from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from src.generation.citation_builder import Citation
from src.memory.conversation_memory import ConversationMemory
from src.memory.conversation_summarizer import ConversationSummarizer
from src.memory.query_resolver import QueryResolutionResult, QueryResolver
from src.memory.session_manager import SessionManager
from src.security.access_control import UserContext

class RAGChainError(RuntimeError): pass

@dataclass(slots=True)
class RAGResponse:
    question: str
    resolved_question: str
    answer: str
    final_answer: str
    user_role: str
    session_id: str | None = None
    query_was_resolved: bool = False
    remembered_entity: str | None = None
    memory_turn_count: int = 0
    memory_summary: str = ""
    summary_updated: bool = False
    citations: list[Citation] = field(default_factory=list)
    sources_text: str = ""
    retrieval_candidates: int = 0
    reranked_candidates: int = 0
    security_filtered: int = 0
    relevance_filtered: int = 0
    authorized_results: int = 0
    evidence_count: int = 0
    used_llm: bool = False
    has_evidence: bool = False

class RAGChain:
    def __init__(self, retrieval_pipeline, answer_generator, session_manager=None,
                 query_resolver=None, conversation_summarizer=None, *,
                 memory_enabled=True, memory_strategy="hybrid",
                 max_recent_messages=10, summary_enabled=True):
        if retrieval_pipeline is None: raise ValueError("retrieval_pipeline cannot be None.")
        if answer_generator is None: raise ValueError("answer_generator cannot be None.")
        if max_recent_messages <= 0: raise ValueError("max_recent_messages must be greater than zero.")
        strategy = (memory_strategy or "recent").strip().lower()
        if strategy not in {"recent", "hybrid"}: raise ValueError("memory_strategy must be 'recent' or 'hybrid'.")

        self.retrieval_pipeline = retrieval_pipeline
        self.answer_generator = answer_generator
        self.memory_enabled = bool(memory_enabled)
        self.memory_strategy = strategy
        self.max_recent_messages = int(max_recent_messages)
        self.summary_enabled = bool(summary_enabled) and strategy == "hybrid"
        manager_max = self.max_recent_messages + 2 if self.summary_enabled else self.max_recent_messages
        self.session_manager = session_manager or SessionManager(max_turns=manager_max)
        self.query_resolver = query_resolver or QueryResolver()
        self.conversation_summarizer = conversation_summarizer or ConversationSummarizer()

    def invoke(self, question: str, user: UserContext, session_id: str | None = None):
        original = (question or "").strip()
        if not original: raise ValueError("Question cannot be empty.")
        if user is None: raise ValueError("user cannot be None.")

        memory = self._get_memory(session_id)
        resolution = self._resolve_query(original, memory)
        resolved = resolution.resolved_query

        try:
            retrieval = self.retrieval_pipeline.retrieve(query=resolved, user=user)
        except Exception as exc:
            raise RAGChainError(f"OnlyNativeIQ retrieval failed: {exc}") from exc

        authorized = retrieval.final_results

        try:
            generated = self.answer_generator.generate(
                question=resolved,
                user_role=self._role_value(user),
                retrieval_results=authorized,
            )
        except Exception as exc:
            raise RAGChainError(f"OnlyNativeIQ answer generation failed: {exc}") from exc

        final_answer = self.answer_generator.format_final_answer(generated)
        remembered = resolution.referenced_entity
        summary_updated = False

        if memory is not None:
            entity = self._infer_entity_from_results(authorized)
            if entity:
                memory.set_last_referenced_entity(entity)
                remembered = entity
            elif memory.last_referenced_entity:
                remembered = memory.last_referenced_entity

            memory.add_user_message(original)
            memory.add_assistant_message(generated.answer)
            summary_updated = self._update_hybrid_summary(memory)

        return RAGResponse(
            question=original,
            resolved_question=resolved,
            answer=generated.answer,
            final_answer=final_answer,
            user_role=self._role_value(user),
            session_id=memory.session_id if memory else None,
            query_was_resolved=resolution.was_resolved,
            remembered_entity=remembered,
            memory_turn_count=memory.turn_count if memory else 0,
            memory_summary=memory.summary if memory else "",
            summary_updated=summary_updated,
            citations=generated.citations,
            sources_text=generated.sources_text,
            retrieval_candidates=self._safe_int(retrieval, "hybrid_candidates_count"),
            reranked_candidates=self._safe_int(retrieval, "reranked_results_count"),
            security_filtered=self._safe_int(retrieval, "filtered_results_count"),
            relevance_filtered=self._safe_int(retrieval, "relevance_filtered_results_count"),
            authorized_results=self._safe_int(retrieval, "authorized_results_count"),
            evidence_count=generated.evidence_count,
            used_llm=generated.used_llm,
            has_evidence=generated.has_evidence,
        )

    def _get_memory(self, session_id):
        if not self.memory_enabled or session_id is None: return None
        session_id = (session_id or "").strip()
        if not session_id: raise ValueError("session_id cannot be empty.")
        return self.session_manager.get_or_create_session(session_id)

    def _resolve_query(self, question, memory):
        try:
            return self.query_resolver.resolve(query=question, memory=memory)
        except Exception as exc:
            raise RAGChainError(f"OnlyNativeIQ query resolution failed: {exc}") from exc

    def _update_hybrid_summary(self, memory: ConversationMemory) -> bool:
        if not self.summary_enabled or self.memory_strategy != "hybrid": return False
        overflow = memory.turn_count - self.max_recent_messages
        if overflow <= 0: return False
        older = memory.oldest_turns(overflow)
        if not older: return False

        result = self.conversation_summarizer.summarize(
            turns=older,
            existing_summary=memory.summary,
            last_referenced_entity=memory.last_referenced_entity,
        )
        memory.set_summary(self.conversation_summarizer.sanitize_summary(result.summary))
        memory.remove_oldest_turns(overflow)
        return True

    @classmethod
    def _infer_entity_from_results(cls, results):
        for result in results:
            entity = cls._product_entity_from_file_name((getattr(result, "file_name", None) or "").strip())
            if entity: return entity
        return None

    @staticmethod
    def _product_entity_from_file_name(file_name):
        if not file_name: return None
        stem = Path(file_name).stem
        suffix = "_Product_Guide"
        if not stem.endswith(suffix): return None
        product = stem[:-len(suffix)].strip("_ ")
        return " ".join(x for x in product.split("_") if x) or None

    @staticmethod
    def _role_value(user):
        role = getattr(user, "role", None)
        if role is None: raise ValueError("UserContext does not contain a role.")
        value = str(getattr(role, "value", role)).strip()
        if not value: raise ValueError("User role cannot be empty.")
        return value

    @staticmethod
    def _safe_int(obj, attribute):
        value = getattr(obj, attribute, 0)
        try: return int(value or 0)
        except (TypeError, ValueError): return 0
