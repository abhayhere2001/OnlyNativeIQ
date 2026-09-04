from __future__ import annotations
from pathlib import Path
import sys
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding import GroundingBuilder
from src.generation.rag_chain import RAGChain
from src.llm.llm_factory import LLMFactory
from src.memory.conversation_summarizer import ConversationSummarizer
from src.memory.query_resolver import QueryResolver
from src.memory.session_manager import SessionManager
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.security.access_control import UserContext, UserRole
from src.utils.config_loader import ConfigLoader

def build_rag_chain():
    config = ConfigLoader(settings_path="config/settings.yaml", project_root=PROJECT_ROOT)
    retrieval = RetrievalPipeline.from_config(config)
    llm = LLMFactory(config).create()
    grounding = GroundingBuilder(max_context_characters=int(
        config.get("generation.grounding.max_context_characters", default=12000)
    ))
    generator = AnswerGenerator(
        llm=llm,
        grounding_builder=grounding,
        include_sources=bool(config.get("generation.behavior.include_sources", default=True)),
    )

    enabled = bool(config.get("memory.enabled", default=True))
    strategy = str(config.get("memory.strategy", default="hybrid"))
    recent = int(config.get("memory.max_recent_messages", default=10))
    summary_enabled = bool(config.get("memory.summary_enabled", default=True))

    manager_max = recent + 2 if enabled and strategy.lower() == "hybrid" and summary_enabled else recent
    manager = SessionManager(max_turns=manager_max, max_context_characters=6000)

    return RAGChain(
        retrieval_pipeline=retrieval,
        answer_generator=generator,
        session_manager=manager,
        query_resolver=QueryResolver(),
        conversation_summarizer=ConversationSummarizer(),
        memory_enabled=enabled,
        memory_strategy=strategy,
        max_recent_messages=recent,
        summary_enabled=summary_enabled,
    )

def main():
    role = UserRole(sys.argv[1]) if len(sys.argv) > 1 else UserRole.CUSTOMER
    session_id = sys.argv[2] if len(sys.argv) > 2 else "conversation_demo_001"
    chain = build_rag_chain()
    user = UserContext(role=role)

    questions = [
        "What is the shelf life of Classic Thekua?",
        "How should I store it?",
        "What is its price?",
        "Tell me about it.",
        "Shelf life?",
        "What allergens does it have?",
        "How should I store it?",
        "What is its price?",
        "Tell me about it.",
        "How much ghee does it use?",
    ]

    print("=" * 110)
    print("OnlyNativeIQ - Hybrid Memory Conversational RAG Test")
    print("=" * 110)
    print(f"Role       : {role.value}")
    print(f"Session ID : {session_id}")

    for i, q in enumerate(questions, 1):
        r = chain.invoke(question=q, user=user, session_id=session_id)
        print("\n" + "-" * 110)
        print(f"TURN {i}")
        print("-" * 110)
        print(f"Original Question : {r.question}")
        print(f"Resolved Question : {r.resolved_question}")
        print(f"Remembered Entity : {r.remembered_entity}")
        print(f"Recent Msg Count  : {r.memory_turn_count}")
        print(f"Summary Updated   : {r.summary_updated}")
        print(f"Memory Summary    : {r.memory_summary or '(empty)'}")
        print(r.final_answer)

    memory = chain.session_manager.get_session(session_id)
    print("\n" + "=" * 110)
    print("FINAL HYBRID MEMORY")
    print("=" * 110)
    print(f"Recent messages: {memory.turn_count}")
    print(f"Last entity    : {memory.last_referenced_entity}")
    print("\nSUMMARY\n" + "-" * 110)
    print(memory.summary or "(empty)")
    print("\nRECENT HISTORY\n" + "-" * 110)
    print(memory.render_recent_history() or "(empty)")

if __name__ == "__main__":
    main()
