"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Streamlit Application
File         : streamlit_app.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026
Updated On   : 04-Sep-2026

Description  :
    Final production-style authenticated Streamlit application for OnlyNativeIQ.

Application Flow:
    Login
      -> AuthenticatedUser
      -> trusted role
      -> UserContext
      -> conversational RAG
      -> authorized answer
      -> citations
      -> session memory

Security:
    - No UI role selector exists.
    - Role comes only from successful authentication.
    - Logout clears visible chat and server-side memory.
    - New Conversation clears server-side conversational memory.
    - Developer diagnostics are opt-in and disabled by default.
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv


# =============================================================================
# Bootstrap
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

load_dotenv(
    PROJECT_ROOT
    / ".env"
)


# =============================================================================
# UI imports
# =============================================================================

from app.components import (
    render_authenticated_sidebar,
    render_chat_history,
    render_citations,
    render_diagnostics,
    render_login,
    render_welcome,
)

from app.ui_helpers import (
    build_diagnostics,
    build_user_context,
    development_mode_enabled,
    ensure_ui_state,
    get_bool_config,
    login_user,
    logout_user,
    reset_conversation,
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
    RAGChainError,
)

from src.llm.llm_factory import (
    LLMFactory,
)

from src.memory.conversation_summarizer import (
    ConversationSummarizer,
)

from src.memory.query_resolver import (
    QueryResolver,
)

from src.memory.session_manager import (
    SessionManager,
)

from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline,
)

from src.security.authentication import (
    LocalAuthenticationProvider,
)

from src.utils.config_loader import (
    ConfigLoader,
)


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="OnlyNativeIQ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.25rem 0.4rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(120,120,120,0.16);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Cached runtime resources
# =============================================================================

@st.cache_resource
def get_config() -> ConfigLoader:
    """
    Load settings once per Streamlit server process.
    """

    return ConfigLoader(
        settings_path=(
            "config/settings.yaml"
        ),
        project_root=(
            PROJECT_ROOT
        ),
    )


@st.cache_resource
def get_authentication_provider(
) -> LocalAuthenticationProvider:
    """
    Load trusted local authentication store.
    """

    return (
        LocalAuthenticationProvider(
            PROJECT_ROOT
            / "config"
            / "users.yaml"
        )
    )


@st.cache_resource
def build_rag_chain() -> RAGChain:
    """
    Build the complete OnlyNativeIQ conversational RAG stack.
    """

    config = get_config()

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

    grounding_builder = (
        GroundingBuilder(
            max_context_characters=int(
                config.get(
                    "generation.grounding."
                    "max_context_characters",
                    default=12000,
                )
            )
        )
    )

    answer_generator = (
        AnswerGenerator(
            llm=llm,
            grounding_builder=(
                grounding_builder
            ),
            include_sources=(
                get_bool_config(
                    config,
                    "generation.behavior.include_sources",
                    True,
                )
            ),
        )
    )

    memory_enabled = (
        get_bool_config(
            config,
            "memory.enabled",
            True,
        )
    )

    memory_strategy = str(
        config.get(
            "memory.strategy",
            default="hybrid",
        )
    ).strip().lower()

    max_recent_messages = int(
        config.get(
            "memory.max_recent_messages",
            default=10,
        )
    )

    summary_enabled = (
        get_bool_config(
            config,
            "memory.summary_enabled",
            True,
        )
    )

    manager_max_turns = (
        max_recent_messages + 2
        if (
            memory_enabled
            and memory_strategy == "hybrid"
            and summary_enabled
        )
        else max_recent_messages
    )

    session_manager = (
        SessionManager(
            max_turns=(
                manager_max_turns
            ),
            max_context_characters=6000,
        )
    )

    return RAGChain(
        retrieval_pipeline=(
            retrieval_pipeline
        ),
        answer_generator=(
            answer_generator
        ),
        session_manager=(
            session_manager
        ),
        query_resolver=(
            QueryResolver()
        ),
        conversation_summarizer=(
            ConversationSummarizer()
        ),
        memory_enabled=(
            memory_enabled
        ),
        memory_strategy=(
            memory_strategy
        ),
        max_recent_messages=(
            max_recent_messages
        ),
        summary_enabled=(
            summary_enabled
        ),
    )


# =============================================================================
# Authenticated application
# =============================================================================

def run_authenticated_app(
    authenticated_user,
) -> None:
    """
    Run the chat application for an already authenticated user.
    """

    config = get_config()

    try:

        rag_chain = (
            build_rag_chain()
        )

    except Exception as exc:

        st.error(
            "OnlyNativeIQ could not initialize."
        )

        if development_mode_enabled(
            config
        ):

            st.exception(
                exc
            )

        st.stop()

    dev_mode = (
        development_mode_enabled(
            config
        )
    )

    (
        show_diagnostics,
        new_conversation_clicked,
        logout_clicked,
    ) = render_authenticated_sidebar(
        authenticated_user=(
            authenticated_user
        ),
        session_id=(
            st.session_state[
                "conversation_session_id"
            ]
        ),
        dev_mode=(
            dev_mode
        ),
    )

    if logout_clicked:

        logout_user(
            rag_chain
        )

        st.rerun()

    if new_conversation_clicked:

        reset_conversation(
            rag_chain
        )

        st.rerun()

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    st.title(
        "🌿 OnlyNativeIQ"
    )

    st.caption(
        "Enterprise Knowledge & Customer Support Assistant for OnlyNative"
    )

    # -------------------------------------------------------------------------
    # Chat state
    # -------------------------------------------------------------------------

    messages = (
        st.session_state[
            "chat_messages"
        ]
    )

    if not messages:

        render_welcome(
            authenticated_user
        )

    render_chat_history(
        messages,
        show_diagnostics=(
            show_diagnostics
        ),
    )

    # -------------------------------------------------------------------------
    # New question
    # -------------------------------------------------------------------------

    prompt = st.chat_input(
        "Ask OnlyNativeIQ..."
    )

    if not prompt:

        return

    prompt = (
        prompt.strip()
    )

    if not prompt:

        return

    messages.append(
        {
            "role":
                "user",

            "content":
                prompt,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )

    user_context = (
        build_user_context(
            authenticated_user
        )
    )

    # -------------------------------------------------------------------------
    # Secure RAG invocation
    # -------------------------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching authorized OnlyNative knowledge..."
        ):

            try:

                response = (
                    rag_chain.invoke(
                        question=prompt,
                        user=(
                            user_context
                        ),
                        session_id=(
                            st.session_state[
                                "conversation_session_id"
                            ]
                        ),
                    )
                )

            except RAGChainError as exc:

                st.error(
                    "I couldn't complete the knowledge search."
                )

                if dev_mode:

                    st.exception(
                        exc
                    )

                return

            except Exception as exc:

                st.error(
                    "An unexpected error occurred."
                )

                if dev_mode:

                    st.exception(
                        exc
                    )

                return

        st.markdown(
            response.answer
        )

        render_citations(
            response.citations
        )

        diagnostics = (
            build_diagnostics(
                response
            )
        )

        if show_diagnostics:

            render_diagnostics(
                diagnostics,
                memory_summary=(
                    response.memory_summary
                ),
            )

    # -------------------------------------------------------------------------
    # Save visible assistant message
    # -------------------------------------------------------------------------

    messages.append(
        {
            "role":
                "assistant",

            "content":
                response.answer,

            "citations":
                response.citations,

            "diagnostics":
                diagnostics,

            "memory_summary":
                response.memory_summary,
        }
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    OnlyNativeIQ application entry point.
    """

    ensure_ui_state()

    authenticated_user = (
        st.session_state.get(
            "authenticated_user"
        )
    )

    # -------------------------------------------------------------------------
    # Login boundary
    # -------------------------------------------------------------------------

    if authenticated_user is None:

        try:

            authentication_provider = (
                get_authentication_provider()
            )

        except Exception as exc:

            st.error(
                "Authentication service could not initialize."
            )

            st.exception(
                exc
            )

            st.stop()

        authenticated_user = (
            render_login(
                authentication_provider
            )
        )

        if authenticated_user is not None:

            login_user(
                authenticated_user
            )

            st.rerun()

        return

    # -------------------------------------------------------------------------
    # Authenticated experience
    # -------------------------------------------------------------------------

    run_authenticated_app(
        authenticated_user
    )


if __name__ == "__main__":
    main()
