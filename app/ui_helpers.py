"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Streamlit UI Helpers
File         : ui_helpers.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026
Updated On   : 04-Sep-2026

Description  :
    Session-state, authentication, configuration and diagnostics helpers
    for the production-style OnlyNativeIQ Streamlit application.
================================================================================
"""

from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st

from src.security.access_control import (
    UserContext,
)

from src.security.authentication import (
    AuthenticatedUser,
)


# =============================================================================
# Session IDs
# =============================================================================

def new_session_id() -> str:
    """
    Generate a fresh conversation ID.
    """

    return (
        "ui_"
        + uuid4().hex
    )


# =============================================================================
# Streamlit state
# =============================================================================

def ensure_ui_state() -> None:
    """
    Initialize required UI session-state fields.
    """

    defaults = {
        "conversation_session_id":
            new_session_id(),

        "chat_messages":
            [],

        "authenticated_user":
            None,
    }

    for key, value in (
        defaults.items()
    ):

        if key not in st.session_state:

            st.session_state[
                key
            ] = value


# =============================================================================
# Authentication state
# =============================================================================

def login_user(
    authenticated_user: AuthenticatedUser,
) -> None:
    """
    Establish a trusted authenticated session.
    """

    if authenticated_user is None:
        raise ValueError(
            "authenticated_user cannot be None."
        )

    st.session_state[
        "authenticated_user"
    ] = authenticated_user

    st.session_state[
        "conversation_session_id"
    ] = new_session_id()

    st.session_state[
        "chat_messages"
    ] = []


def logout_user(
    rag_chain,
) -> None:
    """
    Logout and clear all conversation memory belonging to the UI session.
    """

    _delete_server_memory(
        rag_chain
    )

    st.session_state[
        "authenticated_user"
    ] = None

    st.session_state[
        "conversation_session_id"
    ] = new_session_id()

    st.session_state[
        "chat_messages"
    ] = []


def reset_conversation(
    rag_chain,
) -> None:
    """
    Start a fresh conversation while keeping the user authenticated.
    """

    _delete_server_memory(
        rag_chain
    )

    st.session_state[
        "conversation_session_id"
    ] = new_session_id()

    st.session_state[
        "chat_messages"
    ] = []


def _delete_server_memory(
    rag_chain,
) -> None:
    """
    Remove the active RAG memory session if present.
    """

    if rag_chain is None:
        return

    session_id = (
        st.session_state.get(
            "conversation_session_id"
        )
    )

    if not session_id:
        return

    try:

        rag_chain.session_manager.delete_session(
            session_id
        )

    except Exception:
        # Session cleanup should never prevent logout/new conversation.
        pass


# =============================================================================
# Trusted authorization context
# =============================================================================

def build_user_context(
    authenticated_user: AuthenticatedUser,
) -> UserContext:
    """
    Build RAG UserContext strictly from authenticated server-side identity.
    """

    if authenticated_user is None:
        raise ValueError(
            "Authenticated user is required."
        )

    return UserContext(
        role=(
            authenticated_user.role
        )
    )


# =============================================================================
# Configuration helpers
# =============================================================================

def get_bool_config(
    config,
    key: str,
    default: bool,
) -> bool:
    """
    Safely normalize boolean-like configuration values.
    """

    value = config.get(
        key,
        default=default,
    )

    if isinstance(
        value,
        bool,
    ):
        return value

    return str(
        value
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def development_mode_enabled(
    config,
) -> bool:
    """
    Developer diagnostics are disabled unless explicitly enabled.
    """

    environment_value = os.getenv(
        "ONLYNATIVEIQ_DEV_MODE"
    )

    if environment_value is not None:

        return (
            environment_value
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

    return get_bool_config(
        config,
        "ui.development_mode",
        False,
    )


# =============================================================================
# Diagnostics
# =============================================================================

def build_diagnostics(
    response,
) -> dict:
    """
    Build a compact developer diagnostic object.
    """

    return {
        "session_id":
            response.session_id,

        "original_question":
            response.question,

        "resolved_question":
            response.resolved_question,

        "query_was_resolved":
            response.query_was_resolved,

        "remembered_entity":
            response.remembered_entity,

        "memory_turn_count":
            response.memory_turn_count,

        "summary_updated":
            response.summary_updated,

        "retrieval_candidates":
            response.retrieval_candidates,

        "reranked_candidates":
            response.reranked_candidates,

        "security_filtered":
            response.security_filtered,

        "relevance_filtered":
            response.relevance_filtered,

        "authorized_results":
            response.authorized_results,

        "grounding_evidence":
            response.evidence_count,

        "llm_invoked":
            response.used_llm,

        "has_evidence":
            response.has_evidence,
    }
