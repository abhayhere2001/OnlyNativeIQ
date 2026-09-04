"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Streamlit UI Components
File         : components.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026
Updated On   : 04-Sep-2026

Description  :
    Reusable production-style Streamlit UI components for OnlyNativeIQ.

Responsibilities:
    - Authentication screen
    - Authenticated sidebar
    - Chat history rendering
    - Trusted source rendering
    - Developer diagnostics
    - Empty-state / welcome experience
================================================================================
"""

from __future__ import annotations

from typing import Iterable

import streamlit as st


# =============================================================================
# Branding
# =============================================================================

APP_TITLE = "OnlyNativeIQ"
APP_SUBTITLE = (
    "Enterprise Knowledge & Customer Support Assistant for OnlyNative"
)


# =============================================================================
# Login
# =============================================================================

def render_login(
    authentication_provider,
):
    """
    Render the OnlyNativeIQ login experience.

    Returns
    -------
    AuthenticatedUser | None
        Authenticated identity when login succeeds.
    """

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 3rem;
        }
        .on-login-card {
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 18px;
            padding: 1.8rem 2rem 1.5rem 2rem;
            background: rgba(120,120,120,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌿 OnlyNativeIQ")
    st.caption(APP_SUBTITLE)

    st.write("")

    left, center, right = st.columns(
        [0.8, 1.6, 0.8]
    )

    with center:

        st.markdown(
            '<div class="on-login-card">',
            unsafe_allow_html=True,
        )

        st.subheader("Sign in")

        st.caption(
            "Use your authorized OnlyNativeIQ account. "
            "Your access level is assigned by the system and cannot be changed here."
        )

        with st.form(
            "onlynativeiq_login_form",
            clear_on_submit=False,
        ):
            username = st.text_input(
                "Username",
                placeholder="Enter username",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
            )

            submitted = st.form_submit_button(
                "Sign in",
                use_container_width=True,
                type="primary",
            )

        if submitted:

            authenticated_user = (
                authentication_provider.authenticate(
                    username=username,
                    password=password,
                )
            )

            if authenticated_user is None:
                st.error(
                    "Invalid username or password."
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                return None

            st.success(
                "Authentication successful."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            return authenticated_user

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    return None


# =============================================================================
# Sidebar
# =============================================================================

def render_authenticated_sidebar(
    *,
    authenticated_user,
    session_id: str,
    dev_mode: bool,
) -> tuple[bool, bool, bool]:
    """
    Render authenticated application sidebar.

    Returns
    -------
    tuple
        show_diagnostics, new_conversation_clicked, logout_clicked
    """

    with st.sidebar:

        st.title("🌿 OnlyNativeIQ")

        st.caption(
            "Secure enterprise knowledge assistant"
        )

        st.divider()

        st.markdown(
            f"### {authenticated_user.display_name}"
        )

        st.caption(
            f"Username: `{authenticated_user.username}`"
        )

        st.caption(
            f"Access role: `{authenticated_user.role.value}`"
        )

        st.divider()

        new_conversation_clicked = st.button(
            "＋ New Conversation",
            use_container_width=True,
            type="primary",
        )

        logout_clicked = st.button(
            "Logout",
            use_container_width=True,
        )

        st.divider()

        st.caption("Conversation session")

        st.code(
            session_id,
            language=None,
        )

        if dev_mode:
            st.divider()

            show_diagnostics = st.toggle(
                "Developer diagnostics",
                value=False,
                help=(
                    "Shows retrieval, memory, grounding and authorization "
                    "diagnostics. Keep disabled in public production."
                ),
            )
        else:
            show_diagnostics = False

        st.divider()

        st.caption(
            "Access to recipes, formulations and production information "
            "is enforced by the authenticated role."
        )

    return (
        show_diagnostics,
        new_conversation_clicked,
        logout_clicked,
    )


# =============================================================================
# Empty state
# =============================================================================

def render_welcome(
    authenticated_user,
) -> None:
    """
    Show role-aware welcome and access guidance.
    """

    role = (
        authenticated_user.role.value
    )

    display_name = (
        authenticated_user.display_name
    )

    role_messages = {

        "customer": {
            "title":
                "Customer Access",

            "message": (
                "You can ask about OnlyNative products, prices, "
                "pack sizes, shelf life, storage, allergens, "
                "shipping, returns and other public product information."
            ),
        },

        "employee": {
            "title":
                "Internal Employee Access",

            "message": (
                "You can access public product information plus "
                "authorized internal OnlyNative knowledge such as "
                "company policies, employee guidelines, leave information "
                "and internal FAQs. Production formulations and other "
                "confidential production information remain restricted."
            ),
        },

        "production_employee": {
            "title":
                "Production Employee Access",

            "message": (
                "You can access public and internal information plus "
                "authorized production knowledge such as formulations, "
                "ingredients, preparation processes and production resources."
            ),
        },

        "manager": {
            "title":
                "Manager Access",

            "message": (
                "You can access the broadest authorized OnlyNative knowledge, "
                "including public information, internal company knowledge "
                "and permitted confidential production information."
            ),
        },
    }

    access_info = (
        role_messages.get(
            role,
            {
                "title":
                    "Authorized Access",

                "message":
                    "You can access OnlyNative information permitted "
                    "for your authenticated account.",
            },
        )
    )

    st.success(
        f"Signed in as **{display_name}**"
    )

    st.markdown(
        f"### {access_info['title']}"
    )

    st.info(
        access_info[
            "message"
        ]
    )

    st.caption(
        "OnlyNativeIQ automatically applies your authenticated "
        "access permissions to every question."
    )


# =============================================================================
# Citations
# =============================================================================

def citation_label(
    citation,
    index: int,
) -> str:
    """
    Build a compact citation label.
    """

    file_name = (
        getattr(
            citation,
            "file_name",
            None,
        )
        or "Unknown source"
    )

    section = (
        getattr(
            citation,
            "section",
            None,
        )
        or ""
    )

    chunk_id = (
        getattr(
            citation,
            "chunk_id",
            None,
        )
        or ""
    )

    parts = [
        f"[{index}]",
        file_name,
    ]

    if section:
        parts.append(
            f"Section: {section}"
        )

    if chunk_id:
        parts.append(
            f"Chunk: {chunk_id}"
        )

    return " · ".join(
        parts
    )


def render_citations(
    citations: Iterable,
) -> None:
    """
    Render trusted source references.
    """

    citations = list(
        citations or []
    )

    if not citations:
        return

    with st.expander(
        "Sources",
        expanded=False,
    ):

        for index, citation in enumerate(
            citations,
            start=1,
        ):

            st.markdown(
                citation_label(
                    citation,
                    index,
                )
            )


# =============================================================================
# Diagnostics
# =============================================================================

def render_diagnostics(
    diagnostics: dict,
    *,
    memory_summary: str = "",
) -> None:
    """
    Render developer-only diagnostics.
    """

    with st.expander(
        "Developer diagnostics",
        expanded=False,
    ):

        st.json(
            diagnostics
        )

        if memory_summary:

            st.markdown(
                "**Hybrid memory summary**"
            )

            st.code(
                memory_summary,
                language=None,
            )


# =============================================================================
# Chat history
# =============================================================================

def render_chat_history(
    messages: list[dict],
    *,
    show_diagnostics: bool,
) -> None:
    """
    Render all visible chat messages saved in Streamlit session state.
    """

    for item in messages:

        role = item[
            "role"
        ]

        with st.chat_message(
            role
        ):

            st.markdown(
                item[
                    "content"
                ]
            )

            if role == "assistant":

                render_citations(
                    item.get(
                        "citations",
                        [],
                    )
                )

                if (
                    show_diagnostics
                    and item.get(
                        "diagnostics"
                    )
                ):

                    render_diagnostics(
                        item[
                            "diagnostics"
                        ],
                        memory_summary=(
                            item.get(
                                "memory_summary",
                                "",
                            )
                        ),
                    )
