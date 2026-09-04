"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_llm_factory.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Unit tests for OnlyNativeIQ LLM provider selection and configuration.

Test Coverage:
    - Gemini provider selection
    - Groq provider selection
    - OpenAI provider selection
    - Unsupported provider
    - Missing API keys
    - Invalid temperature
    - Invalid max_tokens
    - None config validation

Important:
    These tests mock provider constructors and do not make real API calls.
================================================================================
"""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.llm_factory import (
    LLMConfigurationError,
    LLMFactory,
    UnsupportedLLMProviderError,
)


# =============================================================================
# Config helper
# =============================================================================

def _config(
    provider: str = "gemini",
    temperature=0.0,
    max_tokens=800,
) -> MagicMock:
    """
    Build a fake ConfigLoader with predictable values.
    """

    config = MagicMock()

    values = {
        "llm.provider":
            provider,

        "llm.temperature":
            temperature,

        "llm.max_tokens":
            max_tokens,

        "llm.gemini.model":
            "gemini-2.5-flash",

        "llm.groq.model":
            "llama-3.3-70b-versatile",

        "llm.openai.model":
            "gpt-4o-mini",
    }

    def get_value(
        key,
        default=None,
        required=False,
    ):
        if key in values:
            return values[key]

        if required:
            raise RuntimeError(
                f"Missing required config: {key}"
            )

        return default

    config.get.side_effect = (
        get_value
    )

    return config


# =============================================================================
# Constructor validation
# =============================================================================

def test_none_config_is_rejected() -> None:

    with pytest.raises(
        ValueError,
        match="config cannot be None",
    ):
        LLMFactory(
            config=None
        )


# =============================================================================
# Unsupported provider
# =============================================================================

def test_unsupported_provider_is_rejected() -> None:

    factory = LLMFactory(
        _config(
            provider="unsupported"
        )
    )

    with pytest.raises(
        UnsupportedLLMProviderError
    ):
        factory.create()


# =============================================================================
# Gemini
# =============================================================================

@patch.dict(
    "os.environ",
    {
        "GOOGLE_API_KEY":
            "fake-google-key"
    },
    clear=False,
)
@patch(
    "langchain_google_genai."
    "ChatGoogleGenerativeAI"
)
def test_gemini_provider_is_created(
    mock_gemini,
) -> None:

    model_instance = MagicMock()

    mock_gemini.return_value = (
        model_instance
    )

    factory = LLMFactory(
        _config(
            provider="gemini"
        )
    )

    result = factory.create()

    assert result is model_instance

    mock_gemini.assert_called_once_with(
        model="gemini-2.5-flash",
        google_api_key="fake-google-key",
        temperature=0.0,
        max_output_tokens=800,
    )


def test_gemini_missing_api_key_is_rejected(
    monkeypatch,
) -> None:

    monkeypatch.delenv(
        "GOOGLE_API_KEY",
        raising=False,
    )

    factory = LLMFactory(
        _config(
            provider="gemini"
        )
    )

    with pytest.raises(
        LLMConfigurationError,
        match="GOOGLE_API_KEY",
    ):
        factory.create()


# =============================================================================
# Groq
# =============================================================================

@patch.dict(
    "os.environ",
    {
        "GROQ_API_KEY":
            "fake-groq-key"
    },
    clear=False,
)
@patch(
    "langchain_groq.ChatGroq"
)
def test_groq_provider_is_created(
    mock_groq,
) -> None:

    model_instance = MagicMock()

    mock_groq.return_value = (
        model_instance
    )

    factory = LLMFactory(
        _config(
            provider="groq"
        )
    )

    result = factory.create()

    assert result is model_instance

    mock_groq.assert_called_once_with(
        model=(
            "llama-3.3-70b-versatile"
        ),
        api_key="fake-groq-key",
        temperature=0.0,
        max_tokens=800,
    )


def test_groq_missing_api_key_is_rejected(
    monkeypatch,
) -> None:

    monkeypatch.delenv(
        "GROQ_API_KEY",
        raising=False,
    )

    factory = LLMFactory(
        _config(
            provider="groq"
        )
    )

    with pytest.raises(
        LLMConfigurationError,
        match="GROQ_API_KEY",
    ):
        factory.create()


# =============================================================================
# OpenAI
# =============================================================================

@patch.dict(
    "os.environ",
    {
        "OPENAI_API_KEY":
            "fake-openai-key"
    },
    clear=False,
)
@patch(
    "langchain_openai.ChatOpenAI"
)
def test_openai_provider_is_created(
    mock_openai,
) -> None:

    model_instance = MagicMock()

    mock_openai.return_value = (
        model_instance
    )

    factory = LLMFactory(
        _config(
            provider="openai"
        )
    )

    result = factory.create()

    assert result is model_instance

    mock_openai.assert_called_once_with(
        model="gpt-4o-mini",
        api_key="fake-openai-key",
        temperature=0.0,
        max_tokens=800,
    )


def test_openai_missing_api_key_is_rejected(
    monkeypatch,
) -> None:

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    factory = LLMFactory(
        _config(
            provider="openai"
        )
    )

    with pytest.raises(
        LLMConfigurationError,
        match="OPENAI_API_KEY",
    ):
        factory.create()


# =============================================================================
# Temperature validation
# =============================================================================

@pytest.mark.parametrize(
    "temperature",
    [
        -0.1,
        2.1,
        "invalid",
    ],
)
def test_invalid_temperature_is_rejected(
    temperature,
) -> None:

    factory = LLMFactory(
        _config(
            provider="gemini",
            temperature=temperature,
        )
    )

    # Validate directly so this test does
    # not depend on provider/API-key setup.
    with pytest.raises(
        LLMConfigurationError
    ):
        factory._temperature()


@pytest.mark.parametrize(
    "temperature",
    [
        0.0,
        0.5,
        1.0,
        2.0,
    ],
)
def test_valid_temperature_is_accepted(
    temperature,
) -> None:

    factory = LLMFactory(
        _config(
            temperature=temperature
        )
    )

    assert (
        factory._temperature()
        == float(temperature)
    )


# =============================================================================
# Max-token validation
# =============================================================================

@pytest.mark.parametrize(
    "max_tokens",
    [
        0,
        -1,
        "invalid",
    ],
)
def test_invalid_max_tokens_is_rejected(
    max_tokens,
) -> None:

    factory = LLMFactory(
        _config(
            max_tokens=max_tokens
        )
    )

    with pytest.raises(
        LLMConfigurationError
    ):
        factory._max_tokens()


@pytest.mark.parametrize(
    "max_tokens",
    [
        1,
        100,
        800,
        2000,
    ],
)
def test_valid_max_tokens_is_accepted(
    max_tokens,
) -> None:

    factory = LLMFactory(
        _config(
            max_tokens=max_tokens
        )
    )

    assert (
        factory._max_tokens()
        == max_tokens
    )


# =============================================================================
# Provider normalization
# =============================================================================

@patch.dict(
    "os.environ",
    {
        "GOOGLE_API_KEY":
            "fake-google-key"
    },
    clear=False,
)
@patch(
    "langchain_google_genai."
    "ChatGoogleGenerativeAI"
)
def test_provider_name_is_case_insensitive(
    mock_gemini,
) -> None:

    mock_gemini.return_value = (
        MagicMock()
    )

    factory = LLMFactory(
        _config(
            provider="  GeMiNi  "
        )
    )

    factory.create()

    mock_gemini.assert_called_once()