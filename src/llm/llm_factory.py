"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : LLM
File         : llm_factory.py
Author       : Abhay Kumar Pandey
Created On   : 04-Sep-2026

Description  :
    Creates the configured LangChain chat model used by OnlyNativeIQ.

Supported Providers:
    - Google Gemini
    - Groq
    - OpenAI

Responsibilities:
    - Read provider/model configuration from ConfigLoader.
    - Read API keys only from environment variables.
    - Validate provider configuration.
    - Instantiate a LangChain BaseChatModel.
    - Keep provider-specific code outside the RAG/generation layers.

Security:
    API keys must never be stored in source code or settings.yaml.
================================================================================
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from src.utils.config_loader import (
    ConfigLoader,
)


# =============================================================================
# Exceptions
# =============================================================================

class LLMConfigurationError(RuntimeError):
    """
    Raised when LLM configuration is invalid or incomplete.
    """


class UnsupportedLLMProviderError(
    LLMConfigurationError
):
    """
    Raised when an unsupported provider is configured.
    """


# =============================================================================
# Factory
# =============================================================================

class LLMFactory:
    """
    Factory for creating the OnlyNativeIQ chat model.
    """

    SUPPORTED_PROVIDERS = {
        "gemini",
        "groq",
        "openai",
    }

    def __init__(
        self,
        config: ConfigLoader,
    ) -> None:

        if config is None:
            raise ValueError(
                "config cannot be None."
            )

        self.config = config

    # =========================================================================
    # Public API
    # =========================================================================

    def create(
        self,
    ) -> BaseChatModel:
        """
        Create the configured chat model.

        Returns
        -------
        BaseChatModel
            LangChain-compatible chat model.
        """

        provider = (
            self.config.get(
                "llm.provider",
                required=True,
            )
        )

        if not isinstance(
            provider,
            str,
        ):
            raise LLMConfigurationError(
                "llm.provider must be a string."
            )

        provider = (
            provider.strip().lower()
        )

        if (
            provider
            not in self.SUPPORTED_PROVIDERS
        ):
            raise UnsupportedLLMProviderError(
                f"Unsupported LLM provider: "
                f"'{provider}'. "
                f"Supported providers: "
                f"{sorted(self.SUPPORTED_PROVIDERS)}"
            )

        if provider == "gemini":
            return self._create_gemini()

        if provider == "groq":
            return self._create_groq()

        if provider == "openai":
            return self._create_openai()

        # Defensive fallback.
        raise UnsupportedLLMProviderError(
            f"Unsupported LLM provider: "
            f"'{provider}'."
        )

    # =========================================================================
    # Gemini
    # =========================================================================

    def _create_gemini(
        self,
    ) -> BaseChatModel:
        """
        Create Google Gemini chat model.
        """

        try:
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
            )

        except ImportError as exc:
            raise LLMConfigurationError(
                "Gemini provider requires "
                "'langchain-google-genai'."
            ) from exc

        api_key = self._require_api_key(
            "GOOGLE_API_KEY"
        )

        model = self.config.get(
            "llm.gemini.model",
            required=True,
        )

        temperature = self._temperature()

        max_tokens = self._max_tokens()

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # =========================================================================
    # Groq
    # =========================================================================

    def _create_groq(
        self,
    ) -> BaseChatModel:
        """
        Create Groq chat model.
        """

        try:
            from langchain_groq import (
                ChatGroq,
            )

        except ImportError as exc:
            raise LLMConfigurationError(
                "Groq provider requires "
                "'langchain-groq'."
            ) from exc

        api_key = self._require_api_key(
            "GROQ_API_KEY"
        )

        model = self.config.get(
            "llm.groq.model",
            required=True,
        )

        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=self._temperature(),
            max_tokens=self._max_tokens(),
        )

    # =========================================================================
    # OpenAI
    # =========================================================================

    def _create_openai(
        self,
    ) -> BaseChatModel:
        """
        Create OpenAI chat model.
        """

        try:
            from langchain_openai import (
                ChatOpenAI,
            )

        except ImportError as exc:
            raise LLMConfigurationError(
                "OpenAI provider requires "
                "'langchain-openai'."
            ) from exc

        api_key = self._require_api_key(
            "OPENAI_API_KEY"
        )

        model = self.config.get(
            "llm.openai.model",
            required=True,
        )

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=self._temperature(),
            max_tokens=self._max_tokens(),
        )

    # =========================================================================
    # Shared configuration
    # =========================================================================

    def _temperature(
        self,
    ) -> float:
        """
        Read and validate generation temperature.
        """

        value = self.config.get(
            "llm.temperature",
            default=0.0,
        )

        try:
            temperature = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise LLMConfigurationError(
                "llm.temperature must be numeric."
            ) from exc

        if not (
            0.0 <= temperature <= 2.0
        ):
            raise LLMConfigurationError(
                "llm.temperature must be "
                "between 0.0 and 2.0."
            )

        return temperature

    def _max_tokens(
        self,
    ) -> int:
        """
        Read and validate maximum output tokens.
        """

        value = self.config.get(
            "llm.max_tokens",
            default=800,
        )

        try:
            max_tokens = int(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise LLMConfigurationError(
                "llm.max_tokens must be "
                "an integer."
            ) from exc

        if max_tokens <= 0:
            raise LLMConfigurationError(
                "llm.max_tokens must be "
                "greater than zero."
            )

        return max_tokens

    # =========================================================================
    # API-key validation
    # =========================================================================

    @staticmethod
    def _require_api_key(
        environment_variable: str,
    ) -> str:
        """
        Return an API key from the environment.

        Raises
        ------
        LLMConfigurationError
            When the required environment variable is absent.
        """

        value = os.getenv(
            environment_variable
        )

        if (
            value is None
            or not value.strip()
        ):
            raise LLMConfigurationError(
                f"Required environment variable "
                f"'{environment_variable}' is not set."
            )

        return value.strip()