"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Configuration
File         : config_loader.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Loads and validates OnlyNativeIQ runtime configuration from YAML files.

Responsibilities:
    - Load settings.yaml.
    - Provide safe access to nested configuration values.
    - Validate mandatory configuration sections.
    - Support project-root-relative file paths.
    - Keep environment secrets separate from YAML configuration.

Notes:
    - API keys belong in .env, not settings.yaml.
    - Business/document metadata belongs in document_registry.json.
    - Technical/runtime tuning belongs in settings.yaml.
================================================================================
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")


class ConfigurationError(RuntimeError):
    """Raised when application configuration is missing or invalid."""


class ConfigLoader:
    """Load and expose OnlyNativeIQ configuration."""

    REQUIRED_TOP_LEVEL_SECTIONS = {
        "project",
        "knowledge_base",
        "metadata",
        "chunking",
        "embeddings",
        "vector_store",
        "bm25",
        "vector_retrieval",
        "hybrid_retrieval",
        "reranking",
        "grounding",
        "llm",
        "memory",
        "security",
        "evaluation",
        "logging",
    }

    def __init__(
        self,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
        project_root: str | Path | None = None,
    ) -> None:
        self.settings_path = Path(settings_path)

        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )

        self._settings = self._load_settings()
        self._validate_top_level_sections()

    def get_settings(self) -> dict[str, Any]:
        """Return a defensive copy of all loaded settings."""
        return deepcopy(self._settings)

    def get_section(
        self,
        section_name: str,
    ) -> dict[str, Any]:
        """Return one configuration section."""

        section = self._settings.get(section_name)

        if not isinstance(section, dict):
            raise ConfigurationError(
                f"Configuration section '{section_name}' "
                "is missing or is not a mapping."
            )

        return deepcopy(section)

    def get(
        self,
        dotted_path: str,
        default: Any = None,
        required: bool = False,
    ) -> Any:
        """
        Retrieve a nested configuration value.

        Example:
            config.get("chunking.chunk_size")
        """

        current: Any = self._settings

        for key in dotted_path.split("."):
            if not isinstance(current, dict):
                current = None
                break

            current = current.get(key)

            if current is None:
                break

        if current is None:
            if required:
                raise ConfigurationError(
                    f"Required configuration value "
                    f"'{dotted_path}' was not found."
                )

            return default

        return current

    def resolve_path(
        self,
        dotted_path: str,
        required: bool = True,
    ) -> Path:
        """
        Resolve a configured path against project_root.
        """

        value = self.get(
            dotted_path,
            required=required,
        )

        if value is None:
            raise ConfigurationError(
                f"Path configuration '{dotted_path}' is missing."
            )

        path = Path(str(value))

        if path.is_absolute():
            return path

        return (self.project_root / path).resolve()

    def _load_settings(self) -> dict[str, Any]:
        """Load YAML settings."""

        resolved_settings_path = (
            self.settings_path
            if self.settings_path.is_absolute()
            else self.project_root / self.settings_path
        )

        if not resolved_settings_path.exists():
            raise ConfigurationError(
                f"Settings file does not exist: "
                f"{resolved_settings_path}"
            )

        try:
            with resolved_settings_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML in settings file "
                f"'{resolved_settings_path}': {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ConfigurationError(
                "settings.yaml must contain a top-level mapping."
            )

        return payload

    def _validate_top_level_sections(self) -> None:
        """Validate mandatory configuration sections."""

        missing = (
            self.REQUIRED_TOP_LEVEL_SECTIONS
            - set(self._settings.keys())
        )

        if missing:
            raise ConfigurationError(
                "Missing required configuration section(s): "
                f"{', '.join(sorted(missing))}"
            )
