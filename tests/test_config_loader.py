"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_config_loader.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Tests centralized YAML configuration loading and path resolution.
================================================================================
"""

from pathlib import Path

import pytest

from src.utils.config_loader import (
    ConfigurationError,
    ConfigLoader,
)


def _write_settings(
    tmp_path: Path,
    content: str,
) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    path = config_dir / "settings.yaml"
    path.write_text(content, encoding="utf-8")

    return path


def _valid_settings() -> str:
    return """
project:
  name: OnlyNativeIQ

knowledge_base:
  root: knowledge_base

metadata:
  registry_path: data/metadata/document_registry.json
  schema_path: data/metadata/metadata_schema.json

chunking:
  chunk_size: 700
  chunk_overlap: 100

embeddings:
  provider: sentence_transformers

vector_store:
  provider: chroma

bm25:
  enabled: true

vector_retrieval:
  enabled: true

hybrid_retrieval:
  enabled: true

reranking:
  enabled: true

grounding:
  reject_empty_context: true

llm:
  provider: gemini

memory:
  enabled: true

security:
  enforce_access_control: true

evaluation:
  retrieval_top_k: 5

logging:
  level: INFO
"""


def test_valid_settings_load_successfully(
    tmp_path: Path,
) -> None:
    _write_settings(
        tmp_path,
        _valid_settings(),
    )

    config = ConfigLoader(
        project_root=tmp_path
    )

    assert config.get(
        "project.name"
    ) == "OnlyNativeIQ"

    assert config.get(
        "chunking.chunk_size"
    ) == 700


def test_get_returns_default_for_missing_optional_key(
    tmp_path: Path,
) -> None:
    _write_settings(
        tmp_path,
        _valid_settings(),
    )

    config = ConfigLoader(
        project_root=tmp_path
    )

    assert config.get(
        "chunking.include_section_title",
        default=True,
    ) is True


def test_required_missing_value_raises_error(
    tmp_path: Path,
) -> None:
    _write_settings(
        tmp_path,
        _valid_settings(),
    )

    config = ConfigLoader(
        project_root=tmp_path
    )

    with pytest.raises(
        ConfigurationError
    ):
        config.get(
            "chunking.does_not_exist",
            required=True,
        )


def test_resolve_path_uses_project_root(
    tmp_path: Path,
) -> None:
    _write_settings(
        tmp_path,
        _valid_settings(),
    )

    config = ConfigLoader(
        project_root=tmp_path
    )

    resolved = config.resolve_path(
        "knowledge_base.root"
    )

    assert resolved == (
        tmp_path / "knowledge_base"
    ).resolve()


def test_missing_settings_file_raises_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ConfigurationError
    ):
        ConfigLoader(
            project_root=tmp_path
        )


def test_missing_required_top_level_section_raises_error(
    tmp_path: Path,
) -> None:
    settings = _valid_settings().replace(
        """
reranking:
  enabled: true
""",
        "",
    )

    _write_settings(
        tmp_path,
        settings,
    )

    with pytest.raises(
        ConfigurationError
    ):
        ConfigLoader(
            project_root=tmp_path
        )
