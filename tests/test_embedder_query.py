"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_embedder.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Unit tests for chunk/query embedding generation.
================================================================================
"""

from unittest.mock import MagicMock

from src.ingestion.embedder import (
    DocumentEmbedder,
)


def test_embed_query_returns_one_vector() -> None:
    model = MagicMock()

    model.encode.return_value = [
        [0.1, 0.2, 0.3]
    ]

    embedder = DocumentEmbedder(
        model_name="fake-model",
        normalize_embeddings=True,
        model=model,
    )

    vector = embedder.embed_query(
        "Classic Thekua"
    )

    assert vector == [
        0.1,
        0.2,
        0.3,
    ]

    model.encode.assert_called_once_with(
        ["Classic Thekua"],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
