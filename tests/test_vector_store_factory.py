"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_vector_store_factory.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Tests provider selection and vector-store factory validation.
================================================================================
"""

import pytest

from src.retrieval.vector_store_factory import (
    UnsupportedVectorStoreError,
    VectorStoreError,
    VectorStoreFactory,
)


def test_unsupported_provider_raises_error() -> None:
    with pytest.raises(
        UnsupportedVectorStoreError
    ):
        VectorStoreFactory.create(
            provider="unknown"
        )


def test_chroma_requires_persist_directory() -> None:
    with pytest.raises(
        VectorStoreError
    ):
        VectorStoreFactory.create(
            provider="chroma"
        )


def test_faiss_requires_persist_directory() -> None:
    with pytest.raises(
        VectorStoreError
    ):
        VectorStoreFactory.create(
            provider="faiss"
        )
