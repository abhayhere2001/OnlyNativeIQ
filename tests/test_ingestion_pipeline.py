"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_ingestion_pipeline.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Tests ingestion orchestration through vector and BM25 indexing.
================================================================================
"""

from unittest.mock import MagicMock

from src.ingestion.ingestion_pipeline import (
    IngestionPipeline,
)
from src.schemas.document import (
    DocumentChunk,
    EmbeddedChunk,
    ParsedSection,
    SourceDocument,
)


def _source() -> SourceDocument:
    return SourceDocument(
        document_id="ON-TEST-001",
        file_name="Test.txt",
        file_path="knowledge_base/products/Test.txt",
        file_extension=".txt",
        domain="products",
        content="Description\nTest content.",
        document_type="product_guide",
        version="2.0",
        status="approved",
        access_level="public",
        audience=["customer"],
        language="en",
    )


def _section() -> ParsedSection:
    return ParsedSection(
        section_id="ON-TEST-001-S001",
        document_id="ON-TEST-001",
        title="Description",
        content="Test content.",
        section_index=1,
        file_name="Test.txt",
        domain="products",
        access_level="public",
        audience=["customer"],
        language="en",
    )


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="ON-TEST-001-S001-C001",
        document_id="ON-TEST-001",
        content="Description\nTest content.",
        file_name="Test.txt",
        domain="products",
        chunk_index=1,
        access_level="public",
        audience=["customer"],
        language="en",
    )


def test_pipeline_builds_vector_and_bm25_indexes() -> None:
    loader = MagicMock()
    metadata_manager = MagicMock()
    parser = MagicMock()
    cleaner = MagicMock()
    chunker = MagicMock()
    embedder = MagicMock()
    vector_store = MagicMock()
    bm25_retriever = MagicMock()

    source = _source()
    section = _section()
    chunk = _chunk()

    embedded = EmbeddedChunk(
        chunk=chunk,
        embedding=[0.1, 0.2],
        model_name="fake-model",
        embedding_dimension=2,
    )

    loader.load_all.return_value = [
        source
    ]

    metadata_manager.enrich_documents.return_value = [
        source
    ]

    parser.parse_documents.return_value = [
        section
    ]

    cleaner.clean_sections.return_value = [
        section
    ]

    chunker.chunk_sections.return_value = [
        chunk
    ]

    embedder.embed_chunks.return_value = [
        embedded
    ]

    vector_store.add_embeddings.return_value = 1
    bm25_retriever.build_index.return_value = 1

    pipeline = IngestionPipeline(
        loader=loader,
        metadata_manager=metadata_manager,
        parser=parser,
        cleaner=cleaner,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        embeddings_enabled=True,
        vector_store_enabled=True,
        bm25_enabled=True,
    )

    result = pipeline.run()

    vector_store.add_embeddings.assert_called_once_with(
        [embedded]
    )

    bm25_retriever.build_index.assert_called_once_with(
        [chunk]
    )

    assert result.vector_records_written == 1
    assert result.bm25_records_written == 1
    assert result.succeeded is True
