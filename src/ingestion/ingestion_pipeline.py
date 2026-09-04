"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Document Ingestion
File         : ingestion_pipeline.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Orchestrates the OnlyNativeIQ ingestion pipeline.

Current Pipeline:
    1. Load documents.
    2. Enrich metadata.
    3. Parse logical sections.
    4. Clean parsed sections.
    5. Create access-safe chunks.
    6. Optionally generate embeddings.
    7. Optionally persist embeddings to Chroma/FAISS.
    8. Optionally build/persist BM25 index.

Future Extensions:
    9. Hybrid retrieval and reranking.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_cleaner import DocumentCleaner
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.document_parser import DocumentParser
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.metadata_manager import MetadataManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store_factory import (
    BaseVectorStore,
    VectorStoreFactory,
)
from src.schemas.document import (
    DocumentChunk,
    EmbeddedChunk,
    ParsedSection,
    SourceDocument,
)
from src.utils.config_loader import ConfigLoader


@dataclass(slots=True)
class IngestionResult:
    documents_loaded: int
    documents_enriched: int
    sections_parsed: int
    sections_cleaned: int
    chunks_created: int
    embeddings_created: int
    vector_records_written: int
    bm25_records_written: int

    documents: list[SourceDocument] = field(
        default_factory=list
    )
    cleaned_sections: list[ParsedSection] = field(
        default_factory=list
    )
    chunks: list[DocumentChunk] = field(
        default_factory=list
    )
    embedded_chunks: list[EmbeddedChunk] = field(
        default_factory=list
    )

    embeddings_enabled: bool = True
    vector_store_enabled: bool = True
    bm25_enabled: bool = True

    @property
    def succeeded(self) -> bool:
        success = (
            self.documents_loaded
            == self.documents_enriched
            and self.sections_parsed
            == self.sections_cleaned
            and (
                self.sections_cleaned == 0
                or self.chunks_created > 0
            )
        )

        if self.embeddings_enabled:
            success = (
                success
                and self.chunks_created
                == self.embeddings_created
            )

        if (
            self.vector_store_enabled
            and self.embeddings_enabled
        ):
            success = (
                success
                and self.vector_records_written
                == self.embeddings_created
            )

        if self.bm25_enabled:
            success = (
                success
                and self.bm25_records_written
                == self.chunks_created
            )

        return success


class IngestionPipeline:
    def __init__(
        self,
        knowledge_base_path: str | Path = "knowledge_base",
        registry_path: str | Path = (
            "data/metadata/document_registry.json"
        ),
        schema_path: str | Path = (
            "data/metadata/metadata_schema.json"
        ),
        strict_metadata: bool = True,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        include_section_title: bool = True,
        embeddings_enabled: bool = True,
        embedding_model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_batch_size: int = 32,
        normalize_embeddings: bool = True,
        embedding_device: str | None = None,
        embedding_show_progress_bar: bool = False,
        vector_store_enabled: bool = True,
        bm25_enabled: bool = True,
        loader: DocumentLoader | None = None,
        metadata_manager: MetadataManager | None = None,
        parser: DocumentParser | None = None,
        cleaner: DocumentCleaner | None = None,
        chunker: DocumentChunker | None = None,
        embedder: DocumentEmbedder | None = None,
        vector_store: BaseVectorStore | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        self.embeddings_enabled = (
            embeddings_enabled
        )
        self.vector_store_enabled = (
            vector_store_enabled
        )
        self.bm25_enabled = (
            bm25_enabled
        )

        self.loader = loader or DocumentLoader(
            knowledge_base_path
        )

        self.metadata_manager = (
            metadata_manager
            or MetadataManager(
                registry_path=registry_path,
                schema_path=schema_path,
                strict=strict_metadata,
            )
        )

        self.parser = parser or DocumentParser()
        self.cleaner = cleaner or DocumentCleaner()

        self.chunker = (
            chunker
            or DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                include_section_title=include_section_title,
            )
        )

        self.embedder = (
            embedder
            or DocumentEmbedder(
                model_name=embedding_model_name,
                batch_size=embedding_batch_size,
                normalize_embeddings=normalize_embeddings,
                device=embedding_device,
                show_progress_bar=embedding_show_progress_bar,
            )
        )

        self.vector_store = vector_store
        self.bm25_retriever = (
            bm25_retriever
        )

    @classmethod
    def from_config(
        cls,
        config: ConfigLoader,
    ) -> "IngestionPipeline":

        vector_store_enabled = config.get(
            "vector_store.enabled",
            default=True,
        )

        vector_store = None

        if vector_store_enabled:
            vector_store = (
                VectorStoreFactory.from_config(
                    config
                )
            )

        bm25_enabled = config.get(
            "bm25.enabled",
            default=True,
        )

        bm25_retriever = None

        if bm25_enabled:
            bm25_retriever = (
                BM25Retriever.from_config(
                    config
                )
            )

        return cls(
            knowledge_base_path=config.resolve_path(
                "knowledge_base.root"
            ),
            registry_path=config.resolve_path(
                "metadata.registry_path"
            ),
            schema_path=config.resolve_path(
                "metadata.schema_path"
            ),
            strict_metadata=config.get(
                "metadata.strict_validation",
                default=True,
            ),
            chunk_size=config.get(
                "chunking.chunk_size",
                required=True,
            ),
            chunk_overlap=config.get(
                "chunking.chunk_overlap",
                required=True,
            ),
            include_section_title=config.get(
                "chunking.include_section_title",
                default=True,
            ),
            embeddings_enabled=config.get(
                "embeddings.enabled",
                default=True,
            ),
            embedding_model_name=config.get(
                "embeddings.model_name",
                required=True,
            ),
            embedding_batch_size=config.get(
                "embeddings.batch_size",
                default=32,
            ),
            normalize_embeddings=config.get(
                "embeddings.normalize_embeddings",
                default=True,
            ),
            embedding_device=config.get(
                "embeddings.device",
                default=None,
            ),
            embedding_show_progress_bar=config.get(
                "embeddings.show_progress_bar",
                default=False,
            ),
            vector_store_enabled=vector_store_enabled,
            bm25_enabled=bm25_enabled,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
        )

    def run(self) -> IngestionResult:
        loaded_documents = (
            self.loader.load_all()
        )

        enriched_documents = (
            self.metadata_manager.enrich_documents(
                loaded_documents
            )
        )

        parsed_sections = (
            self.parser.parse_documents(
                enriched_documents
            )
        )

        cleaned_sections = (
            self.cleaner.clean_sections(
                parsed_sections
            )
        )

        chunks = (
            self.chunker.chunk_sections(
                cleaned_sections
            )
        )

        embedded_chunks: list[
            EmbeddedChunk
        ] = []

        if self.embeddings_enabled:
            embedded_chunks = (
                self.embedder.embed_chunks(
                    chunks
                )
            )

        vector_records_written = 0

        if (
            self.vector_store_enabled
            and self.embeddings_enabled
        ):
            if self.vector_store is None:
                raise RuntimeError(
                    "Vector store is enabled but not configured."
                )

            vector_records_written = (
                self.vector_store.add_embeddings(
                    embedded_chunks
                )
            )

        bm25_records_written = 0

        if self.bm25_enabled:
            if self.bm25_retriever is None:
                raise RuntimeError(
                    "BM25 is enabled but BM25Retriever is not configured."
                )

            bm25_records_written = (
                self.bm25_retriever.build_index(
                    chunks
                )
            )

        return IngestionResult(
            documents_loaded=len(
                loaded_documents
            ),
            documents_enriched=len(
                enriched_documents
            ),
            sections_parsed=len(
                parsed_sections
            ),
            sections_cleaned=len(
                cleaned_sections
            ),
            chunks_created=len(
                chunks
            ),
            embeddings_created=len(
                embedded_chunks
            ),
            vector_records_written=(
                vector_records_written
            ),
            bm25_records_written=(
                bm25_records_written
            ),
            documents=enriched_documents,
            cleaned_sections=cleaned_sections,
            chunks=chunks,
            embedded_chunks=embedded_chunks,
            embeddings_enabled=(
                self.embeddings_enabled
            ),
            vector_store_enabled=(
                self.vector_store_enabled
            ),
            bm25_enabled=(
                self.bm25_enabled
            ),
        )

    def run_summary(
        self,
    ) -> dict[str, int | bool]:
        result = self.run()

        return {
            "documents_loaded":
                result.documents_loaded,
            "documents_enriched":
                result.documents_enriched,
            "sections_parsed":
                result.sections_parsed,
            "sections_cleaned":
                result.sections_cleaned,
            "chunks_created":
                result.chunks_created,
            "embeddings_created":
                result.embeddings_created,
            "vector_records_written":
                result.vector_records_written,
            "bm25_records_written":
                result.bm25_records_written,
            "succeeded":
                result.succeeded,
        }
