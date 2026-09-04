"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Ingestion Script
File         : ingest_documents.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Command-line entry point for running the OnlyNativeIQ ingestion pipeline.

Responsibilities:
    - Load centralized runtime configuration.
    - Execute the ingestion pipeline.
    - Print a concise pipeline summary.
    - Optionally print generated chunks for diagnostic inspection.

Notes:
    - Chunk printing is controlled through config/settings.yaml.
    - This script is intended for development, diagnostics and manual ingestion.
    - Core ingestion logic remains inside src/ingestion/ingestion_pipeline.py.
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable


# =============================================================================
# PROJECT ROOT SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.schemas.document import DocumentChunk
from src.utils.config_loader import ConfigLoader


def print_pipeline_summary(result) -> None:
    """Print a compact summary of the current ingestion run."""

    print()
    print("=" * 80)
    print("OnlyNativeIQ - Ingestion Pipeline Summary")
    print("=" * 80)
    print(f"Documents loaded     : {result.documents_loaded}")
    print(f"Documents enriched   : {result.documents_enriched}")
    print(f"Sections parsed      : {result.sections_parsed}")
    print(f"Sections cleaned     : {result.sections_cleaned}")
    print(f"Chunks created       : {result.chunks_created}")
    print(f"Embeddings created   : {result.embeddings_created}")
    print(f"Vector records       : {result.vector_records_written}")
    print(f"BM25 records         : {result.bm25_records_written}")
    print(f"Pipeline succeeded   : {result.succeeded}")
    print("=" * 80)


def print_chunks(
    chunks: Iterable[DocumentChunk],
    show_metadata: bool = True,
    max_chunks: int | None = None,
    max_content_chars: int | None = None,
) -> None:
    """
    Print generated chunks for diagnostic inspection.

    Args:
        chunks:
            Generated DocumentChunk objects.

        show_metadata:
            Whether to display chunk metadata.

        max_chunks:
            Maximum number of chunks to print.
            None prints all chunks.

        max_content_chars:
            Maximum characters of chunk content to print.
            None prints the entire chunk.
    """

    all_chunks = list(chunks)

    display_chunks = (
        all_chunks[:max_chunks]
        if max_chunks is not None
        else all_chunks
    )

    print()
    print("=" * 80)
    print(
        f"GENERATED CHUNKS "
        f"({len(display_chunks)} of {len(all_chunks)} displayed)"
    )
    print("=" * 80)

    for index, chunk in enumerate(display_chunks, start=1):
        print()
        print("-" * 80)
        print(f"CHUNK {index}/{len(display_chunks)}")
        print("-" * 80)

        if show_metadata:
            print(f"Chunk ID     : {chunk.chunk_id}")
            print(f"Document ID  : {chunk.document_id}")
            print(f"File Name    : {chunk.file_name}")
            print(f"Domain       : {chunk.domain}")
            print(f"Section      : {chunk.section}")
            print(f"Page Number  : {chunk.page_number}")
            print(f"Access Level : {chunk.access_level}")
            print(f"Audience     : {chunk.audience}")
            print(f"Language     : {chunk.language}")
            print(f"Characters   : {len(chunk.content)}")
            print()

        content = chunk.content

        if (
            max_content_chars is not None
            and len(content) > max_content_chars
        ):
            content = (
                content[:max_content_chars]
                + "\n...[truncated for diagnostics]"
            )

        print(content)

    print()
    print("=" * 80)


def main() -> None:
    """Run the OnlyNativeIQ ingestion pipeline."""

    config = ConfigLoader(
        settings_path="config/settings.yaml",
        project_root=PROJECT_ROOT,
    )

    pipeline = IngestionPipeline.from_config(config)

    result = pipeline.run()

    # Always print the pipeline summary.
    print_pipeline_summary(result)

    # Optional chunk diagnostics.
    print_chunks_enabled = config.get(
        "diagnostics.print_chunks",
        default=False,
    )

    if print_chunks_enabled:
        print_chunks(
            chunks=result.chunks,
            show_metadata=config.get(
                "diagnostics.print_chunk_metadata",
                default=True,
            ),
            max_chunks=config.get(
                "diagnostics.max_chunks_to_print",
                default=None,
            ),
            max_content_chars=config.get(
                "diagnostics.max_chunk_content_chars",
                default=None,
            ),
        )


if __name__ == "__main__":
    main()
