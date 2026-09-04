"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
File         : inspect_classic_thekua.py
Author       : Abhay Kumar Pandey
Description  :
    Diagnostic script to inspect parsing and chunking of the real
    Classic Thekua product guide.
================================================================================
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.ingestion.document_loader import DocumentLoader
from src.ingestion.metadata_manager import MetadataManager
from src.ingestion.document_parser import DocumentParser
from src.ingestion.document_cleaner import DocumentCleaner
from src.ingestion.chunker import DocumentChunker
from src.utils.config_loader import ConfigLoader


TARGET_FILE = "Classic_Thekua_Product_Guide.docx"


def separator(
    character: str = "=",
    width: int = 110,
):
    print(
        character * width
    )


def main():

    config = ConfigLoader(
        settings_path="config/settings.yaml",
        project_root=PROJECT_ROOT,
    )

    loader = DocumentLoader(
        config.resolve_path(
            "knowledge_base.root"
        )
    )

    metadata_manager = MetadataManager(
        registry_path=config.resolve_path(
            "metadata.registry_path"
        ),
        schema_path=config.resolve_path(
            "metadata.schema_path"
        ),
        strict=config.get(
            "metadata.strict_validation",
            default=True,
        ),
    )

    parser = DocumentParser()

    cleaner = DocumentCleaner()

    chunker = DocumentChunker(
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
        include_document_context=True,
    )

    # -------------------------------------------------------------------------
    # Load actual corpus
    # -------------------------------------------------------------------------

    documents = loader.load_all()

    target_documents = [
        document
        for document in documents
        if document.file_name == TARGET_FILE
    ]

    if not target_documents:

        print(
            f"ERROR: {TARGET_FILE} was not found."
        )
        return

    document = target_documents[0]

    separator()
    print(
        "RAW DOCUMENT"
    )
    separator()

    print(
        f"File: {document.file_name}"
    )

    print()

    print(
        document.content
    )

    # -------------------------------------------------------------------------
    # Metadata enrichment
    # -------------------------------------------------------------------------

    document = metadata_manager.enrich_document(
        document
    )

    # -------------------------------------------------------------------------
    # Parsed sections
    # -------------------------------------------------------------------------

    sections = parser.parse_document(
        document
    )

    separator()
    print(
        "PARSED SECTIONS"
    )
    separator()

    for section in sections:

        print()

        separator(
            "-",
            110,
        )

        print(
            f"Section ID   : {section.section_id}"
        )

        print(
            f"Title        : {section.title}"
        )

        print(
            f"Section Type : {section.section_type}"
        )

        print(
            f"Access       : {section.access_level}"
        )

        print(
            f"Audience     : {section.audience}"
        )

        print()

        print(
            section.content
        )

    # -------------------------------------------------------------------------
    # Cleaning
    # -------------------------------------------------------------------------

    cleaned_sections = cleaner.clean_sections(
        sections
    )

    # -------------------------------------------------------------------------
    # Chunks
    # -------------------------------------------------------------------------

    chunks = chunker.chunk_sections(
        cleaned_sections
    )

    separator()
    print(
        "GENERATED CHUNKS"
    )
    separator()

    for chunk in chunks:

        print()

        separator(
            "-",
            110,
        )

        print(
            f"Chunk ID : {chunk.chunk_id}"
        )

        print(
            f"Section  : {chunk.section}"
        )

        print(
            f"Access   : {chunk.access_level}"
        )

        print(
            f"Audience : {chunk.audience}"
        )

        print()

        print(
            chunk.content
        )


if __name__ == "__main__":
    main()