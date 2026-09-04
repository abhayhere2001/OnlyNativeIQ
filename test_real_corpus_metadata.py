"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Integration Test
File         : test_real_corpus_metadata.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Performs an integration test using the real OnlyNativeIQ knowledge base
    and the real metadata registry.

Purpose      :
    - Load all supported PDF/DOCX/TXT knowledge documents.
    - Enrich them using document_registry.json.
    - Validate required metadata.
    - Print a corpus metadata summary.

Notes        :
    This script is intentionally written to remain compatible with Python 3.11.
================================================================================
"""

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.metadata_manager import MetadataManager


def main() -> None:
    """
    Run the real-corpus metadata integration test.
    """

    # -------------------------------------------------------------------------
    # Initialize loader
    # -------------------------------------------------------------------------

    loader = DocumentLoader(
        knowledge_base_path="knowledge_base"
    )

    # -------------------------------------------------------------------------
    # Initialize metadata manager
    # -------------------------------------------------------------------------

    metadata_manager = MetadataManager(
        registry_path="data/metadata/document_registry.json",
        schema_path="data/metadata/metadata_schema.json",
        strict=True,
    )

    print()
    print("=" * 100)
    print("OnlyNativeIQ - Real Corpus Metadata Integration Test")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Step 1: Load actual knowledge-base documents
    # -------------------------------------------------------------------------

    documents = loader.load_all()

    print()
    print(
        f"Documents discovered and loaded: "
        f"{len(documents)}"
    )

    # -------------------------------------------------------------------------
    # Step 2: Enrich documents using the real metadata registry
    # -------------------------------------------------------------------------

    enriched_documents = metadata_manager.enrich_documents(
        documents
    )

    print(
        f"Documents successfully enriched: "
        f"{len(enriched_documents)}"
    )

    print()
    print("-" * 100)

    # -------------------------------------------------------------------------
    # Step 3: Display metadata for every document
    # -------------------------------------------------------------------------

    for index, document in enumerate(
        enriched_documents,
        start=1,
    ):
        metadata_source = document.metadata.get(
            "metadata_source"
        )

        relative_path = document.metadata.get(
            "relative_path"
        )

        print(
            f"{index:02d}. {document.file_name}"
        )

        print(
            f"    Document ID   : "
            f"{document.document_id}"
        )

        print(
            f"    Domain        : "
            f"{document.domain}"
        )

        print(
            f"    Document Type : "
            f"{document.document_type}"
        )

        print(
            f"    Version       : "
            f"{document.version}"
        )

        print(
            f"    Status        : "
            f"{document.status}"
        )

        print(
            f"    Access Level  : "
            f"{document.access_level}"
        )

        print(
            f"    Audience      : "
            f"{document.audience}"
        )

        print(
            f"    Language      : "
            f"{document.language}"
        )

        print(
            f"    Metadata From : "
            f"{metadata_source}"
        )

        print(
            f"    Relative Path : "
            f"{relative_path}"
        )

        print(
            f"    Characters    : "
            f"{len(document.content)}"
        )

        print("-" * 100)

    # -------------------------------------------------------------------------
    # Step 4: Validate important corpus metadata
    # -------------------------------------------------------------------------

    missing_document_ids = [
        document.file_name
        for document in enriched_documents
        if not document.document_id
    ]

    unclassified_documents = [
        document.file_name
        for document in enriched_documents
        if (
            not document.access_level
            or document.access_level.lower()
            == "unclassified"
        )
    ]

    unknown_status_documents = [
        document.file_name
        for document in enriched_documents
        if (
            not document.status
            or document.status.lower()
            == "unknown"
        )
    ]

    missing_audience = [
        document.file_name
        for document in enriched_documents
        if not document.audience
    ]

    # -------------------------------------------------------------------------
    # Step 5: Print corpus validation summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CORPUS VALIDATION SUMMARY")
    print("=" * 100)

    print(
        f"Total Documents       : "
        f"{len(enriched_documents)}"
    )

    print(
        f"Missing Document IDs  : "
        f"{len(missing_document_ids)}"
    )

    print(
        f"Unclassified Access   : "
        f"{len(unclassified_documents)}"
    )

    print(
        f"Unknown Status        : "
        f"{len(unknown_status_documents)}"
    )

    print(
        f"Missing Audience      : "
        f"{len(missing_audience)}"
    )

    # -------------------------------------------------------------------------
    # Step 6: Print details of any metadata problems
    # -------------------------------------------------------------------------

    if missing_document_ids:
        print()
        print("Documents missing Document ID:")

        for file_name in missing_document_ids:
            print(
                f"  - {file_name}"
            )

    if unclassified_documents:
        print()
        print(
            "Documents with unclassified "
            "access level:"
        )

        for file_name in unclassified_documents:
            print(
                f"  - {file_name}"
            )

    if unknown_status_documents:
        print()
        print(
            "Documents with unknown status:"
        )

        for file_name in unknown_status_documents:
            print(
                f"  - {file_name}"
            )

    if missing_audience:
        print()
        print(
            "Documents with missing audience:"
        )

        for file_name in missing_audience:
            print(
                f"  - {file_name}"
            )

    # -------------------------------------------------------------------------
    # Step 7: Final result
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)

    has_errors = (
        bool(missing_document_ids)
        or bool(unclassified_documents)
        or bool(unknown_status_documents)
        or bool(missing_audience)
    )

    if not has_errors:
        print(
            "SUCCESS: Real corpus metadata "
            "validation completed successfully."
        )

    else:
        print(
            "ATTENTION: Some corpus metadata "
            "requires correction before chunking."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()