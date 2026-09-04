"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_metadata.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Unit tests for metadata_manager.py.

Coverage:
    - Registry loading
    - Rich registry matching
    - Legacy Corpus v2.0 registry compatibility
    - Header metadata extraction
    - Metadata precedence
    - Audience normalization
    - Required-field validation
    - Missing registry handling
    - Duplicate registry detection
================================================================================
"""

from pathlib import Path
import json

import pytest

from src.ingestion.metadata_manager import (
    DuplicateDocumentMetadataError,
    MetadataManager,
    MetadataRegistryError,
    MetadataValidationError,
)
from src.schemas.document import SourceDocument


def _create_source_document(
    tmp_path: Path,
    *,
    file_name: str = "Classic_Thekua_Product_Guide.docx",
    domain: str = "products",
    content: str = "",
) -> SourceDocument:
    """
    Create a reusable SourceDocument for metadata tests.
    """

    file_path = (
        tmp_path
        / "knowledge_base"
        / domain
        / file_name
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        "dummy",
        encoding="utf-8",
    )

    return SourceDocument(
        document_id=None,
        file_name=file_name,
        file_path=str(file_path),
        file_extension=file_path.suffix.lower(),
        domain=domain,
        content=content,
        metadata={
            "relative_path": (
                f"{domain}/{file_name}"
            )
        },
    )


def _write_registry(
    tmp_path: Path,
    payload: dict,
) -> Path:
    """
    Write document_registry.json for a test.
    """

    registry_path = (
        tmp_path
        / "data"
        / "metadata"
        / "document_registry.json"
    )

    registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return registry_path


def _write_schema(
    tmp_path: Path,
    required_fields: list[str] | None = None,
) -> Path:
    """
    Write metadata_schema.json for a test.
    """

    schema_path = (
        tmp_path
        / "data"
        / "metadata"
        / "metadata_schema.json"
    )

    schema_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "required_document_metadata":
            required_fields
            or [
                "document_id",
                "file_name",
                "document_type",
                "domain",
                "version",
                "status",
                "access_level",
                "audience",
                "language",
            ]
    }

    schema_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return schema_path


def test_enrich_document_from_rich_registry(
    tmp_path: Path,
) -> None:
    """
    Rich registry metadata should enrich SourceDocument fields.
    """

    source = _create_source_document(
        tmp_path,
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": [
                {
                    "document_id":
                        "ON-PROD-0001",
                    "file_name":
                        "Classic_Thekua_Product_Guide.docx",
                    "relative_path":
                        "products/Classic_Thekua_Product_Guide.docx",
                    "document_type":
                        "product_guide",
                    "domain":
                        "products",
                    "version":
                        "2.0",
                    "status":
                        "approved",
                    "access_level":
                        "public",
                    "audience": [
                        "customer",
                        "employee",
                    ],
                    "language":
                        "en",
                }
            ]
        },
    )

    schema_path = _write_schema(
        tmp_path,
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=True,
    )

    enriched = manager.enrich_document(
        source,
    )

    assert enriched.document_id == (
        "ON-PROD-0001"
    )

    assert enriched.document_type == (
        "product_guide"
    )

    assert enriched.version == "2.0"
    assert enriched.status == "approved"
    assert enriched.access_level == "public"

    assert enriched.audience == [
        "customer",
        "employee",
    ]

    assert enriched.language == "en"

    assert enriched.metadata[
        "metadata_source"
    ] == "registry"


def test_legacy_registry_file_path_is_supported(
    tmp_path: Path,
) -> None:
    """
    Corpus v2.0 legacy entries using 'file' should still match
    the current knowledge_base structure.
    """

    source = _create_source_document(
        tmp_path,
        file_name=(
            "Classic_Thekua_Product_Guide.docx"
        ),
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": [
                {
                    "file":
                        "01_raw/products/"
                        "Classic_Thekua_Product_Guide.docx",
                    "format":
                        "DOCX",
                    "ingestible":
                        True,
                }
            ]
        },
    )

    schema_path = _write_schema(
        tmp_path,
        required_fields=[
            "file_name",
            "domain",
            "language",
        ],
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=True,
    )

    entry = manager.get_registry_entry(
        source,
    )

    assert entry is not None

    assert entry["file_name"] == (
        "Classic_Thekua_Product_Guide.docx"
    )

    assert entry["relative_path"] == (
        "products/"
        "Classic_Thekua_Product_Guide.docx"
    )


def test_header_metadata_is_used_when_registry_is_incomplete(
    tmp_path: Path,
) -> None:
    """
    Metadata embedded in a document header should be used when
    registry metadata is missing.
    """

    content = """
Document ID: ON-PROD-0001
Version: 2.0
Status: Approved
Access Level: Public
Audience: Customer, Employee
Language: en

Classic Thekua
"""

    source = _create_source_document(
        tmp_path,
        content=content,
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": []
        },
    )

    schema_path = _write_schema(
        tmp_path,
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=True,
    )

    enriched = manager.enrich_document(
        source,
    )

    assert enriched.document_id == (
        "ON-PROD-0001"
    )

    assert enriched.version == "2.0"

    assert enriched.status.lower() == (
        "approved"
    )

    assert enriched.access_level.lower() == (
        "public"
    )

    assert enriched.audience == [
        "customer",
        "employee",
    ]

    assert enriched.language == "en"

    assert enriched.metadata[
        "metadata_source"
    ] == "document_header"


def test_registry_metadata_has_priority_over_document_header(
    tmp_path: Path,
) -> None:
    """
    Explicit registry metadata should override conflicting
    source-document header values.
    """

    content = """
Document ID: HEADER-ID
Version: 1.0
Status: Draft
Access Level: Public
Audience: Customer
"""

    source = _create_source_document(
        tmp_path,
        content=content,
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": [
                {
                    "document_id":
                        "ON-PROD-0001",
                    "file_name":
                        source.file_name,
                    "document_type":
                        "product_guide",
                    "domain":
                        "products",
                    "version":
                        "2.0",
                    "status":
                        "approved",
                    "access_level":
                        "internal",
                    "audience": [
                        "employee"
                    ],
                    "language":
                        "en",
                }
            ]
        },
    )

    schema_path = _write_schema(
        tmp_path,
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=True,
    )

    enriched = manager.enrich_document(
        source,
    )

    assert enriched.document_id == (
        "ON-PROD-0001"
    )

    assert enriched.version == "2.0"

    assert enriched.status == "approved"

    assert enriched.access_level == (
        "internal"
    )

    assert enriched.audience == [
        "employee"
    ]


def test_audience_string_is_normalized(
    tmp_path: Path,
) -> None:
    """
    Comma-separated audience metadata should become a normalized list.
    """

    source = _create_source_document(
        tmp_path,
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": [
                {
                    "document_id":
                        "ON-PROD-0001",
                    "file_name":
                        source.file_name,
                    "document_type":
                        "product_guide",
                    "domain":
                        "products",
                    "version":
                        "2.0",
                    "status":
                        "approved",
                    "access_level":
                        "public",
                    "audience":
                        "Customer, Production Employee",
                    "language":
                        "en",
                }
            ]
        },
    )

    schema_path = _write_schema(
        tmp_path,
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=True,
    )

    enriched = manager.enrich_document(
        source,
    )

    assert enriched.audience == [
        "customer",
        "production_employee",
    ]


def test_missing_required_metadata_raises_in_strict_mode(
    tmp_path: Path,
) -> None:
    """
    Strict mode should stop ingestion when mandatory metadata
    cannot be resolved.
    """

    source = _create_source_document(
        tmp_path,
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": []
        },
    )

    schema_path = _write_schema(
        tmp_path,
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=True,
    )

    with pytest.raises(
        MetadataValidationError
    ):
        manager.enrich_document(
            source,
        )


def test_non_strict_mode_allows_incomplete_metadata(
    tmp_path: Path,
) -> None:
    """
    Non-strict mode should warn rather than stop ingestion.
    """

    source = _create_source_document(
        tmp_path,
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": []
        },
    )

    schema_path = _write_schema(
        tmp_path,
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=False,
    )

    enriched = manager.enrich_document(
        source,
    )

    assert enriched.file_name == (
        source.file_name
    )

    assert enriched.status == "unknown"

    assert enriched.access_level == (
        "unclassified"
    )


def test_missing_registry_raises_in_strict_mode(
    tmp_path: Path,
) -> None:
    """
    Strict mode requires document_registry.json.
    """

    schema_path = _write_schema(
        tmp_path,
    )

    with pytest.raises(
        MetadataRegistryError
    ):
        MetadataManager(
            registry_path=(
                tmp_path
                / "missing_registry.json"
            ),
            schema_path=schema_path,
            strict=True,
        )


def test_duplicate_file_name_registry_entries_raise_error(
    tmp_path: Path,
) -> None:
    """
    Duplicate file_name entries create ambiguous metadata
    resolution and must be rejected.
    """

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": [
                {
                    "document_id":
                        "ON-001",
                    "file_name":
                        "Same.docx",
                },
                {
                    "document_id":
                        "ON-002",
                    "file_name":
                        "Same.docx",
                },
            ]
        },
    )

    schema_path = _write_schema(
        tmp_path,
        required_fields=[
            "document_id",
            "file_name",
        ],
    )

    with pytest.raises(
        DuplicateDocumentMetadataError
    ):
        MetadataManager(
            registry_path=registry_path,
            schema_path=schema_path,
            strict=True,
        )


def test_document_type_is_inferred_from_filename(
    tmp_path: Path,
) -> None:
    """
    Structural document type may be inferred from a known
    filename pattern.
    """

    source = _create_source_document(
        tmp_path,
        file_name=(
            "Shipping_Delivery_Policy.txt"
        ),
        domain="policies",
        content="""
Document ID: ON-POL-0001
Version: 2.0
Status: Approved
Access Level: Public
Audience: Customer, Employee
Language: en
""",
    )

    registry_path = _write_registry(
        tmp_path,
        {
            "documents": []
        },
    )

    schema_path = _write_schema(
        tmp_path,
    )

    manager = MetadataManager(
        registry_path=registry_path,
        schema_path=schema_path,
        strict=True,
    )

    enriched = manager.enrich_document(
        source,
    )

    assert enriched.document_type == (
        "policy"
    )