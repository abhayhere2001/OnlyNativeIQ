"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Document Ingestion
File         : metadata_manager.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Loads corpus-governance metadata and enriches SourceDocument objects before
    parsing, chunking and indexing.

Responsibilities:
    - Load document_registry.json and metadata_schema.json.
    - Match registry entries to loaded source documents.
    - Enrich documents with document_id, version, status, access level,
      audience, language and document type.
    - Extract metadata from source-document headers when the registry is
      incomplete.
    - Validate required metadata before downstream ingestion.

Important:
    - Knowledge documents are PDF, DOCX and TXT only.
    - JSON/JSONL files under data/metadata and data/evaluation are engineering
      support artifacts and are not source documents.
    - Documents that contain both public and confidential sections will need
      chunk-level access classification later in the pipeline.
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from src.schemas.document import SourceDocument


logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("data/metadata/document_registry.json")
DEFAULT_SCHEMA_PATH = Path("data/metadata/metadata_schema.json")

DEFAULT_LANGUAGE = "en"
DEFAULT_STATUS = "unknown"
DEFAULT_ACCESS_LEVEL = "unclassified"

FALLBACK_REQUIRED_DOCUMENT_FIELDS = {
    "document_id",
    "file_name",
    "document_type",
    "domain",
    "version",
    "status",
    "access_level",
    "audience",
    "language",
}


class MetadataError(RuntimeError):
    """Base exception for metadata-management failures."""


class MetadataRegistryError(MetadataError):
    """Raised when the document registry is invalid or cannot be loaded."""


class MetadataValidationError(MetadataError):
    """Raised when required metadata is missing or invalid."""


class DuplicateDocumentMetadataError(MetadataError):
    """Raised when the registry contains ambiguous duplicate entries."""


class MetadataManager:
    """
    Enrich and validate SourceDocument objects.

    Metadata precedence:
        1. Explicit registry value
        2. Metadata embedded in the source-document header
        3. Existing SourceDocument value
        4. Conservative fallback/default
    """

    def __init__(
        self,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        strict: bool = True,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.schema_path = Path(schema_path)
        self.strict = strict

        self._registry_entries = self._load_registry()
        self._required_document_fields = self._load_required_document_fields()

        self._registry_by_file_name: dict[str, dict[str, Any]] = {}
        self._registry_by_relative_path: dict[str, dict[str, Any]] = {}
        self._build_registry_indexes()

    def enrich_document(self, document: SourceDocument) -> SourceDocument:
        """Return an enriched copy of one SourceDocument."""

        enriched = deepcopy(document)
        registry_entry = self._find_registry_entry(enriched)
        header_metadata = self._extract_header_metadata(enriched.content)

        enriched.document_id = self._pick(
            registry_entry.get("document_id"),
            header_metadata.get("document_id"),
            enriched.document_id,
        )
        enriched.document_type = self._pick(
            registry_entry.get("document_type"),
            header_metadata.get("document_type"),
            enriched.document_type,
            self._infer_document_type(enriched.file_name),
        )
        enriched.version = self._pick(
            registry_entry.get("version"),
            header_metadata.get("version"),
            enriched.version,
        )
        enriched.status = self._pick(
            registry_entry.get("status"),
            header_metadata.get("status"),
            enriched.status,
            DEFAULT_STATUS,
        )
        enriched.access_level = self._pick(
            registry_entry.get("access_level"),
            header_metadata.get("access_level"),
            enriched.access_level,
            DEFAULT_ACCESS_LEVEL,
        )
        enriched.language = self._pick(
            registry_entry.get("language"),
            header_metadata.get("language"),
            enriched.language,
            DEFAULT_LANGUAGE,
        )
        enriched.audience = self._normalize_audience(
            self._pick(
                registry_entry.get("audience"),
                header_metadata.get("audience"),
                enriched.audience,
                [],
            )
        )
        enriched.domain = self._pick(
            registry_entry.get("domain"),
            enriched.domain,
            "uncategorized",
        )

        merged_metadata = dict(enriched.metadata)
        merged_metadata.update(self._extra_registry_metadata(registry_entry))
        merged_metadata["metadata_source"] = self._metadata_source_label(
            registry_entry=registry_entry,
            header_metadata=header_metadata,
        )

        relative_path = merged_metadata.get("relative_path")
        if relative_path:
            merged_metadata["relative_path"] = self._normalize_path(
                str(relative_path)
            )

        enriched.metadata = merged_metadata
        self.validate_document(enriched)
        return enriched

    def enrich_documents(
        self,
        documents: Iterable[SourceDocument],
    ) -> list[SourceDocument]:
        """Enrich and validate a collection of SourceDocument objects."""

        enriched_documents = [
            self.enrich_document(document)
            for document in documents
        ]

        logger.info(
            "Metadata enrichment completed for %d document(s).",
            len(enriched_documents),
        )
        return enriched_documents

    def validate_document(self, document: SourceDocument) -> None:
        """Validate required metadata before the document moves downstream."""

        values = {
            "document_id": document.document_id,
            "file_name": document.file_name,
            "document_type": document.document_type,
            "domain": document.domain,
            "version": document.version,
            "status": document.status,
            "access_level": document.access_level,
            "audience": document.audience,
            "language": document.language,
        }

        missing_fields = [
            field_name
            for field_name in self._required_document_fields
            if not self._has_meaningful_value(values.get(field_name))
        ]

        if not missing_fields:
            return

        message = (
            f"Document '{document.file_name}' is missing required metadata: "
            f"{', '.join(sorted(missing_fields))}"
        )

        if self.strict:
            raise MetadataValidationError(message)

        logger.warning(message)

    def get_registry_entry(
        self,
        document: SourceDocument,
    ) -> dict[str, Any] | None:
        """Return a copy of the registry entry matched to a document."""

        entry = self._find_registry_entry(document)
        return deepcopy(entry) if entry else None

    def _load_registry(self) -> list[dict[str, Any]]:
        """Load document_registry.json and normalize supported registry shapes."""

        if not self.registry_path.exists():
            if self.strict:
                raise MetadataRegistryError(
                    f"Metadata registry does not exist: "
                    f"{self.registry_path.resolve()}"
                )
            logger.warning(
                "Metadata registry not found: %s",
                self.registry_path.resolve(),
            )
            return []

        try:
            payload = json.loads(
                self.registry_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise MetadataRegistryError(
                f"Unable to read metadata registry '{self.registry_path}': {exc}"
            ) from exc

        documents = payload.get("documents")
        if not isinstance(documents, list):
            raise MetadataRegistryError(
                "document_registry.json must contain a top-level 'documents' list."
            )

        normalized_entries: list[dict[str, Any]] = []

        for raw_entry in documents:
            if not isinstance(raw_entry, dict):
                raise MetadataRegistryError(
                    "Every document registry entry must be a JSON object."
                )

            entry = dict(raw_entry)

            # Support the older Corpus v2.0 registry shape:
            # {"file": "01_raw/products/X.docx", "format": "DOCX", ...}
            if "file" in entry and "file_name" not in entry:
                path_value = self._normalize_path(str(entry["file"]))
                entry["file_name"] = Path(path_value).name
                entry.setdefault(
                    "relative_path",
                    self._strip_legacy_raw_prefix(path_value),
                )

            normalized_entries.append(entry)

        return normalized_entries

    def _load_required_document_fields(self) -> set[str]:
        """Load required metadata fields from metadata_schema.json."""

        if not self.schema_path.exists():
            logger.warning(
                "Metadata schema not found. Using built-in required fields: %s",
                self.schema_path.resolve(),
            )
            return set(FALLBACK_REQUIRED_DOCUMENT_FIELDS)

        try:
            payload = json.loads(
                self.schema_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise MetadataRegistryError(
                f"Unable to read metadata schema '{self.schema_path}': {exc}"
            ) from exc

        required = payload.get("required_document_metadata")
        if not isinstance(required, list) or not required:
            return set(FALLBACK_REQUIRED_DOCUMENT_FIELDS)

        return {
            str(field).strip()
            for field in required
            if str(field).strip()
        }

    def _build_registry_indexes(self) -> None:
        """Build fast lookup indexes and detect ambiguous duplicates."""

        for entry in self._registry_entries:
            file_name = entry.get("file_name")
            relative_path = entry.get("relative_path")

            if file_name:
                key = str(file_name).strip().lower()
                if key in self._registry_by_file_name:
                    raise DuplicateDocumentMetadataError(
                        f"Duplicate file_name in metadata registry: {file_name}"
                    )
                self._registry_by_file_name[key] = entry

            if relative_path:
                key = self._normalize_path(str(relative_path)).lower()
                if key in self._registry_by_relative_path:
                    raise DuplicateDocumentMetadataError(
                        f"Duplicate relative_path in metadata registry: {relative_path}"
                    )
                self._registry_by_relative_path[key] = entry

    def _find_registry_entry(
        self,
        document: SourceDocument,
    ) -> dict[str, Any]:
        """Match by relative path first, then by filename."""

        relative_path = document.metadata.get("relative_path")

        if relative_path:
            key = self._normalize_path(str(relative_path)).lower()
            entry = self._registry_by_relative_path.get(key)
            if entry:
                return entry

        return self._registry_by_file_name.get(
            document.file_name.lower(),
            {},
        )

    @staticmethod
    def _extract_header_metadata(content: str) -> dict[str, Any]:
        """Extract simple metadata labels already embedded in source text."""

        if not content:
            return {}

        header = content[:5000]
        patterns = {
            "document_id": r"(?im)^\s*Document\s*ID\s*:\s*(.+?)\s*$",
            "version": r"(?im)^\s*Version\s*:\s*(.+?)\s*$",
            "status": r"(?im)^\s*Status\s*:\s*(.+?)\s*$",
            "access_level": r"(?im)^\s*Access\s*Level\s*:\s*(.+?)\s*$",
            "audience": r"(?im)^\s*Audience\s*:\s*(.+?)\s*$",
            "language": r"(?im)^\s*Language\s*:\s*(.+?)\s*$",
            "document_type": r"(?im)^\s*Document\s*Type\s*:\s*(.+?)\s*$",
        }

        extracted: dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, header)
            if match:
                extracted[key] = match.group(1).strip()

        return extracted

    @staticmethod
    def _infer_document_type(file_name: str) -> str:
        """Infer stable structural document type from the filename."""

        name = Path(file_name).stem.lower().replace("-", "_")

        if "product_guide" in name:
            return "product_guide"
        if "employee_handbook" in name:
            return "employee_handbook"
        if "code_of_conduct" in name:
            return "code_of_conduct"
        if "company_faq" in name:
            return "company_faq"
        if "brand_story" in name:
            return "brand_story"
        if "_sop" in name:
            return "sop"
        if "_policy" in name:
            return "policy"
        if "faq" in name:
            return "faq"

        return "knowledge_document"

    @staticmethod
    def _normalize_audience(value: Any) -> list[str]:
        """Normalize audience metadata to a list of lowercase role names."""

        if value is None:
            return []
        if isinstance(value, list):
            items = value
        elif isinstance(value, tuple):
            items = list(value)
        elif isinstance(value, str):
            items = re.split(r"[,;]", value)
        else:
            items = [str(value)]

        normalized: list[str] = []
        for item in items:
            text = str(item).strip().lower().replace(" ", "_")
            if text and text not in normalized:
                normalized.append(text)

        return normalized

    @staticmethod
    def _extra_registry_metadata(
        registry_entry: dict[str, Any],
    ) -> dict[str, Any]:
        """Return registry values not already represented by core fields."""

        core_fields = {
            "document_id",
            "file_name",
            "file",
            "document_type",
            "domain",
            "version",
            "status",
            "access_level",
            "audience",
            "language",
            "relative_path",
        }

        return {
            key: value
            for key, value in registry_entry.items()
            if key not in core_fields
        }

    @staticmethod
    def _metadata_source_label(
        registry_entry: dict[str, Any],
        header_metadata: dict[str, Any],
    ) -> str:
        if registry_entry and header_metadata:
            return "registry+document_header"
        if registry_entry:
            return "registry"
        if header_metadata:
            return "document_header"
        return "source_document/defaults"

    @staticmethod
    def _pick(*values: Any) -> Any:
        """Return the first meaningful value from candidate values."""

        for value in values:
            if MetadataManager._has_meaningful_value(value):
                return value
        return None

    @staticmethod
    def _has_meaningful_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return True

    @staticmethod
    def _normalize_path(value: str) -> str:
        return value.replace("\\", "/").strip("/")

    @staticmethod
    def _strip_legacy_raw_prefix(value: str) -> str:
        """Map legacy 01_raw paths to the new knowledge_base-relative layout."""

        normalized = MetadataManager._normalize_path(value)

        for prefix in ("01_raw/", "knowledge_base/"):
            if normalized.lower().startswith(prefix.lower()):
                return normalized[len(prefix):]

        return normalized
