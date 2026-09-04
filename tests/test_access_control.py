"""
================================================================================
Project      : OnlyNativeIQ
Organization : OnlyNative
Module       : Automated Tests
File         : test_access_control.py
Author       : Abhay Kumar Pandey
Created On   : 03-Sep-2026
Description  :
    Tests OnlyNativeIQ role-based retrieval authorization.
================================================================================
"""

from src.schemas.retrieval import (
    RetrievalResult,
)
from src.security.access_control import (
    AccessController,
    UserContext,
    UserRole,
)
from src.security.document_filter import (
    DocumentFilter,
)


def _result(
    access_level: str,
    audience: list[str],
    chunk_id: str = "C1",
) -> RetrievalResult:

    return RetrievalResult(
        chunk_id=chunk_id,
        content="Test content",
        score=1.0,
        rank=1,
        access_level=access_level,
        audience=audience,
    )


def test_customer_can_access_public_customer_content() -> None:
    controller = AccessController()

    result = _result(
        "public",
        [
            "customer",
            "employee",
            "production_employee",
            "manager",
        ],
    )

    assert controller.is_authorized(
        result,
        UserContext(
            UserRole.CUSTOMER
        ),
    )


def test_customer_cannot_access_confidential_formulation() -> None:
    controller = AccessController()

    result = _result(
        "confidential",
        [
            "production_employee",
            "manager",
        ],
    )

    assert not controller.is_authorized(
        result,
        UserContext(
            UserRole.CUSTOMER
        ),
    )


def test_employee_cannot_access_confidential_formulation() -> None:
    controller = AccessController()

    result = _result(
        "confidential",
        [
            "production_employee",
            "manager",
        ],
    )

    assert not controller.is_authorized(
        result,
        UserContext(
            UserRole.EMPLOYEE
        ),
    )


def test_production_employee_can_access_confidential_formulation() -> None:
    controller = AccessController()

    result = _result(
        "confidential",
        [
            "production_employee",
            "manager",
        ],
    )

    assert controller.is_authorized(
        result,
        UserContext(
            UserRole.PRODUCTION_EMPLOYEE
        ),
    )


def test_manager_can_access_confidential_formulation() -> None:
    controller = AccessController()

    result = _result(
        "confidential",
        [
            "production_employee",
            "manager",
        ],
    )

    assert controller.is_authorized(
        result,
        UserContext(
            UserRole.MANAGER
        ),
    )


def test_internal_content_is_not_customer_visible() -> None:
    controller = AccessController()

    result = _result(
        "internal",
        [
            "employee",
            "production_employee",
            "manager",
        ],
    )

    assert not controller.is_authorized(
        result,
        UserContext(
            UserRole.CUSTOMER
        ),
    )


def test_unclassified_content_is_denied_by_default() -> None:
    controller = AccessController(
        deny_unclassified_documents=True
    )

    result = _result(
        "unclassified",
        [],
    )

    assert not controller.is_authorized(
        result,
        UserContext(
            UserRole.MANAGER
        ),
    )


def test_filter_preserves_only_authorized_results() -> None:
    document_filter = DocumentFilter(
        AccessController()
    )

    results = [
        _result(
            "confidential",
            [
                "production_employee",
                "manager",
            ],
            "CONFIDENTIAL",
        ),
        _result(
            "public",
            [
                "customer",
                "employee",
                "production_employee",
                "manager",
            ],
            "PUBLIC",
        ),
    ]

    output = (
        document_filter.filter_results(
            results=results,
            user=UserContext(
                UserRole.CUSTOMER
            ),
        )
    )

    assert [
        item.chunk_id
        for item
        in output.authorized_results
    ] == [
        "PUBLIC"
    ]

    assert output.filtered_count == 1
