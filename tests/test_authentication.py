"""
Unit tests for OnlyNativeIQ local authentication.
"""

from pathlib import Path

import yaml

from src.security.authentication import (
    LocalAuthenticationProvider,
    PasswordHasher,
)


def test_password_hash_round_trip():
    record = PasswordHasher.hash_password(
        "A-Strong-Demo-Password!"
    )

    assert PasswordHasher.verify_password(
        "A-Strong-Demo-Password!",
        record,
    )

    assert not PasswordHasher.verify_password(
        "wrong-password",
        record,
    )


def test_valid_user_authenticates(
    tmp_path: Path,
):
    password = (
        "Another-Strong-Password!"
    )

    hashed = (
        PasswordHasher.hash_password(
            password
        )
    )

    path = (
        tmp_path
        / "users.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "users": {
                    "alice": {
                        "display_name":
                            "Alice",

                        "role":
                            "employee",

                        "enabled":
                            True,

                        **hashed,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    provider = (
        LocalAuthenticationProvider(
            path
        )
    )

    user = provider.authenticate(
        "alice",
        password,
    )

    assert user is not None

    assert (
        user.username
        == "alice"
    )

    assert (
        user.role.value
        == "employee"
    )


def test_wrong_password_is_rejected(
    tmp_path: Path,
):
    hashed = (
        PasswordHasher.hash_password(
            "Correct-Password!"
        )
    )

    path = (
        tmp_path
        / "users.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "users": {
                    "alice": {
                        "role":
                            "manager",

                        "enabled":
                            True,

                        **hashed,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    provider = (
        LocalAuthenticationProvider(
            path
        )
    )

    assert (
        provider.authenticate(
            "alice",
            "Wrong-Password!",
        )
        is None
    )


def test_disabled_user_is_rejected(
    tmp_path: Path,
):
    hashed = (
        PasswordHasher.hash_password(
            "Correct-Password!"
        )
    )

    path = (
        tmp_path
        / "users.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            {
                "users": {
                    "alice": {
                        "role":
                            "manager",

                        "enabled":
                            False,

                        **hashed,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    provider = (
        LocalAuthenticationProvider(
            path
        )
    )

    assert (
        provider.authenticate(
            "alice",
            "Correct-Password!",
        )
        is None
    )
