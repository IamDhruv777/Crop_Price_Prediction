"""Unit tests for the dashboard's local demo authentication."""
import json

import pytest

from dashboard import auth


@pytest.fixture()
def user_store(tmp_path, monkeypatch):
    store = tmp_path / "users.json"
    monkeypatch.setattr(auth, "USERS_FILE", store)
    return store


def test_demo_accounts_are_seeded(user_store):
    store = auth.load_users()
    assert "9999999999" in store["users"]
    assert "8888888888" in store["users"]
    assert store["users"]["9999999999"]["is_demo"] is True


def test_password_is_hashed_not_plaintext(user_store):
    auth.register_user("Test Farmer", "9876543210", "secret123", "Farmer", "Maharashtra", "Nashik")
    raw = user_store.read_text(encoding="utf-8")
    assert "secret123" not in raw
    record = json.loads(raw)["users"]["9876543210"]
    assert "hash" in record and "salt" in record
    assert record["hash"] != "secret123"


def test_register_then_authenticate(user_store):
    created = auth.register_user("Test Farmer", "9876543210", "secret123", "Farmer", "Maharashtra", "Nashik")
    assert created["phone"] == "9876543210"
    assert "hash" not in created and "salt" not in created

    logged_in = auth.authenticate("9876543210", "secret123")
    assert logged_in is not None
    assert logged_in["name"] == "Test Farmer"
    assert logged_in["role"] == "Farmer"


def test_wrong_password_rejected(user_store):
    auth.register_user("Test Farmer", "9876543210", "secret123", "Farmer")
    assert auth.authenticate("9876543210", "wrong-password") is None


def test_duplicate_phone_rejected(user_store):
    auth.register_user("First", "9876543210", "secret123", "Farmer")
    with pytest.raises(ValueError, match="already registered"):
        auth.register_user("Second", "9876543210", "another123", "Shopkeeper / Vendor")


def test_invalid_phone_rejected(user_store):
    with pytest.raises(ValueError, match="10-digit"):
        auth.register_user("Test", "12345", "secret123", "Farmer")


def test_short_password_rejected(user_store):
    with pytest.raises(ValueError, match="at least 6"):
        auth.register_user("Test", "9876543210", "123", "Farmer")


def test_invalid_role_rejected(user_store):
    with pytest.raises(ValueError, match="valid role"):
        auth.register_user("Test", "9876543210", "secret123", "Astronaut")


def test_indian_country_code_normalised(user_store):
    user = auth.register_user("Test", "919876543210", "secret123", "Farmer")
    assert user["phone"] == "9876543210"
    assert auth.authenticate("9876543210", "secret123") is not None
