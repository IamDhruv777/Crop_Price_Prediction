"""
Lightweight demo authentication for the Krishi Bhav Saathi dashboard.

This is intentionally local-only: accounts live in a JSON file with salted
PBKDF2 password hashes. It is suitable for a demo deployment, not a
multi-tenant production service. Set ``KRISHI_USERS_FILE`` to relocate the
store, or delete it to reset the demo accounts.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
USERS_FILE = Path(os.environ.get("KRISHI_USERS_FILE", REPO_ROOT / "data" / ".app_users.json"))

ROLES = [
    "Farmer",
    "Shopkeeper / Vendor",
    "Trader / Commission Agent",
    "Researcher / Officer",
]

STATE_NAMES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Chandigarh",
]

PBKDF2_ITERATIONS = 260_000


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt_hex: Optional[str] = None) -> Dict[str, str]:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return {
        "salt": salt.hex(),
        "hash": digest.hex(),
        "iterations": str(PBKDF2_ITERATIONS),
    }


def verify_password(password: str, record: Dict[str, Any]) -> bool:
    try:
        salt = bytes.fromhex(record["salt"])
        iterations = int(record.get("iterations", PBKDF2_ITERATIONS))
        expected = record["hash"]
    except (KeyError, TypeError, ValueError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest.hex(), expected)


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------
def _empty_store() -> Dict[str, Any]:
    return {
        "version": 1,
        "users": {
            # Demo accounts: clearly marked, password shown on the login screen.
            "9999999999": {
                "name": "Ramesh Patil",
                "phone": "9999999999",
                "role": "Farmer",
                "state": "Maharashtra",
                "district": "Nashik",
                "language": "hi",
                "created_at": "2026-01-01T00:00:00+00:00",
                "is_demo": True,
                **_hash_password("kisan1234"),
            },
            "8888888888": {
                "name": "Sunita Devi",
                "phone": "8888888888",
                "role": "Shopkeeper / Vendor",
                "state": "Uttar Pradesh",
                "district": "Kanpur Nagar",
                "language": "hi",
                "created_at": "2026-01-01T00:00:00+00:00",
                "is_demo": True,
                **_hash_password("vyapari1234"),
            },
        },
    }


def load_users() -> Dict[str, Any]:
    if not USERS_FILE.exists():
        store = _empty_store()
        save_users(store)
        return store
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as handle:
            store = json.load(handle)
        if not isinstance(store.get("users"), dict):
            raise ValueError("Invalid user store")
        return store
    except Exception:
        # A corrupt local store must never break the dashboard; reset it.
        store = _empty_store()
        save_users(store)
        return store


def save_users(store: Dict[str, Any]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(USERS_FILE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(store, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, USERS_FILE)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _normalise_phone(phone: str) -> str:
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) != 10:
        raise ValueError("Please enter a valid 10-digit mobile number.")
    return digits


def register_user(
    name: str,
    phone: str,
    password: str,
    role: str,
    state: str = "",
    district: str = "",
    language: str = "hi",
) -> Dict[str, Any]:
    name = (name or "").strip()
    if len(name) < 2:
        raise ValueError("Please enter your name.")
    phone = _normalise_phone(phone)
    if role not in ROLES:
        raise ValueError("Please choose a valid role.")
    if password != password.strip():
        raise ValueError("Password cannot start or end with a space.")
    hashed = _hash_password(password)

    store = load_users()
    if phone in store["users"]:
        raise ValueError("This mobile number is already registered. Please log in instead.")

    user = {
        "name": name,
        "phone": phone,
        "role": role,
        "state": (state or "").strip(),
        "district": (district or "").strip(),
        "language": language if language in {"hi", "en"} else "hi",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_demo": False,
        **hashed,
    }
    store["users"][phone] = user
    save_users(store)
    return public_user(user)


def authenticate(phone: str, password: str) -> Optional[Dict[str, Any]]:
    try:
        phone = _normalise_phone(phone)
    except ValueError:
        return None
    user = load_users()["users"].get(phone)
    if not user or not verify_password(password, user):
        return None
    return public_user(user)


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in user.items() if key not in {"hash", "salt", "iterations"}}


# ---------------------------------------------------------------------------
# Streamlit session helpers
# ---------------------------------------------------------------------------
def init_auth_state() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("auth_view", "landing")
    st.session_state.setdefault("pending_role", "Farmer")


def login_session(user: Dict[str, Any]) -> None:
    st.session_state["logged_in"] = True
    st.session_state["user"] = user
    st.session_state["auth_view"] = "dashboard"
    st.session_state["lang"] = user.get("language", "hi")


def logout_session() -> None:
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
    st.session_state["auth_view"] = "landing"


def current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("user") if st.session_state.get("logged_in") else None


def require_login() -> Dict[str, Any]:
    """Guard used by the multipage dashboard scripts."""
    user = current_user()
    if user:
        return user
    st.warning("🔐 Please sign in from the home page to open this module.")
    st.markdown("[← Return to the home page](/)")
    st.stop()
