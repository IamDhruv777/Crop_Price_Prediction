"""End-to-end smoke test for landing -> sign-up -> dashboard."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard import auth

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
DASHBOARD_DIR = APP.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import auth as app_auth  # the module name imported inside dashboard/app.py


def _first(widgets, predicate):
    matches = [widget for widget in widgets if predicate(widget)]
    assert matches, "Expected widget not found"
    return matches[0]


def test_landing_signup_dashboard_flow(tmp_path, monkeypatch):
    store = tmp_path / "users.json"
    monkeypatch.setattr(auth, "USERS_FILE", store)
    monkeypatch.setattr(app_auth, "USERS_FILE", store)

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.error
    assert any("आपकी मेहनत का" in block.value for block in at.markdown)

    _first(at.button, lambda b: "मुफ़्त खाता" in b.label).click().run()
    assert not at.error

    _first(at.text_input, lambda w: w.label == "आपका नाम").set_value("Test Kisan")
    _first(at.text_input, lambda w: w.label == "मोबाइल नंबर (10 अंक)").set_value("9876501234")
    _first(at.text_input, lambda w: w.label == "पासवर्ड").set_value("secret123")
    _first(at.text_input, lambda w: w.label == "पासवर्ड दोबारा").set_value("secret123")
    _first(at.selectbox, lambda w: w.label == "आप कौन हैं?").set_value("Farmer")
    _first(at.selectbox, lambda w: w.label == "राज्य").set_value("Maharashtra")
    _first(at.text_input, lambda w: w.label == "जिला (वैकल्पिक)").set_value("Nashik")
    at.checkbox[0].check()
    _first(at.button, lambda b: "खाता बनाएं" in b.label).click().run()

    assert not at.error
    assert any("नमस्ते" in block.value for block in at.markdown)
    assert [s.label for s in at.selectbox] == ["1. फसल", "2. राज्य", "3. जिला", "4. मंडी"]
    assert len(at.metric) >= 3
    assert any("तारीख" in widget.label for widget in at.date_input)
    assert any("📅" in tab.label for tab in at.tabs)
