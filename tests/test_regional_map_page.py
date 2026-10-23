"""Regression test for the Regional Map analytics page.

The page previously crashed with ``NameError: get_mondrian_default_delta``
because the helper was referenced but never defined/imported.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parent.parent / "dashboard" / "pages" / "4_Regional_Map.py"


def test_regional_map_page_renders_for_researcher():
    at = AppTest.from_file(str(PAGE), default_timeout=240)
    at.session_state["logged_in"] = True
    at.session_state["user_role"] = "Researcher"
    at.session_state["lang"] = "en"
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.metric) >= 4
    # A real map figure must be produced, not an empty placeholder.
    assert len(at.get("plotly_chart")) >= 1
