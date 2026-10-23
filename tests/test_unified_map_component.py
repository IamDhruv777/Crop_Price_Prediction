"""Regression tests for the unified zoom map.

The map previously turned every district grey at street level, and hovering a
district painted the whole viewport one solid colour. These tests keep the
high-zoom experience from regressing.
"""
import pandas as pd

from dashboard.components.unified_zoom_map import build_unified_map_html


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state": "Punjab",
                "district": "Ludhiana",
                "market_name": "Ludhiana",
                "modal_price": 1800.0,
                "predicted_price_7d": 1830.0,
                "latitude": 30.901,
                "longitude": 75.857,
            },
            {
                "state": "Punjab",
                "district": "Ludhiana",
                "market_name": "Khanna",
                "modal_price": 1750.0,
                "predicted_price_7d": 1725.0,
                "latitude": 30.704,
                "longitude": 76.221,
            },
        ]
    )


def test_street_level_uses_light_tint_and_mandi_labels():
    html = build_unified_map_html("Wheat", _sample_frame(), is_hi=False)

    assert "getStreetLevelDistrictStyle" in html
    assert "fillOpacity: noData ? 0.12 : 0.28" in html
    assert "mandi-price-badge" in html
    assert "refreshMandiLabels" in html
    # Hover must not turn a viewport-filling district into a solid colour block.
    assert "fillOpacity: zoomedIn ? 0.38 : 0.92" in html


def test_no_data_message_for_empty_frame():
    html = build_unified_map_html("Wheat", pd.DataFrame(), is_hi=False)
    assert "No mandi data available" in html
