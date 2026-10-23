"""Tests for the honest daily planning path between validated anchors."""
import numpy as np
import pandas as pd

from src.inference.forecast_path import build_forecast_path


def _latest(**overrides) -> pd.Series:
    row = {
        "date": "2017-06-16",
        "modal_price": 1000.0,
        "predicted_price_1d": 1010.0,
        "lower_bound_1d": 900.0,
        "upper_bound_1d": 1100.0,
        "predicted_price_7d": 1060.0,
        "lower_bound_7d": 950.0,
        "upper_bound_7d": 1150.0,
        "predicted_price_30d": 1000.0,
        "lower_bound_30d": 800.0,
        "upper_bound_30d": 1200.0,
    }
    row.update(overrides)
    return pd.Series(row)


def test_path_marks_direct_anchors_and_interpolates_the_rest():
    path = build_forecast_path(_latest(), max_days=30)

    assert len(path) == 31
    assert path.loc[path["day"] == 0, "kind"].item() == "actual"
    assert path.loc[path["day"] == 1, "kind"].item() == "validated"
    assert path.loc[path["day"] == 7, "kind"].item() == "validated"
    assert path.loc[path["day"] == 30, "kind"].item() == "validated"
    assert path.loc[path["day"] == 13, "kind"].item() == "interpolated"

    # Day 4 sits three sixths of the way from the 1-day to the 7-day anchor.
    assert path.loc[path["day"] == 4, "price"].item() == 1035.0
    # Day 13 is interpolated between the 7-day and 30-day anchors.
    expected = 1060.0 + (1000.0 - 1060.0) * (6 / 23)
    assert np.isclose(path.loc[path["day"] == 13, "price"].item(), expected)

    assert path.loc[path["day"] == 13, "date"].item() == pd.Timestamp("2017-06-29")
    assert path["price"].notna().all()
    assert (path["low"] <= path["price"]).all()
    assert (path["price"] <= path["high"]).all()


def test_missing_30_day_anchor_never_extrapolates():
    path = build_forecast_path(_latest(predicted_price_30d=np.nan), max_days=30)

    assert path["day"].max() == 7
    assert "interpolated" in set(path["kind"])
    assert 30 not in set(path["day"])


def test_max_days_caps_without_padding_beyond_available_anchor():
    path = build_forecast_path(_latest(), max_days=10)

    assert path["day"].max() == 10
    assert len(path) == 11


def test_missing_date_or_too_few_anchors_returns_empty_frame():
    assert build_forecast_path(_latest(date=None)).empty
    only_today = _latest(
        predicted_price_1d=np.nan,
        predicted_price_7d=np.nan,
        predicted_price_30d=np.nan,
    )
    assert build_forecast_path(only_today).empty


def test_bounds_are_normalised_when_inputs_are_swapped():
    path = build_forecast_path(
        _latest(lower_bound_7d=1150.0, upper_bound_7d=950.0),
        max_days=7,
    )

    assert (path["low"] <= path["high"]).all()
