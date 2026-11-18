"""Build a clearly-labelled daily planning path from validated forecast anchors.

The deployed system produces direct model outputs only at 1, 7 and 30 days.
The values on the other days in this module are *linear interpolation* between
those anchors. They are useful for planning, but they are not independent
daily model forecasts and must never be presented as such.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# day, price column, lower-bound column, upper-bound column
ANCHOR_SPECS: tuple[tuple[int, str, str, str], ...] = (
    (0, "modal_price", "modal_price", "modal_price"),
    (1, "predicted_price_1d", "lower_bound_1d", "upper_bound_1d"),
    (7, "predicted_price_7d", "lower_bound_7d", "upper_bound_7d"),
    (30, "predicted_price_30d", "lower_bound_30d", "upper_bound_30d"),
)

_PATH_COLUMNS = ["day", "date", "price", "low", "high", "kind"]


def _number(source: Any, key: str) -> float:
    """Read a finite float from a Series/dict, returning NaN when unavailable."""
    try:
        value = float(source.get(key))
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def build_forecast_path(
    latest: Any,
    max_days: int = 30,
    anchor_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Return day-by-day price anchors and interpolated planning values.

    Parameters
    ----------
    latest:
        A row (``pandas.Series`` or mapping) containing ``date``,
        ``modal_price``, and the deployed prediction/bound columns.
    max_days:
        Maximum day offset to return. Capped at the furthest available anchor
        so the function never extrapolates beyond a validated horizon.
    anchor_date:
        Optional reference date to anchor Day 0 to (e.g. today's live calendar date).
        If omitted, defaults to the historical date in ``latest['date']``.

    Returns
    -------
    pandas.DataFrame
        Columns ``day``, ``date``, ``price``, ``low``, ``high`` and ``kind``.
        ``kind`` is ``actual`` for today, ``validated`` for a direct model
        output, and ``interpolated`` for every other day. Empty when fewer than
        two anchors are available.
    """
    if latest is None:
        return pd.DataFrame(columns=_PATH_COLUMNS)

    norm_anchor = None
    if anchor_date is not None:
        try:
            norm_anchor = pd.to_datetime(anchor_date).normalize()
        except (TypeError, ValueError):
            norm_anchor = None

    if norm_anchor is None:
        raw_date = latest.get("date") if hasattr(latest, "get") else None
        if raw_date is None or pd.isna(raw_date):
            return pd.DataFrame(columns=_PATH_COLUMNS)

        try:
            norm_anchor = pd.to_datetime(raw_date).normalize()
        except (TypeError, ValueError):
            return pd.DataFrame(columns=_PATH_COLUMNS)

    anchors: dict[int, tuple[float, float, float]] = {}
    for day, price_key, low_key, high_key in ANCHOR_SPECS:
        price = _number(latest, price_key)
        if not np.isfinite(price):
            continue
        low = _number(latest, low_key)
        high = _number(latest, high_key)
        if not np.isfinite(low):
            low = price
        if not np.isfinite(high):
            high = price
        low, high = min(low, high), max(low, high)
        anchors[day] = (price, low, high)

    if 0 not in anchors or len(anchors) < 2:
        return pd.DataFrame(columns=_PATH_COLUMNS)

    anchor_days = sorted(anchors)
    last_day = min(max(int(max_days), 0), anchor_days[-1])
    if last_day < 1:
        return pd.DataFrame(columns=_PATH_COLUMNS)

    anchor_prices = [anchors[day][0] for day in anchor_days]
    anchor_lows = [anchors[day][1] for day in anchor_days]
    anchor_highs = [anchors[day][2] for day in anchor_days]

    rows: list[dict[str, Any]] = []
    for day in range(last_day + 1):
        if day in anchors:
            price, low, high = anchors[day]
            kind = "actual" if day == 0 else "validated"
        else:
            price = float(np.interp(day, anchor_days, anchor_prices))
            low = float(np.interp(day, anchor_days, anchor_lows))
            high = float(np.interp(day, anchor_days, anchor_highs))
            kind = "interpolated"

        rows.append(
            {
                "day": day,
                "date": norm_anchor + pd.Timedelta(days=day),
                "price": price,
                "low": low,
                "high": high,
                "kind": kind,
            }
        )

    return pd.DataFrame(rows, columns=_PATH_COLUMNS)
