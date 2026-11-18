"""2026 Price Indexer & Inflation Calibration Engine.

Bridges the historical 10-year AGMARKNET dataset (2008-2017) to real-world 2026
market economics. Uses official Government of India Minimum Support Price (MSP)
hikes and verified APMC mandi inflation rates to scale base prices and prediction
bounds while preserving exact model-predicted percentage changes, directional
signals, and Conformal Prediction interval widths.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

# Multipliers from 2016-17 AGMARKNET base to 2026 Market Reality:
# - Wheat: MSP ₹1,525 -> ₹2,425 (+59%). Open market in MH (Devala/Nashik): ~₹2,850 - ₹2,950 / qtl (~1.89295x -> ₹2,900)
# - Rice: MSP ₹1,510 -> ₹2,320. Open market milled: ~₹3,800 - ₹4,200 / qtl (~1.612x)
# - Soyabean: MSP ₹2,775 -> ₹4,892 (+76%). Open market: ~₹4,600 - ₹4,900 / qtl (~1.727x)
# - Maize: MSP ₹1,365 -> ₹2,225 (+63%). Open market: ~₹2,300 - ₹2,500 / qtl (~1.737x)
# - Green Gram (Moong): MSP ₹5,225 -> ₹8,682 (+66%). Open market: ~₹8,200 - ₹8,600 / qtl (~1.669x)
# - Black Gram (Urd): MSP ₹5,000 -> ₹7,400 (+48%). Open market: ~₹7,600 - ₹8,400 / qtl (~1.436x)
# - Arhar (Tur): MSP ₹5,050 -> ₹7,550 (+49%). Open market (pulse deficit): ~₹9,800 - ₹11,000 / qtl (~2.052x)
# - Lentil (Masur): MSP ₹4,000 -> ₹6,700 (+67%). Open market: ~₹6,400 - ₹6,800 / qtl (~1.456x)
# - Kabuli Chana: Open market: ~₹13,800 - ₹15,000 / qtl (~1.420x)
# - White Peas: Open market: ~₹4,900 - ₹5,400 / qtl (~1.617x)
# - Cowpea (Lobia): Open market: ~₹7,200 - ₹7,800 / qtl (~1.465x)

CROP_2026_MULTIPLIERS: dict[str, float] = {
    "Wheat": 2900.0 / 1532.0,  # ~1.89295 -> Exact Devala Wheat = ₹2,900 / qtl
    "Rice": 3900.0 / 2420.0,
    "Soyabean": 4750.0 / 2750.0,
    "Maize": 2380.0 / 1370.0,
    "Green Gram (Moong)": 8450.0 / 5064.0,
    "Black Gram (Urd Beans)": 7900.0 / 5501.0,
    "Arhar (Tur-Red Gram)": 10500.0 / 5117.0,
    "Lentil (Masur)": 6550.0 / 4500.0,
    "Kabuli Chana (Chickpeas-White)": 14200.0 / 10000.0,
    "White Peas": 5000.0 / 3093.0,
    "Cowpea (Lobia-Karamani)": 7400.0 / 5050.0,
}

PRICE_COLS = [
    "modal_price",
    "min_price",
    "max_price",
    "predicted_price_1d",
    "predicted_price_7d",
    "predicted_price_30d",
    "lower_bound_1d",
    "upper_bound_1d",
    "uncertainty_delta_1d",
    "lower_bound_7d",
    "upper_bound_7d",
    "uncertainty_delta_7d",
    "lower_bound_30d",
    "upper_bound_30d",
    "uncertainty_delta_30d",
    "actual_price_1d",
    "actual_price_7d",
    "actual_price_30d",
    "price_rolling_mean_7",
    "price_rolling_mean_30",
]


def get_crop_multiplier(crop: str) -> float:
    """Return the 2017 -> 2026 price calibration multiplier for the crop."""
    for key, mult in CROP_2026_MULTIPLIERS.items():
        if key.lower() in crop.lower() or crop.lower() in key.lower():
            return mult
    return 1.65  # Default historical-to-modern inflation factor


def calibrate_series(series: pd.Series, crop: str | None = None) -> pd.Series:
    """Return a scaled copy of a single prediction series/row for 2026."""
    s = series.copy()
    c = crop or s.get("commodity", "")
    mult = get_crop_multiplier(str(c))
    for col in PRICE_COLS:
        if col in s and pd.notna(s[col]) and isinstance(s[col], (int, float, np.number)):
            s[col] = round(float(s[col]) * mult, 2)
    return s


def calibrate_dataframe(df: pd.DataFrame, crop: str | None = None) -> pd.DataFrame:
    """Return a scaled copy of a predictions DataFrame for 2026."""
    if df.empty:
        return df.copy()
    out = df.copy()
    if crop:
        mult = get_crop_multiplier(crop)
        for col in PRICE_COLS:
            if col in out.columns:
                out[col] = (out[col] * mult).round(2)
    elif "commodity" in out.columns:
        for c in out["commodity"].unique():
            mult = get_crop_multiplier(str(c))
            mask = out["commodity"] == c
            for col in PRICE_COLS:
                if col in out.columns:
                    out.loc[mask, col] = (out.loc[mask, col] * mult).round(2)
    return out
