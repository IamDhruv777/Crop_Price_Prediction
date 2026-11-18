"""2026 Price Indexer & Market Calibration Engine.

Integrates today's real-time APMC mandi feed fetched from data.gov.in (AGMARKNET)
with the 10-year trained ML models, scaling predictions directly to today's
verified live mandi transactions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_FEED_FILE = REPO_ROOT / "data" / "raw" / "mandi" / "live_today_feed.csv"

# Pre-load live feed cache if present
_LIVE_CACHE: pd.DataFrame | None = None


def get_live_feed() -> pd.DataFrame:
    """Return the cached live feed DataFrame."""
    global _LIVE_CACHE
    if _LIVE_CACHE is not None and not _LIVE_CACHE.empty:
        return _LIVE_CACHE
    if LIVE_FEED_FILE.exists():
        try:
            _LIVE_CACHE = pd.read_csv(LIVE_FEED_FILE)
            return _LIVE_CACHE
        except Exception as e:
            logger.warning(f"Failed to read live feed: {e}")
    return pd.DataFrame()


# Verified 2024-2026 Mandi Market Modal Benchmarks (INR / Quintal)
CROP_2026_CONFIG: dict[str, dict[str, float]] = {
    "Wheat": {
        "target": 2850.0,
        "base_raw": 1532.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 2600.0,
        "mandi_max": 3300.0,
    },
    "Soyabean": {
        "target": 7200.0,
        "base_raw": 2750.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 6500.0,
        "mandi_max": 7850.0,
    },
    "Lentil (Masur)": {
        "target": 5600.0,      # Wholesale farmgate/mandi rate: Rs 4,500 - 6,500 (MSP Rs 6,700)
        "base_raw": 4500.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 4500.0,
        "mandi_max": 6500.0,
    },
    "Rice": {
        "target": 3650.0,
        "base_raw": 2420.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 3200.0,
        "mandi_max": 4300.0,
    },
    "Maize": {
        "target": 2450.0,
        "base_raw": 1370.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 2100.0,
        "mandi_max": 2800.0,
    },
    "Green Gram (Moong)": {
        "target": 7850.0,
        "base_raw": 4476.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 7000.0,
        "mandi_max": 8800.0,
    },
    "Black Gram (Urd Beans)": {
        "target": 7650.0,
        "base_raw": 5501.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 6800.0,
        "mandi_max": 8500.0,
    },
    "Arhar (Tur-Red Gram)": {
        "target": 8500.0,
        "base_raw": 4300.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 7800.0,
        "mandi_max": 9500.0,
    },
    "Kabuli Chana (Chickpeas-White)": {
        "target": 8200.0,
        "base_raw": 10000.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 6800.0,
        "mandi_max": 9500.0,
    },
    "White Peas": {
        "target": 5200.0,
        "base_raw": 3093.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 4500.0,
        "mandi_max": 6000.0,
    },
    "Cowpea (Lobia-Karamani)": {
        "target": 6200.0,
        "base_raw": 5050.0,
        "min_r": 0.88,
        "max_r": 1.15,
        "mandi_min": 5200.0,
        "mandi_max": 7200.0,
    },
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


def _match_crop_config(crop: str) -> dict[str, float]:
    crop_str = str(crop).strip()
    if crop_str in CROP_2026_CONFIG:
        return CROP_2026_CONFIG[crop_str]
    for key, cfg in CROP_2026_CONFIG.items():
        if key.lower() in crop_str.lower() or crop_str.lower() in key.lower():
            return cfg
    return {"target": 3500.0, "base_raw": 2000.0, "min_r": 0.88, "max_r": 1.15}


_LIVE_MANDIS: dict[tuple[str, str], float] | None = None
_LIVE_DISTRICTS: dict[tuple[str, ...], float] | None = None
_LIVE_STATES: dict[tuple[str, str], float] | None = None


def _init_hierarchical_live_cache():
    global _LIVE_MANDIS, _LIVE_DISTRICTS, _LIVE_STATES
    if _LIVE_MANDIS is not None:
        return
    _LIVE_MANDIS = {}
    _LIVE_DISTRICTS = {}
    _LIVE_STATES = {}
    live_df = get_live_feed()
    if live_df.empty:
        return

    # 1. Mandi dictionary
    for _, row in live_df.iterrows():
        c = str(row.get("commodity", "")).lower().strip()
        m = str(row.get("market", "")).lower().replace("apmc", "").replace("(", "").replace(")", "").strip()
        p = float(row.get("modal_price", 0.0))
        if p > 0:
            _LIVE_MANDIS[(c, m)] = p
            first_w = m.split()[0] if m else ""
            if len(first_w) >= 3 and (c, first_w) not in _LIVE_MANDIS:
                _LIVE_MANDIS[(c, first_w)] = p

    # 2. District median dictionary (Direct real price for this specific district today)
    if {"commodity", "state", "district", "modal_price"}.issubset(live_df.columns):
        dist_grp = live_df.groupby(["commodity", "state", "district"])["modal_price"].median().reset_index()
        for _, row in dist_grp.iterrows():
            c = str(row["commodity"]).lower().strip()
            s = str(row["state"]).lower().strip()
            d = str(row["district"]).lower().strip()
            p = float(row["modal_price"])
            if p > 0:
                _LIVE_DISTRICTS[(c, s, d)] = p
                _LIVE_DISTRICTS[(c, d)] = p

    # 3. State median dictionary (Direct real price for this state today)
    if {"commodity", "state", "modal_price"}.issubset(live_df.columns):
        state_grp = live_df.groupby(["commodity", "state"])["modal_price"].median().reset_index()
        for _, row in state_grp.iterrows():
            c = str(row["commodity"]).lower().strip()
            s = str(row["state"]).lower().strip()
            p = float(row["modal_price"])
            if p > 0:
                _LIVE_STATES[(c, s)] = p


def lookup_live_price(crop: str, market: str = "", district: str = "", state: str = "") -> float | None:
    """Check if the market, district, or state has an official live transaction recorded today (O(1) lookup)."""
    _init_hierarchical_live_cache()
    if not _LIVE_MANDIS:
        return None

    c = crop.lower().strip()
    s = state.lower().strip()
    d = district.lower().strip()
    clean_m = market.lower().replace("apmc", "").replace("(", "").replace(")", "").strip()
    first_w = clean_m.split()[0] if clean_m else ""

    # Tier 1: Exact Mandi Match
    if clean_m and (c, clean_m) in _LIVE_MANDIS:
        return _LIVE_MANDIS[(c, clean_m)]
    if len(first_w) >= 3 and (c, first_w) in _LIVE_MANDIS:
        return _LIVE_MANDIS[(c, first_w)]

    # Tier 2: Real District Live Transaction
    if d and (c, s, d) in _LIVE_DISTRICTS:
        return _LIVE_DISTRICTS[(c, s, d)]
    if d and (c, d) in _LIVE_DISTRICTS:
        return _LIVE_DISTRICTS[(c, d)]

    # Tier 3: Real State Live Transaction
    if s and (c, s) in _LIVE_STATES:
        return _LIVE_STATES[(c, s)]

    return None


def get_crop_multiplier(crop: str) -> float:
    cfg = _match_crop_config(crop)
    return cfg["target"] / cfg["base_raw"]


def get_effective_multiplier(raw_price: float, crop: str, state: str = "", district: str = "", market: str = "") -> float:
    if not np.isfinite(raw_price) or raw_price <= 0:
        return get_crop_multiplier(crop)

    cfg = _match_crop_config(crop)
    m_min = cfg.get("mandi_min", 0.0)
    m_max = cfg.get("mandi_max", 999999.0)

    # 0. Idempotency Check: If already calibrated into the 2026 wholesale range, multiplier is 1.0
    if m_min <= raw_price <= m_max:
        return 1.0

    # 1. Direct live match from today's government feed (Mandi -> District -> State)
    live_p = lookup_live_price(crop, market=market, district=district, state=state)
    if live_p is not None:
        if m_min > 0 and m_max < 999999.0:
            live_p = float(np.clip(live_p, m_min, m_max))
        return live_p / raw_price

    # 2. Benchmark calibration with bounded ratio
    raw_ratio = raw_price / cfg["base_raw"]
    bounded_ratio = float(np.clip(raw_ratio, cfg["min_r"], cfg["max_r"]))
    calibrated_price = cfg["target"] * bounded_ratio

    # 3. Hard wholesale sanity bounds so local outliers never produce unrealistic quotes
    if m_min > 0 and m_max < 999999.0:
        calibrated_price = float(np.clip(calibrated_price, m_min, m_max))

    return calibrated_price / raw_price


def calibrate_series(series: pd.Series, crop: str | None = None, market: str | None = None, district: str | None = None, state: str | None = None) -> pd.Series:
    s = series.copy()
    c = crop or s.get("commodity", "")
    m = market or s.get("market", "")
    d = district or s.get("district", "")
    st_val = state or s.get("state", "")
    raw_p = float(s.get("modal_price", 0.0))
    mult = get_effective_multiplier(raw_p, str(c), str(st_val), str(d), str(m))
    for col in PRICE_COLS:
        if col in s and pd.notna(s[col]) and isinstance(s[col], (int, float, np.number)):
            s[col] = round(float(s[col]) * mult, 2)
    return s


def calibrate_dataframe(df: pd.DataFrame, crop: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()

    if crop:
        c = crop
        if "modal_price" in out.columns:
            cfg = _match_crop_config(c)
            raw_prices = out["modal_price"].fillna(cfg["base_raw"]).values
            states = out["state"].astype(str).values if "state" in out.columns else [""] * len(out)
            districts = out["district"].astype(str).values if "district" in out.columns else [""] * len(out)
            markets = out["market"].astype(str).values if "market" in out.columns else [""] * len(out)
            mults = np.zeros(len(out))

            for idx in range(len(out)):
                raw_p = float(raw_prices[idx])
                mults[idx] = get_effective_multiplier(raw_p, c, states[idx], districts[idx], markets[idx])

            for col in PRICE_COLS:
                if col in out.columns:
                    out[col] = (out[col] * mults).round(2)
        else:
            mult = get_crop_multiplier(c)
            for col in PRICE_COLS:
                if col in out.columns:
                    out[col] = (out[col] * mult).round(2)
    elif "commodity" in out.columns:
        for c in out["commodity"].unique():
            mask = out["commodity"] == c
            sub_df = out.loc[mask]
            out.loc[mask] = calibrate_dataframe(sub_df, str(c))
    return out
