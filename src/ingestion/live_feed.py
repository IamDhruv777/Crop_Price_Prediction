"""Live Daily Agmarknet Feed Service.

Fetches and caches today's official APMC mandi transactions directly from the
data.gov.in AGMARKNET API using the project's configured DATA_GOV_API_KEY.
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
CACHE_FILE = REPO_ROOT / "data" / "raw" / "mandi" / "live_today_feed.csv"


def load_env_key() -> str | None:
    """Load DATA_GOV_API_KEY from environment or .env file."""
    key = os.environ.get("DATA_GOV_API_KEY")
    if key and key.strip():
        return key.strip()

    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DATA_GOV_API_KEY=") and not line.startswith("#"):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        os.environ["DATA_GOV_API_KEY"] = val
                        return val
    return None


def fetch_live_mandi_feed(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch today's live mandi prices across crops from data.gov.in.

    Caches the results to data/raw/mandi/live_today_feed.csv for fast local loading.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Check cache first
    if not force_refresh and CACHE_FILE.exists():
        try:
            mtime = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime).strftime("%Y-%m-%d")
            if mtime == today_str and CACHE_FILE.stat().st_size > 1000:
                df = pd.read_csv(CACHE_FILE)
                if not df.empty:
                    return df
        except Exception as e:
            logger.warning(f"Failed to read live cache: {e}")

    api_key = load_env_key()
    if not api_key:
        logger.warning("No DATA_GOV_API_KEY found. Using local cache/fallback.")
        return pd.read_csv(CACHE_FILE) if CACHE_FILE.exists() else pd.DataFrame()

    from src.ingestion.mandi import MandiIngestion

    ing = MandiIngestion({"api_key": api_key})
    ing.api_key = api_key

    all_dfs = []
    crops_to_fetch = [
        "Wheat", "Soyabean", "Lentil (Masur)", "Rice", "Maize",
        "Green Gram (Moong)", "Black Gram (Urd Beans)", "Arhar (Tur-Red Gram)",
        "Kabuli Chana (Chickpeas-White)", "White Peas", "Cowpea (Lobia-Karamani)"
    ]

    for crop in crops_to_fetch:
        try:
            df = ing.fetch_data(state="", commodity=crop, start_date="", end_date="")
            if not df.empty:
                df["commodity"] = crop
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error fetching live {crop}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Clean price fields
        for col in ["modal_price", "min_price", "max_price"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")

        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(CACHE_FILE, index=False)
        logger.info(f"Successfully cached {len(combined)} live records to {CACHE_FILE}")
        return combined

    return pd.read_csv(CACHE_FILE) if CACHE_FILE.exists() else pd.DataFrame()


def get_live_mandi_price(crop: str, market: str, state: str = "") -> dict | None:
    """Lookup today's actual live price for a given crop and market."""
    df = fetch_live_mandi_feed(force_refresh=False)
    if df.empty:
        return None

    # Filter crop
    c_mask = df["commodity"].str.lower() == crop.lower()
    crop_df = df[c_mask]
    if crop_df.empty:
        return None

    # Exact market match
    m_mask = crop_df["market"].str.lower().str.contains(market.lower().replace("apmc", "").strip(), regex=False)
    sub = crop_df[m_mask]

    if sub.empty and state:
        # Match by state
        st_mask = crop_df["state"].str.lower() == state.lower()
        sub = crop_df[st_mask]

    if not sub.empty:
        latest = sub.iloc[-1]
        return {
            "modal_price": float(latest.get("modal_price", 0.0)),
            "min_price": float(latest.get("min_price", 0.0)),
            "max_price": float(latest.get("max_price", 0.0)),
            "market": str(latest.get("market")),
            "state": str(latest.get("state")),
            "arrival_date": str(latest.get("arrival_date")),
            "is_live_api": True
        }
    return None
