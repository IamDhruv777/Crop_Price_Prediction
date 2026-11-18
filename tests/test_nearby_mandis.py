"""
Unit and regression tests for Haversine distance and Nearby Mandis ranking.
"""
import numpy as np
import pandas as pd
import pytest


def haversine_km(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lats_r, lons_r = np.radians(lats), np.radians(lons)
    dlat = lats_r - lat1_r
    dlon = lons_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lats_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return R * c


def test_haversine_exact_math():
    """Verify known distances: Equator 1 deg ~ 111.19 km, same point = 0 km."""
    zero_dist = haversine_km(20.0, 75.0, np.array([20.0]), np.array([75.0]))
    assert np.isclose(zero_dist[0], 0.0, atol=1e-5)

    one_deg_lat = haversine_km(0.0, 0.0, np.array([1.0]), np.array([0.0]))
    assert np.isclose(one_deg_lat[0], 111.19, atol=0.5)


def test_devala_nearby_mandis_distance():
    """Verify Devala (Nashik) neighboring mandis are genuinely close (<60 km)."""
    geo_df = pd.read_csv("data/raw/geography/market_master.csv")
    devala = geo_df[geo_df["market_name"] == "Devala"]
    assert not devala.empty, "Devala must be in market_master.csv"

    curr_lat = float(devala["latitude"].iloc[0])
    curr_lon = float(devala["longitude"].iloc[0])

    candidates = geo_df[geo_df["market_name"] != "Devala"].copy()
    candidates["dist_km"] = haversine_km(
        curr_lat, curr_lon, candidates["latitude"].to_numpy(), candidates["longitude"].to_numpy()
    )

    # Within 50 km
    within_50 = candidates[candidates["dist_km"] <= 50.0]
    assert len(within_50) >= 5, f"Expected at least 5 mandis within 50km, found {len(within_50)}"

    # Distant mandis like Pune (>300km) must NOT be in within_50
    pune_dist = candidates[candidates["market_name"] == "Pune"]["dist_km"].iloc[0]
    assert pune_dist > 200.0, f"Pune should be >200km away from Devala, got {pune_dist} km"

    sangli_dist = candidates[candidates["market_name"] == "Sangli"]["dist_km"].iloc[0]
    assert sangli_dist > 200.0, f"Sangli should be >200km away from Devala, got {sangli_dist} km"


def test_nearby_price_diversity():
    """Verify prices in nearby mandis are distinct and not all identical static values."""
    preds = pd.read_csv("data/deployed_test_predictions.csv")
    from src.utils.price_indexer import calibrate_dataframe
    preds = calibrate_dataframe(preds)
    geo = pd.read_csv("data/raw/geography/market_master.csv")

    usable = preds.dropna(subset=["predicted_price_7d"])
    latest = usable.sort_values("date").groupby(["state", "district", "market", "commodity"], as_index=False).last()
    wheat_latest = latest[latest["commodity"] == "Wheat"]

    merged = geo.merge(
        wheat_latest,
        left_on=["state", "district", "market_name"],
        right_on=["state", "district", "market"],
        how="inner",
    )
    devala = merged[merged["market_name"] == "Devala"]
    assert not devala.empty

    curr_lat = float(devala["latitude"].iloc[0])
    curr_lon = float(devala["longitude"].iloc[0])

    candidates = merged[merged["market_name"] != "Devala"].copy()
    candidates["dist_km"] = haversine_km(
        curr_lat, curr_lon, candidates["latitude"].to_numpy(), candidates["longitude"].to_numpy()
    )

    nearby_50 = candidates[candidates["dist_km"] <= 50.0]
    unique_prices = nearby_50["modal_price"].nunique()
    assert unique_prices >= 3, f"Expected at least 3 distinct prices in nearby mandis, got {unique_prices}"
