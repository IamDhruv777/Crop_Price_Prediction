"""
Spatial distance calculations.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from math import radians, sin, cos, sqrt, atan2


EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def compute_distance_matrix(market_df: pd.DataFrame) -> pd.DataFrame:
    """Compute full pairwise distance matrix between markets."""
    # market_df must have: market_id, latitude, longitude
    markets = market_df[['market_id', 'latitude', 'longitude']].copy()
    n = len(markets)
    lats = markets['latitude'].values
    lons = markets['longitude'].values
    ids = markets['market_id'].values
    
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = haversine_distance(lats[i], lons[i], lats[j], lons[j])
    
    dist_df = pd.DataFrame(distances, index=ids, columns=ids)
    return dist_df


def compute_edge_weight(distance_km: float, sigma: float = 50.0) -> float:
    """Gaussian kernel edge weight: w = exp(-d / sigma)."""
    return float(np.exp(-distance_km / sigma))
