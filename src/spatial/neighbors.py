"""
KNN market neighbor module.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from .distance import haversine_distance, compute_edge_weight


def find_knn_markets(market_df: pd.DataFrame, k: int = 5, max_distance_km: float = 200, sigma: float = 50.0) -> pd.DataFrame:
    """
    Find k nearest neighbors for each market within max_distance_km.
    Creates a sparse graph.
    """
    markets = market_df[['market_id', 'latitude', 'longitude']].copy()
    n = len(markets)
    lats = markets['latitude'].values
    lons = markets['longitude'].values
    ids = markets['market_id'].values
    
    edges = []
    
    for i in range(n):
        source_id = ids[i]
        distances = []
        for j in range(n):
            if i != j:
                d = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                if d <= max_distance_km:
                    distances.append((ids[j], d))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Take top k
        top_k = distances[:k]
        
        for rank, (target_id, dist) in enumerate(top_k, 1):
            weight = compute_edge_weight(dist, sigma)
            edges.append({
                'source_market_id': source_id,
                'neighbor_market_id': target_id,
                'distance_km': dist,
                'rank': rank,
                'edge_weight': weight
            })
            
    if not edges:
        return pd.DataFrame(columns=['source_market_id', 'neighbor_market_id', 'distance_km', 'rank', 'edge_weight'])
        
    return pd.DataFrame(edges)


def get_neighbor_ids(neighbors_df: pd.DataFrame, market_id: str) -> List[str]:
    """Return list of neighbor market_ids for a given market."""
    if neighbors_df.empty:
        return []
    neighbors = neighbors_df[neighbors_df['source_market_id'] == market_id]
    return neighbors['neighbor_market_id'].tolist()


def build_adjacency_dict(neighbors_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build adjacency list dict: {market_id: [neighbor_ids]}"""
    adj_dict = {}
    if neighbors_df.empty:
        return adj_dict
        
    for source_id, group in neighbors_df.groupby('source_market_id'):
        adj_dict[source_id] = group['neighbor_market_id'].tolist()
        
    return adj_dict
