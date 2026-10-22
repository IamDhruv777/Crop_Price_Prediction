import pytest
from src.spatial.distance import haversine_distance, compute_edge_weight

def test_haversine_distance():
    # Mumbai to Pune coordinates (~120-150 km)
    mumbai_lat, mumbai_lon = 18.9388, 72.8354
    pune_lat, pune_lon = 18.5204, 73.8567
    
    dist = haversine_distance(mumbai_lat, mumbai_lon, pune_lat, pune_lon)
    assert 110.0 <= dist <= 140.0

def test_haversine_zero_distance():
    dist = haversine_distance(19.0, 73.0, 19.0, 73.0)
    assert dist == 0.0

def test_edge_weight():
    w0 = compute_edge_weight(0.0, sigma=50.0)
    assert w0 == 1.0
    
    w100 = compute_edge_weight(100.0, sigma=50.0)
    assert 0.0 < w100 < 1.0
