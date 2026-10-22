import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_api_predict():
    payload = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "Lasalgaon",
        "crop": "Wheat",
        "horizon": 7,
        "confidence_level": 0.95
    }
    response = client.post("/forecasts/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Wheat"
    assert data["market"] == "Lasalgaon"
    assert len(data["predictions"]) == 7
    assert data["rmse"] > 0
