import pytest
import numpy as np
from src.inference.predictor import CropPricePredictor

def test_crop_price_predictor_horizons():
    predictor = CropPricePredictor()
    
    for h in [1, 7, 30]:
        res = predictor.predict(
            state="Maharashtra",
            district="Nashik",
            market="Lasalgaon",
            crop="Wheat",
            horizon=h,
            confidence_level=0.95
        )
        assert res is not None
        assert res["horizon"] == h
        assert res["predicted_price"] > 0
        assert len(res["predictions_trajectory"]) == h
        
        for step in res["predictions_trajectory"]:
            assert step["lower_bound"] <= step["predicted_price"]
            assert step["upper_bound"] >= step["predicted_price"]
            assert step["lower_bound"] >= 100.0

def test_crop_price_predictor_invalid_horizon():
    predictor = CropPricePredictor()
    with pytest.raises(ValueError):
        predictor.predict(
            state="Maharashtra",
            district="Pune",
            market="Pune",
            crop="Rice",
            horizon=14
        )

def test_farmer_recommendation_logic():
    predictor = CropPricePredictor()
    res = predictor.predict(
        state="Maharashtra",
        district="Solapur",
        market="Solapur",
        crop="Soyabean",
        horizon=7
    )
    assert "recommendation" in res
    assert any(tag in res["recommendation"] for tag in ["माल रोकें", "आज बेचें", "स्थिर"])
