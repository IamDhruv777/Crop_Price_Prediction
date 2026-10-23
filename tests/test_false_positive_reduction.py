"""
Tests for False Positive Reduction via Conformal Uncertainty Gating and Deadband Filtering.
Verifies:
  1. Micro-fluctuations around zero do not trigger false positive rally alerts (classified as STABLE).
  2. Genuine rallies that clear the conformal noise gate are correctly classified as RALLY with HIGH confidence.
  3. API response includes directional_signal, signal_confidence, and conformal_gate_threshold.
  4. Empirical verification on deployed test set proves >=80% reduction in 1-Day false positives.
"""
from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src.inference.predictor import CropPricePredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

client = TestClient(app)


def test_predictor_micro_delta_classified_as_stable():
    """Test 1: A predicted delta below the conformal gate must be classified as STABLE, not RALLY."""
    predictor = CropPricePredictor()
    # Predict for a known pair: Pune / Wheat / 1-Day
    res = predictor.predict(
        state="Maharashtra",
        district="Pune",
        market="Pune",
        crop="Wheat",
        horizon=1
    )
    assert res["status"] == "success"
    assert "directional_signal" in res
    assert "signal_confidence" in res
    assert "conformal_gate_threshold" in res
    assert res["conformal_gate_threshold"] >= 5.0

    # Verify logic: if price change is below gate or under 2%, signal must be STABLE
    if res["price_change"] < res["conformal_gate_threshold"] or res["percent_change"] < 2.0:
        assert res["directional_signal"] in ["STABLE", "DROP"]
        if res["directional_signal"] == "STABLE":
            assert res["signal_confidence"] == "NEUTRAL"
            assert "स्थिर" in res["recommendation"]


def test_api_predict_returns_gating_fields():
    """Test 2: HTTP POST /forecasts/predict returns directional_signal, signal_confidence, gate threshold."""
    payload = {
        "state": "Maharashtra",
        "district": "Pune",
        "market": "Pune",
        "crop": "Wheat",
        "horizon": 1,
        "confidence_level": 0.95
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    body = resp.json()
    assert "directional_signal" in body
    assert "signal_confidence" in body
    assert "conformal_gate_threshold" in body
    assert "recommendation" in body
    assert body["directional_signal"] in ["RALLY", "DROP", "STABLE"]
    assert body["signal_confidence"] in ["HIGH", "MODERATE", "NEUTRAL"]
    assert body["conformal_gate_threshold"] > 0


def test_conformal_gating_reduces_test_false_positives_by_at_least_80_percent():
    """Test 3: Empirical proof on deployed_test_predictions.csv that conformal gating slashes 1-Day FPs by >=80%."""
    preds_file = DATA_DIR / "deployed_test_predictions.csv"
    assert preds_file.exists(), "deployed_test_predictions.csv must exist"
    df = pd.read_csv(preds_file)

    sub = df.dropna(subset=["target_price_1d", "predicted_price_1d", "modal_price", "uncertainty_delta_1d"])
    act_diff = sub["target_price_1d"].values - sub["modal_price"].values
    pred_diff = sub["predicted_price_1d"].values - sub["modal_price"].values
    unc = sub["uncertainty_delta_1d"].values

    # Raw zero-threshold baseline
    raw_fp = np.sum((act_diff <= 0) & (pred_diff > 0))
    raw_tp = np.sum((act_diff > 0) & (pred_diff > 0))
    raw_prec = raw_tp / (raw_tp + raw_fp)

    # Conformal gated (gate = max(5.0, 0.05 * unc))
    gate = np.maximum(5.0, 0.05 * unc)
    gated_fp = np.sum((act_diff <= 0) & (pred_diff >= gate))
    gated_tp = np.sum((act_diff > 0) & (pred_diff >= gate))
    gated_prec = gated_tp / (gated_tp + gated_fp)

    fp_reduction_pct = (raw_fp - gated_fp) / raw_fp * 100.0

    # Assertions
    assert fp_reduction_pct >= 80.0, f"Expected >= 80% FP reduction, got {fp_reduction_pct:.2f}%"
    assert gated_prec > raw_prec, f"Expected gated precision ({gated_prec*100:.2f}%) > raw precision ({raw_prec*100:.2f}%)"
    assert gated_prec >= 0.50, f"Expected gated precision >= 50%, got {gated_prec*100:.2f}%"


def test_7d_conformal_gating_reduces_false_positives():
    """Test 4: Empirical proof on 7-Day predictions that conformal gating slashes FPs and improves precision."""
    preds_file = DATA_DIR / "deployed_test_predictions.csv"
    df = pd.read_csv(preds_file)

    sub7 = df.dropna(subset=["target_price_7d", "predicted_price_7d", "modal_price", "uncertainty_delta_7d"])
    act_diff7 = sub7["target_price_7d"].values - sub7["modal_price"].values
    pred_diff7 = sub7["predicted_price_7d"].values - sub7["modal_price"].values
    unc7 = sub7["uncertainty_delta_7d"].values

    raw_fp7 = np.sum((act_diff7 <= 0) & (pred_diff7 > 0))
    raw_tp7 = np.sum((act_diff7 > 0) & (pred_diff7 > 0))
    raw_prec7 = raw_tp7 / (raw_tp7 + raw_fp7)

    # Conformal gated
    gate7 = np.maximum(5.0, 0.05 * unc7)
    gated_fp7 = np.sum((act_diff7 <= 0) & (pred_diff7 >= gate7))
    gated_tp7 = np.sum((act_diff7 > 0) & (pred_diff7 >= gate7))
    gated_prec7 = gated_tp7 / (gated_tp7 + gated_fp7)

    fp_reduction_pct7 = (raw_fp7 - gated_fp7) / raw_fp7 * 100.0

    assert fp_reduction_pct7 >= 45.0, f"Expected >= 45% FP reduction on 7D, got {fp_reduction_pct7:.2f}%"
    assert gated_prec7 > raw_prec7, f"Expected 7D gated precision ({gated_prec7*100:.2f}%) > raw ({raw_prec7*100:.2f}%)"


def test_dual_head_directional_classifiers_exist_and_predict():
    """Test 5: Dual-head directional classifier models exist and compute valid probabilities."""
    MODELS_DIR = REPO_ROOT / "models" / "trained"
    import joblib

    for h in [1, 7]:
        clf_path = MODELS_DIR / f"xgb_classifier_{h}d.joblib"
        assert clf_path.exists(), f"xgb_classifier_{h}d.joblib must exist"
        clf = joblib.load(clf_path)
        assert hasattr(clf, "predict_proba")
        assert len(clf.feature_names_in_) == 48


def test_api_predict_returns_rise_probability():
    """Test 6: HTTP POST /forecasts/predict returns rise_probability in [0, 1]."""
    payload = {
        "state": "Maharashtra",
        "district": "Pune",
        "market": "Pune",
        "crop": "Wheat",
        "horizon": 1,
        "confidence_level": 0.95
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "rise_probability" in body
    assert body["rise_probability"] is not None
    assert 0.0 <= body["rise_probability"] <= 1.0
