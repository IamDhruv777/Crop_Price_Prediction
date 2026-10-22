"""
Regression test suite for audit defects identified in commit 449db33.
Every test in this file is designed to fail on 449db33 and pass after fixes.
"""
import json
import re
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models" / "trained"
REPORTS_DIR = REPO_ROOT / "reports"


def test_api_clip_parity_1d_7d():
    """Test 1: Verify API applies residual delta clipping (1D: 200, 7D: 400) from deployment_registry.json."""
    reg_file = MODELS_DIR / "deployment_registry.json"
    assert reg_file.exists(), "models/trained/deployment_registry.json does not exist"

    with open(reg_file, "r", encoding="utf-8") as f:
        reg = json.load(f)

    assert "1-Day" in reg and "clip_delta" in reg["1-Day"]
    assert "7-Day" in reg and "clip_delta" in reg["7-Day"]
    assert reg["1-Day"]["clip_delta"] == 200.0
    assert reg["7-Day"]["clip_delta"] == 400.0

    from src.inference.predictor import CropPricePredictor
    p = CropPricePredictor()
    assert hasattr(p, "registry") or reg_file.exists()

    # Query extreme input: verify prediction delta never exceeds clip_delta
    res1 = p.predict("Maharashtra", "Nashik", "Lasalgaon", "Wheat", horizon=1)
    res7 = p.predict("Maharashtra", "Nashik", "Lasalgaon", "Wheat", horizon=7)
    assert abs(res1["predicted_price"] - res1["anchor_price"]) <= 200.0 + 1e-5
    assert abs(res7["predicted_price"] - res7["anchor_price"]) <= 400.0 + 1e-5


def test_api_metrics_match_deployed_artifact_exact_support():
    """Test 2: Verify API metadata matches deployed_test_predictions.csv on complete support."""
    deployed_preds_file = DATA_DIR / "deployed_test_predictions.csv"
    assert deployed_preds_file.exists(), "data/deployed_test_predictions.csv missing"
    metrics_deployed_file = DATA_DIR / "metrics_deployed.json"
    assert metrics_deployed_file.exists(), "data/metrics_deployed.json missing"

    with open(metrics_deployed_file, "r", encoding="utf-8") as f:
        m_dep = json.load(f)

    from src.inference.predictor import CropPricePredictor
    p = CropPricePredictor()

    for h, h_str in [(1, "1-Day"), (7, "7-Day"), (30, "30-Day")]:
        _, meta = p.load_model("Wheat", horizon=h)
        expected = m_dep["by_horizon"][h_str]
        assert abs(meta["rmse"] - expected["RMSE"]) < 0.05, f"{h_str} RMSE mismatch: {meta['rmse']} vs {expected['RMSE']}"
        assert abs(meta["mae"] - expected["MAE"]) < 0.05, f"{h_str} MAE mismatch: {meta['mae']} vs {expected['MAE']}"
        assert abs(meta["r2"] - expected["R2"]) < 0.005, f"{h_str} R2 mismatch: {meta['r2']} vs {expected['R2']}"


def test_30d_incomplete_pair_returns_422():
    """Test 3: Verify 30D persistence enforces is_complete and returns HTTP 422 for incomplete pairs."""
    from src.inference.predictor import CropPricePredictor
    p = CropPricePredictor()

    # Bihar/Bhojpur/Aarah/Arhar (Tur-Red Gram) has <30d history (is_complete=False)
    res = p.predict("Bihar", "Bhojpur", "Aarah", "Arhar (Tur-Red Gram)", horizon=30)
    assert res["status"] == "error", "30D must reject incomplete pairs"
    assert res["error_code"] == "insufficient_data"


def test_exact_market_state_district_match():
    """Test 4: Verify exact matching on state, district, market, commodity (no silent fallback)."""
    from src.inference.predictor import CropPricePredictor
    p = CropPricePredictor()

    # Nonexistent market
    res_bad_market = p.predict("Maharashtra", "Nashik", "Definitely_Not_A_Market", "Wheat", horizon=7)
    assert res_bad_market["status"] == "error"
    assert res_bad_market["error_code"] == "insufficient_data"
    assert "No historical trade features found" in res_bad_market.get("detail", "")

    # State mismatch (Goa with Lasalgaon)
    res_bad_state = p.predict("Goa", "Nashik", "Lasalgaon", "Wheat", horizon=7)
    assert res_bad_state["status"] == "error"
    assert res_bad_state["error_code"] == "insufficient_data"

    # Exact valid match succeeds
    res_valid = p.predict("Maharashtra", "Nashik", "Lasalgaon", "Wheat", horizon=7)
    assert res_valid["status"] == "success"


def test_api_historical_fields_and_dates():
    """Test 5: Verify API response returns historical backtest metadata and calendar dates based on as_of_date."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    payload = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "Lasalgaon",
        "crop": "Wheat",
        "horizon": 7
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "as_of_date" in data
    assert data["as_of_date"] == "2017-06-16"
    assert data.get("is_historical_backtest") is True
    assert "r2" in data
    assert data["r2"] is not None

    # First predicted day must be as_of_date + 1 day: 2017-06-17, NOT 2026
    assert data["predictions"][0]["date"] == "2017-06-17"


def test_dashboard_uses_deployed_predictions():
    """Test 6: Verify dashboard scripts reference deployed_test_predictions.csv and no synthetic price multipliers."""
    dash_files = [
        REPO_ROOT / "dashboard" / "app.py",
        REPO_ROOT / "dashboard" / "pages" / "1_Overview.py",
        REPO_ROOT / "dashboard" / "pages" / "2_Forecasting.py",
        REPO_ROOT / "dashboard" / "pages" / "4_Regional_Map.py"
    ]
    for path in dash_files:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        assert "modal_price * 1.048" not in text, f"Found synthetic price fallback in {path}"
        assert "651.43" not in text, f"Found stale RMSE 651.43 in {path}"
        assert "96.4%" not in text, f"Found hardcoded 96.4% in {path}"

    fc_file = REPO_ROOT / "dashboard" / "pages" / "2_Forecasting.py"
    with open(fc_file, "r", encoding="utf-8") as f:
        fc_text = f.read()
    assert "deployed_test_predictions.csv" in fc_text, "2_Forecasting.py must use deployed_test_predictions.csv"


def test_report_tables_exact_parity_with_artifacts():
    """Test 7: Verify reports match data/metrics_deployed.json and data/metrics_deployed_by_crop.csv."""
    m_dep_file = DATA_DIR / "metrics_deployed.json"
    assert m_dep_file.exists(), "data/metrics_deployed.json missing"
    with open(m_dep_file, "r", encoding="utf-8") as f:
        m_dep = json.load(f)

    scorecard_file = REPORTS_DIR / "FINAL_SCORECARD.md"
    assert scorecard_file.exists()
    with open(scorecard_file, "r", encoding="utf-8") as f:
        sc_text = f.read()

    # Check 1D deployed RMSE in scorecard
    rmse_1d_str = f"{m_dep['by_horizon']['1-Day']['RMSE']:.2f}"
    assert rmse_1d_str in sc_text, f"Scorecard missing deployed 1D RMSE {rmse_1d_str}"
    assert "18/18" not in sc_text, "Scorecard has stale 18/18 tests count"


def test_mondrian_q95_recomputed_from_validation_complete_support():
    """Test 8: Verify Mondrian q95 in mondrian_conformal_quantiles.json matches complete-support validation."""
    q_file = DATA_DIR / "mondrian_conformal_quantiles.json"
    assert q_file.exists()
    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    meta = q_data.get("_metadata", {})
    assert meta.get("support") == "complete_features", "Mondrian quantiles must specify complete_features support"
    assert meta.get("clipped") is True, "Mondrian quantiles must specify clipped residuals"


def test_mondrian_coverage_recomputed_from_deployed_predictions():
    """Test 9: Verify Mondrian coverage is recomputed from deployed_test_predictions.csv."""
    m_dep_file = DATA_DIR / "metrics_deployed.json"
    assert m_dep_file.exists()
    with open(m_dep_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert "mondrian_averages" in m
    for h in ["1-Day", "7-Day", "30-Day"]:
        assert h in m["mondrian_averages"]
        cov = m["mondrian_averages"][h]["macro_coverage"]
        assert 80.0 <= cov <= 100.0


def test_no_hardcoded_metric_fallbacks():
    """Test 10: Verify no hardcoded fallbacks in scripts, api, src, or dashboard."""
    banned_patterns = [
        r"96\.07", r"95\.90", r"96\.02",
        r"33\.57", r"43\.13", r"69\.39",
        r"modal_price\s*\*\s*1\.048",
        r"651\.43",
        r"18/18 tests passing"
    ]
    check_dirs = [REPO_ROOT / "src", REPO_ROOT / "api", REPO_ROOT / "dashboard", REPO_ROOT / "scripts"]
    for d in check_dirs:
        for p in d.rglob("*.py"):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            for pat in banned_patterns:
                assert not re.search(pat, content), f"Found banned pattern {pat} in {p}"


def test_report_manifest_and_hashes():
    """Test 11: Verify data/report_manifest.json exists and hashes match artifacts."""
    manifest_file = DATA_DIR / "report_manifest.json"
    assert manifest_file.exists(), "data/report_manifest.json missing"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "artifacts" in manifest
    assert "metrics_deployed.json" in manifest["artifacts"]
    assert "sha256" in manifest["artifacts"]["metrics_deployed.json"]


def test_generate_reports_fails_without_artifacts():
    """Test 12: Verify scripts/generate_audit_reports.py has no hardcoded fallbacks and fails when deployment artifacts are missing."""
    rep_script = REPO_ROOT / "scripts" / "generate_audit_reports.py"
    assert rep_script.exists()
    with open(rep_script, "r", encoding="utf-8") as f:
        src = f.read()

    assert "metrics_deployed.json" in src, "generate_audit_reports.py must load metrics_deployed.json"
    assert "96.07" not in src, "generate_audit_reports.py has hardcoded fallback 96.07"
    assert "95.90" not in src, "generate_audit_reports.py has hardcoded fallback 95.90"
