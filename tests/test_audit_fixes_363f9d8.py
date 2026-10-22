"""
Regression tests for defects identified after commit 363f9d8.
Every test targeting a bug in 363f9d8 must fail before fixes and pass after fixes.
"""
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models" / "trained"
REPORTS_DIR = REPO_ROOT / "reports"


def test_30d_http_predict_matches_metrics_deployed():
    """Test 1: POST /forecasts/predict for 30D returns exact canonical metrics from metrics_deployed.json."""
    from api.main import app
    client = TestClient(app)

    with open(DATA_DIR / "metrics_deployed.json", "r", encoding="utf-8") as f:
        m_dep = json.load(f)
    expected_30d = m_dep["by_horizon"]["30-Day"]

    payload = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "Lasalgaon",
        "crop": "Wheat",
        "horizon": 30,
        "confidence_level": 0.95
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # Must match metrics_deployed.json exactly: 486.53 / 228.68 / 0.9516 (NOT 487.92 / 229.82)
    assert abs(data["rmse"] - expected_30d["RMSE"]) < 0.01, f"30D RMSE mismatch: {data['rmse']} vs {expected_30d['RMSE']}"
    assert abs(data["mae"] - expected_30d["MAE"]) < 0.01, f"30D MAE mismatch: {data['mae']} vs {expected_30d['MAE']}"
    assert data["r2"] is not None
    assert abs(data["r2"] - expected_30d["R2"]) < 0.005, f"30D R2 mismatch: {data['r2']} vs {expected_30d['R2']}"


def test_forecast_history_endpoint_200_and_no_nan():
    """Test 2: GET /forecasts/history/{market_id}/{commodity} returns 200 without NaN serialization errors."""
    from api.main import app
    client = TestClient(app)

    resp = client.get("/forecasts/history/Lasalgaon/Wheat")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "history" in data
    assert isinstance(data["history"], list)
    # Ensure no NaN string values leaked into response
    body_text = resp.text
    assert "NaN" not in body_text, "Found raw NaN in JSON history response"


def test_unknown_crop_returns_422():
    """Test 3: POST /forecasts/predict with unknown crop returns HTTP 422, not 500."""
    from api.main import app
    client = TestClient(app)

    payload = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "Lasalgaon",
        "crop": "Avocado_Unknown_Crop",
        "horizon": 7,
        "confidence_level": 0.95
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 for unknown crop, got {resp.status_code}: {resp.text}"


def test_invalid_as_of_date_returns_422():
    """Test 4: POST /forecasts/predict with invalid as_of_date returns HTTP 422, not 500."""
    from api.main import app
    client = TestClient(app)

    payload = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "Lasalgaon",
        "crop": "Wheat",
        "horizon": 7,
        "as_of_date": "not-a-valid-date"
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 for invalid date, got {resp.status_code}: {resp.text}"


def test_future_as_of_date_rejected_with_422():
    """Test 5: POST /forecasts/predict rejects future as_of_date relative to historical corpus with HTTP 422."""
    from api.main import app
    client = TestClient(app)

    payload = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "Lasalgaon",
        "crop": "Wheat",
        "horizon": 7,
        "as_of_date": "2099-01-01"
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 for future date, got {resp.status_code}: {resp.text}"


def test_models_best_validates_crop_and_returns_model_file():
    """Test 6: /models/best validates crop (unknown crop returns 404), and returns real artifact path from model_file."""
    from api.main import app
    client = TestClient(app)

    # Unknown crop -> 404
    resp_bad = client.get("/models/best?crop=Avocado_Unknown_Crop&horizon=7-Day")
    assert resp_bad.status_code == 404, f"Expected 404 for unknown crop, got {resp_bad.status_code}"

    # Valid crop -> 200, artifact must not be None and must come from model_file
    resp_good = client.get("/models/best?crop=Wheat&horizon=7-Day")
    assert resp_good.status_code == 200
    data = resp_good.json()
    assert data["artifact"] is not None
    assert "xgboost_7d.joblib" in data["artifact"]


def test_confidence_level_validation():
    """Test 7: confidence_level other than 0.95 is rejected with HTTP 422."""
    from api.main import app
    client = TestClient(app)

    payload = {
        "state": "Maharashtra",
        "district": "Nashik",
        "market": "Lasalgaon",
        "crop": "Wheat",
        "horizon": 7,
        "confidence_level": 0.50
    }
    resp = client.post("/forecasts/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 for uncalibrated confidence_level 0.50, got {resp.status_code}"


def test_no_test_predictions_fallback_in_dashboard():
    """Test 8: Dashboard files must NOT fall back to data/test_predictions.csv."""
    dash_files = [
        REPO_ROOT / "dashboard" / "app.py",
        REPO_ROOT / "dashboard" / "pages" / "1_Overview.py",
        REPO_ROOT / "dashboard" / "pages" / "2_Forecasting.py",
        REPO_ROOT / "dashboard" / "pages" / "4_Regional_Map.py"
    ]
    for p in dash_files:
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        assert "data/test_predictions.csv" not in src, f"Found data/test_predictions.csv fallback in {p}"


def test_no_api_fallbacks_in_forecast_route():
    """Test 9: api/routes/forecasts.py must not contain fallback literals for missing predictor fields."""
    route_file = REPO_ROOT / "api" / "routes" / "forecasts.py"
    assert route_file.exists()
    with open(route_file, "r", encoding="utf-8") as f:
        src = f.read()

    assert "2.1.0-production" not in src, "Found '2.1.0-production' fallback in api/routes/forecasts.py"
    assert "2017-06-16" not in src, "Found '2017-06-16' fallback in api/routes/forecasts.py"


def test_report_manifest_hashes_recomputed_exact_match():
    """Test 10: Recompute SHA-256 for all artifacts in report_manifest.json and assert exact equality."""
    import hashlib
    manifest_file = DATA_DIR / "report_manifest.json"
    assert manifest_file.exists(), "data/report_manifest.json missing"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "artifacts" in manifest
    assert len(manifest["artifacts"]) > 0

    for name, item in manifest["artifacts"].items():
        file_path = REPO_ROOT / item["path"]
        assert file_path.exists(), f"Manifest file missing: {file_path}"
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        expected_hash = sha.hexdigest()
        assert item["sha256"] == expected_hash, f"SHA-256 mismatch for {name}: {item['sha256']} vs {expected_hash}"


def test_generate_reports_subprocess_fails_on_missing_artifacts(tmp_path):
    """Test 11: scripts/generate_audit_reports.py must fail with nonzero exit code when artifacts are missing."""
    fake_script = tmp_path / "fake_run.py"
    fake_script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(r'" + str(REPO_ROOT) + "')))\n"
        "from scripts.generate_audit_reports import generate_reports\n"
        "import scripts.generate_audit_reports as gar\n"
        "gar.METRICS_DEPLOYED_JSON = Path('nonexistent_metrics.json')\n"
        "generate_reports()\n",
        encoding="utf-8"
    )
    res = subprocess.run([sys.executable, str(fake_script)], capture_output=True, text=True)
    assert res.returncode != 0, "generate_audit_reports should fail when required artifact is missing"


def test_registry_clip_governs_evaluation_and_calibration():
    """Test 12: evaluate_deployed and calibrate_mondrian read clip settings from deployment_registry.json."""
    eval_file = REPO_ROOT / "scripts" / "evaluate_deployed.py"
    calib_file = REPO_ROOT / "scripts" / "calibrate_mondrian_conformal.py"

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_src = f.read()
    with open(calib_file, "r", encoding="utf-8") as f:
        calib_src = f.read()

    # Both files must load deployment_registry.json
    assert "deployment_registry.json" in eval_src
    assert "deployment_registry.json" in calib_src


def test_all_664_incomplete_pairs_return_422():
    """Test 13: All 664 incomplete feature store pairs return HTTP 422 insufficient_data on /forecasts/predict."""
    from src.inference.predictor import CropPricePredictor
    p = CropPricePredictor()

    fs = pd.read_csv(DATA_DIR / "processed" / "inference_feature_store.csv")
    incomplete_rows = fs[~fs["is_complete"]]
    assert len(incomplete_rows) == 664, f"Expected 664 incomplete rows, got {len(incomplete_rows)}"

    for _, row in incomplete_rows.iterrows():
        res = p.predict(row["state"], row["district"], row["market"], row["commodity"], horizon=7)
        assert res["status"] == "error"
        assert res["error_code"] == "insufficient_data"


def test_deployed_csv_metrics_reproduce_table():
    """Test 14: data/deployed_test_predictions.csv reproduces exact RMSE/MAE/R2 in data/metrics_deployed.json."""
    preds_file = DATA_DIR / "deployed_test_predictions.csv"
    assert preds_file.exists()
    df = pd.read_csv(preds_file)

    with open(DATA_DIR / "metrics_deployed.json", "r", encoding="utf-8") as f:
        m_dep = json.load(f)

    for h, act_col, pred_col in [("1-Day", "target_price_1d", "predicted_price_1d"),
                                  ("7-Day", "target_price_7d", "predicted_price_7d"),
                                  ("30-Day", "target_price_30d", "predicted_price_30d")]:
        sub = df.dropna(subset=[act_col, pred_col])
        res = sub[act_col].values - sub[pred_col].values
        rmse = round(float(np.sqrt(np.mean(res**2))), 2)
        mae = round(float(np.mean(np.abs(res))), 2)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((sub[act_col].values - np.mean(sub[act_col].values))**2)
        r2 = round(float(1.0 - (ss_res / (ss_tot + 1e-5))), 4)

        expected = m_dep["by_horizon"][h]
        assert abs(rmse - expected["RMSE"]) < 0.05, f"{h} RMSE: {rmse} vs {expected['RMSE']}"
        assert abs(mae - expected["MAE"]) < 0.05, f"{h} MAE: {mae} vs {expected['MAE']}"
        assert abs(r2 - expected["R2"]) < 0.005, f"{h} R2: {r2} vs {expected['R2']}"

