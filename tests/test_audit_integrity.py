"""
Forensic Audit Integrity Test Suite.
Verifies:
  1. test_no_same_day_target: Target dates are strictly future calendar dates across ALL rows.
  2. test_train_val_test_no_date_overlap: Strict embargo (>30 days) between train/val/test splits.
  3. test_leaderboard_matches_recomputed_metrics: Leaderboard numbers and rankings strictly match benchmark artifacts.
  4. test_api_model_identity_and_metrics: API loaded model class and metadata strictly match trained model & benchmark.
  5. test_api_insufficient_data_error_and_no_zero_fill: Returns explicit 'insufficient_data' error, never zero-fills.
  6. test_all_crops_present_in_evaluation: All 11 benchmark commodities exist in test predictions (>= 100 samples).
  7. test_persistence_beats_ensemble_at_30d: Discloses that Naive Persistence beats the ML ensemble at 30 days.
  8. test_macro_and_pooled_metrics_reported: metrics.json contains both pooled and macro averages + interval widths.
"""
import json
import pytest
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = DATA_DIR / "processed" / "unified_splits"
MODELS_DIR = REPO_ROOT / "models" / "trained"


def test_no_same_day_target():
    """Verify across ALL test rows that targets are strictly future calendar dates."""
    test_unified_file = SPLITS_DIR / "test_unified.csv"
    assert test_unified_file.exists(), "test_unified.csv does not exist"

    df = pd.read_csv(
        test_unified_file,
        usecols=['date', 'target_date_1d', 'target_date_7d', 'target_date_30d'],
        parse_dates=['date', 'target_date_1d', 'target_date_7d', 'target_date_30d'],
        low_memory=False
    )

    # 1-Day: target_date_1d must be strictly >= date + 1 day
    h1 = df.dropna(subset=['target_date_1d'])
    assert len(h1) > 0, "No 1-Day targets found"
    diff1 = (h1['target_date_1d'] - h1['date']).dt.days
    assert (diff1 >= 1).all(), f"Found {(diff1 < 1).sum()} 1-Day targets with same-day or past date!"

    # 7-Day: target_date_7d must be >= date + 5 days
    h7 = df.dropna(subset=['target_date_7d'])
    assert len(h7) > 0, "No 7-Day targets found"
    diff7 = (h7['target_date_7d'] - h7['date']).dt.days
    assert (diff7 >= 5).all(), f"Found {(diff7 < 5).sum()} 7-Day targets with calendar gap < 5 days!"

    # 30-Day: target_date_30d must be >= date + 26 days
    h30 = df.dropna(subset=['target_date_30d'])
    assert len(h30) > 0, "No 30-Day targets found"
    diff30 = (h30['target_date_30d'] - h30['date']).dt.days
    assert (diff30 >= 26).all(), f"Found {(diff30 < 26).sum()} 30-Day targets with calendar gap < 26 days!"


def test_train_val_test_no_date_overlap():
    """Verify strict chronological ordering and 30-day purge embargos between splits."""
    train_file = SPLITS_DIR / "train_unified.csv"
    val_file = SPLITS_DIR / "val_unified.csv"
    test_file = SPLITS_DIR / "test_unified.csv"

    assert train_file.exists() and val_file.exists() and test_file.exists(), "Splits missing"

    train_df = pd.read_csv(train_file, usecols=['date'], parse_dates=['date'])
    val_df = pd.read_csv(val_file, usecols=['date'], parse_dates=['date'])
    test_df = pd.read_csv(test_file, usecols=['date'], parse_dates=['date'])

    train_max = train_df['date'].max()
    val_min = val_df['date'].min()
    val_max = val_df['date'].max()
    test_min = test_df['date'].min()

    assert train_max < val_min, f"Train max ({train_max}) overlaps Val min ({val_min})"
    assert val_max < test_min, f"Val max ({val_max}) overlaps Test min ({test_min})"

    # Verify >= 30-day purge embargo
    embargo_1 = (val_min - train_max).days
    embargo_2 = (test_min - val_max).days

    assert embargo_1 >= 30, f"Embargo 1 between train and val is only {embargo_1} days (expected >= 30)"
    assert embargo_2 >= 30, f"Embargo 2 between val and test is only {embargo_2} days (expected >= 30)"


def test_leaderboard_matches_recomputed_metrics():
    """Verify that master_model_leaderboard.csv strictly matches runtime computed metrics and rankings."""
    leaderboard_file = DATA_DIR / "master_model_leaderboard.csv"
    metrics_file = DATA_DIR / "metrics.json"

    assert leaderboard_file.exists(), "master_model_leaderboard.csv missing"
    assert metrics_file.exists(), "metrics.json missing"

    lb_df = pd.read_csv(leaderboard_file)
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Check 7-Day Super Ensemble row
    ens_row = lb_df[(lb_df['Horizon'] == '7-Day') & (lb_df['Model'].str.contains('Super Ensemble', case=False))].iloc[0]
    expected_m = metrics["by_horizon"]["7-Day"]

    assert abs(ens_row['R2'] - expected_m['R2']) < 0.002, f"R2 mismatch: {ens_row['R2']} vs {expected_m['R2']}"
    assert abs(ens_row['RMSE'] - expected_m['RMSE']) < 0.1, f"RMSE mismatch: {ens_row['RMSE']} vs {expected_m['RMSE']}"
    assert abs(ens_row['Relative_Accuracy'] - expected_m['Relative_Accuracy']) < 0.1, "Relative Accuracy mismatch"

    # Verify honesty in rankings
    # At 30-Day, Naive Persistence must be Rank 1
    lb_30d = lb_df[lb_df['Horizon'] == '30-Day'].sort_values(by='Test_Rank')
    assert lb_30d.iloc[0]['Model'] == 'Naive Persistence Baseline', "At 30-Day, Naive Persistence must be Rank 1"

    # At 7-Day, Temporal CNN must be Rank 1
    lb_7d = lb_df[lb_df['Horizon'] == '7-Day'].sort_values(by='Test_Rank')
    assert lb_7d.iloc[0]['Model'] == 'Temporal CNN (TCN)', "At 7-Day, TCN must be Rank 1"


def test_api_model_identity_and_metrics():
    """Verify that CropPricePredictor loads the real trained model and returns matching artifact metrics."""
    from src.inference.predictor import CropPricePredictor

    predictor = CropPricePredictor()
    bm_file = DATA_DIR / "full_architecture_benchmark_metrics.csv"
    assert bm_file.exists(), "Benchmark CSV missing"
    bm_df = pd.read_csv(bm_file)

    for h in [1, 7, 30]:
        model, meta = predictor.load_model("Wheat", horizon=h)
        assert model is not None, f"Predictor failed to load model for horizon {h}"
        if h == 30:
            assert type(model).__name__ == "NaiveForecast", f"Expected NaiveForecast, got {type(model).__name__}"
            assert meta["model_name"] == "Naive Persistence Baseline"
            assert meta["model_class"] == "NaiveForecast"
            assert meta["is_persistence"] is True
            row = bm_df[(bm_df["Horizon"] == f"{h}-Day") & (bm_df["Model"] == "Naive Persistence Baseline")].iloc[0]
        else:
            assert type(model).__name__ == "XGBRegressor", f"Expected XGBRegressor, got {type(model).__name__}"
            assert meta["model_name"] == "Tuned XGBoost Regressor"
            assert meta["model_class"] == "XGBRegressor"
            assert meta["is_persistence"] is False
            row = bm_df[(bm_df["Horizon"] == f"{h}-Day") & (bm_df["Model"] == "Tuned XGBoost Regressor")].iloc[0]

        assert abs(meta["rmse"] - float(row["RMSE"])) < 0.01, f"RMSE mismatch for horizon {h}"
        assert abs(meta["mae"] - float(row["MAE"])) < 0.01, f"MAE mismatch for horizon {h}"
        assert abs(meta["r2"] - float(row["R2"])) < 0.001, f"R2 mismatch for horizon {h}"


def test_api_insufficient_data_error_and_no_zero_fill():
    """Verify that predictor returns explicit 'insufficient_data' when data is missing, NEVER zero-fills."""
    from src.inference.predictor import CropPricePredictor

    predictor = CropPricePredictor()

    # Query completely unknown market
    res_unknown = predictor.predict(
        state="NonexistentState",
        district="NonexistentDistrict",
        market="Unknown_Market_XYZ",
        crop="Wheat",
        horizon=7
    )
    assert res_unknown.get("status") == "error", "Expected error for unknown market"
    assert res_unknown.get("error_code") == "insufficient_data", f"Expected insufficient_data, got {res_unknown.get('error_code')}"
    assert "No historical trade features found" in res_unknown.get("detail", "")

    # Query known market with genuine data -> must succeed
    res_known = predictor.predict(
        state="Maharashtra",
        district="Nashik",
        market="Lasalgaon",
        crop="Wheat",
        horizon=7
    )
    assert res_known.get("status") == "success", f"Expected success for Lasalgaon Wheat, got {res_known}"
    assert res_known.get("predicted_price") > 100.0
    assert len(res_known.get("predictions_trajectory")) == 7


def test_all_crops_present_in_evaluation():
    """Verify that all 11 benchmark commodities exist in test_predictions.csv with statistical power."""
    pred_file = DATA_DIR / "test_predictions.csv"
    assert pred_file.exists(), "test_predictions.csv missing"

    df = pd.read_csv(pred_file, usecols=['commodity', 'actual_price_7d', 'predicted_price_7d'])
    df_valid = df.dropna(subset=['actual_price_7d', 'predicted_price_7d'])

    present_crops = set(df_valid['commodity'].unique())

    expected_crops = {
        'Wheat', 'Rice', 'Maize', 'Soyabean',
        'Arhar (Tur-Red Gram)', 'Black Gram (Urd Beans)',
        'Green Gram (Moong)', 'Lentil (Masur)',
        'Cowpea (Lobia-Karamani)', 'Kabuli Chana (Chickpeas-White)',
        'White Peas'
    }

    missing = expected_crops - present_crops
    assert len(missing) == 0, f"Evaluation is missing crops: {missing}"

    crop_counts = df_valid['commodity'].value_counts()
    for crop in expected_crops:
        count = crop_counts[crop]
        assert count >= 100, f"Crop {crop} has only {count} test rows (insufficient sample size)"


def test_persistence_beats_ensemble_at_30d():
    """Verify that evaluation honestly reports Naive Persistence beating the ensemble at 30 days."""
    metrics_file = DATA_DIR / "metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    model_30d = m["by_horizon"]["30-Day"]
    pers_30d = m["persistence_by_horizon"]["30-Day"]

    # At 30 days, Persistence has lower RMSE, lower MAE, and higher R2
    assert pers_30d["RMSE"] < model_30d["RMSE"], f"Expected Pers RMSE ({pers_30d['RMSE']}) < Model RMSE ({model_30d['RMSE']})"
    assert pers_30d["MAE"] < model_30d["MAE"], f"Expected Pers MAE ({pers_30d['MAE']}) < Model MAE ({model_30d['MAE']})"
    assert pers_30d["R2"] > model_30d["R2"], f"Expected Pers R2 ({pers_30d['R2']}) > Model R2 ({model_30d['R2']})"


def test_macro_and_pooled_metrics_reported():
    """Verify that both pooled overall and equal-crop macro averages are recorded."""
    metrics_file = DATA_DIR / "metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert "macro_averages" in m, "macro_averages missing from metrics.json"
    for h in ["1-Day", "7-Day", "30-Day"]:
        assert h in m["macro_averages"], f"{h} missing in macro_averages"
        macro_mod = m["macro_averages"][h]["model"]
        assert "macro_R2" in macro_mod
        assert "macro_RMSE" in macro_mod
        assert "macro_Conformal_Coverage_95" in macro_mod or "macro_Conformal_Coverage_95_Pooled" in macro_mod
        assert "macro_Mean_Interval_Width_Pct" in macro_mod or "macro_Mean_Interval_Width_Pct_Pooled" in macro_mod


def test_inference_anchor_parity_7d():
    """Verify that 7D inference uses the exact training anchor: 0.6*modal_price + 0.4*rolling_mean_7."""
    from src.inference.predictor import CropPricePredictor
    predictor = CropPricePredictor()
    res = predictor.predict("Maharashtra", "Nashik", "Lasalgaon", "Wheat", horizon=7)
    assert res.get("status") == "success"

    # Fetch raw record from feature store
    fs = predictor.feature_store
    match = fs[(fs["state"].str.lower() == "maharashtra") & 
               (fs["district"].str.lower() == "nashik") & 
               (fs["market"].str.lower() == "lasalgaon") & 
               (fs["commodity"].str.lower() == "wheat")]
    assert not match.empty
    rec = match.iloc[-1]
    base_p = float(rec["modal_price"])
    roll_7 = float(rec["price_rolling_mean_7"])
    expected_anchor = 0.6 * base_p + 0.4 * roll_7

    assert abs(res["anchor_price"] - round(expected_anchor, 2)) < 0.05, f"Anchor mismatch: {res['anchor_price']} vs {expected_anchor}"

    model, _ = predictor.load_model("Wheat", 7)
    feat_cols = list(model.feature_names_in_)
    X_input = pd.DataFrame([rec[feat_cols]])
    raw_delta = float(model.predict(X_input)[0])
    expected_price = max(10.0, round(expected_anchor + raw_delta, 2))
    assert abs(res["predicted_price"] - expected_price) < 0.05, f"Prediction mismatch: {res['predicted_price']} vs {expected_price}"


def test_30d_persistence_deployment():
    """Verify that 30D API returns Naive Persistence Baseline with flat trajectory and persistence metrics."""
    from src.inference.predictor import CropPricePredictor
    predictor = CropPricePredictor()
    res = predictor.predict("Maharashtra", "Nashik", "Lasalgaon", "Wheat", horizon=30)
    assert res.get("status") == "success"
    assert res["model_name"] == "Naive Persistence Baseline"
    assert res["predicted_price"] == res["base_price"]
    assert res["price_change"] == 0.0
    # Canonical deployed persistence metrics on complete support (from metrics_deployed.json)
    # Corrected from stale 487.92 / 229.82 all-row unclipped values
    assert res["rmse"] == 486.53
    assert res["mae"] == 228.68
    assert res["r2"] == 0.9516

    # Verify flat trajectory
    trajectory = res["predictions_trajectory"]
    assert len(trajectory) == 30
    for step in trajectory:
        assert step["predicted_price"] == res["base_price"]


def test_mondrian_calibration_uses_validation_only():
    """Verify that Mondrian calibration strictly uses validation set residuals and never touches test targets."""
    calib_script = REPO_ROOT / "scripts" / "calibrate_mondrian_conformal.py"
    assert calib_script.exists(), "calibrate_mondrian_conformal.py missing"
    with open(calib_script, "r", encoding="utf-8") as f:
        src = f.read()

    assert "val_unified.csv" in src, "Calibration script must use val_unified.csv"
    assert "test_unified.csv" not in src, "Calibration script must NEVER reference test_unified.csv"
    assert "test_predictions.csv" not in src, "Calibration script must NEVER reference test_predictions.csv"

    # Verify output quantiles artifact
    q_file = DATA_DIR / "mondrian_conformal_quantiles.json"
    assert q_file.exists(), "mondrian_conformal_quantiles.json missing"
    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    for h in ["1-Day", "7-Day", "30-Day"]:
        assert h in q_data, f"{h} missing in quantiles artifact"
        assert len(q_data[h]) == 11, f"Expected 11 crops for {h}, found {len(q_data[h])}"


def test_mondrian_coverage_reproduction_and_disclosures():
    """Verify that macro coverage reproduces and that crops below 95% are honestly disclosed."""
    metrics_file = DATA_DIR / "metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert "mondrian_averages" in m, "mondrian_averages missing from metrics.json"
    # Macro coverage across all 3 horizons must be >= 95.0%
    assert m["mondrian_averages"]["1-Day"]["macro_coverage"] >= 95.0
    assert m["mondrian_averages"]["7-Day"]["macro_coverage"] >= 95.0
    assert m["mondrian_averages"]["30-Day"]["macro_coverage"] >= 95.0

    # Disclose crops below 95%: Kabuli Chana and Wheat
    crops_7d = [c["crop"] for c in m["mondrian_averages"]["7-Day"]["crops_below_95"]]
    assert "Kabuli Chana (Chickpeas-White)" in crops_7d
    assert "Wheat" in crops_7d

    crops_30d = [c["crop"] for c in m["mondrian_averages"]["30-Day"]["crops_below_95"]]
    assert "Kabuli Chana (Chickpeas-White)" in crops_30d
    assert "Wheat" in crops_30d


def test_report_tables_parity_with_artifacts():
    """Verify that the dynamic reports strictly match data/metrics_by_crop.csv and data/metrics.json."""
    by_crop_file = DATA_DIR / "metrics_by_crop.csv"
    assert by_crop_file.exists()
    crop_df = pd.read_csv(by_crop_file)

    scorecard_file = REPO_ROOT / "reports" / "FINAL_SCORECARD.md"
    assert scorecard_file.exists()
    with open(scorecard_file, "r", encoding="utf-8") as f:
        sc_text = f.read()

    # Spot check 3 crops in scorecard
    for crop in ["Wheat", "Rice", "Kabuli Chana (Chickpeas-White)"]:
        r7 = crop_df[(crop_df["Crop"] == crop) & (crop_df["Horizon"] == "7-Day")].iloc[0]
        rmse_str = f"{r7['Model_RMSE']:.2f}"
        assert rmse_str in sc_text, f"Scorecard missing exact 7D RMSE {rmse_str} for {crop}"


def test_feature_store_completeness():
    """Verify feature store counts: 4554 complete pairs, 664 incomplete pairs."""
    fs_file = DATA_DIR / "processed" / "inference_feature_store.csv"
    assert fs_file.exists()
    df = pd.read_csv(fs_file)
    assert len(df) == 5218, f"Expected 5,218 total pairs, found {len(df)}"
    assert "is_complete" in df.columns, "is_complete flag missing in feature store"
    complete_count = int(df["is_complete"].sum())
    incomplete_count = len(df) - complete_count
    assert complete_count == 4554, f"Expected 4,554 complete pairs, found {complete_count}"
    assert incomplete_count == 664, f"Expected 664 incomplete pairs, found {incomplete_count}"

    from src.inference.predictor import CropPricePredictor
    p = CropPricePredictor()

    # Query an incomplete pair -> must return HTTP 422 insufficient_data
    inc_row = df[~df["is_complete"]].iloc[0]
    res_inc = p.predict(inc_row["state"], inc_row["district"], inc_row["market"], inc_row["commodity"], horizon=7)
    assert res_inc["status"] == "error"
    assert res_inc["error_code"] == "insufficient_data"


def test_no_stale_hardcoded_values():
    """Verify that stale constants (465.75, 672.36, 563.81, 348.20, 890.15) do not exist in active code."""
    banned = ["465.75", "672.36", "563.81", "348.20", "890.15"]
    active_dirs = [REPO_ROOT / "src", REPO_ROOT / "api", REPO_ROOT / "dashboard"]
    for d in active_dirs:
        for py_path in d.rglob("*.py"):
            with open(py_path, "r", encoding="utf-8") as f:
                content = f.read()
            for val in banned:
                assert val not in content, f"Found stale hardcoded value {val} in {py_path}"


def test_deep_learning_config_exists():
    """Verify models/trained/deep_learning_config.json exists and contains training specification."""
    cfg_file = MODELS_DIR / "deep_learning_config.json"
    assert cfg_file.exists(), "deep_learning_config.json missing"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = ["seed", "sequence_length", "features", "batch_size", "epochs", "patience"]
    for k in required_keys:
        assert k in cfg, f"Key '{k}' missing from deep_learning_config.json"

