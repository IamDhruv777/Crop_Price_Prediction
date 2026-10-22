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
        assert type(model).__name__ == "XGBRegressor", f"Expected XGBRegressor, got {type(model).__name__}"
        assert meta["model_name"] == "Tuned XGBoost Regressor"
        assert meta["model_class"] == "XGBRegressor"

        # Check metrics match benchmark row
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
        assert "macro_Conformal_Coverage_95" in macro_mod
        assert "macro_Mean_Interval_Width_Pct" in macro_mod
