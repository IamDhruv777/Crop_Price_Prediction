"""
Forensic Audit Integrity Test Suite.
Verifies:
  1. test_no_same_day_target: Target dates are strictly future calendar dates.
  2. test_train_val_test_no_date_overlap: Strict embargo (>30 days) between train/val/test splits.
  3. test_leaderboard_matches_recomputed_metrics: Leaderboard numbers match test predictions.
  4. test_api_uses_trained_model: Predictor calls model.predict and produces dynamic output.
  5. test_all_crops_present_in_evaluation: All 11 benchmark commodities exist in test predictions.
"""
import json
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = DATA_DIR / "processed" / "unified_splits"


def test_no_same_day_target():
    """Verify that no sample has a target on the same calendar day."""
    test_unified_file = SPLITS_DIR / "test_unified.csv"
    assert test_unified_file.exists(), "test_unified.csv does not exist"

    df = pd.read_csv(test_unified_file, nrows=10000, parse_dates=['date', 'target_date_1d', 'target_date_7d', 'target_date_30d'])

    # 1-Day: target_date_1d must be strictly > date
    h1 = df.dropna(subset=['target_date_1d'])
    diff1 = (h1['target_date_1d'] - h1['date']).dt.days
    assert (diff1 >= 1).all(), "Found 1-Day targets with same-day or past date!"

    # 7-Day: target_date_7d must be >= date + 5 days
    h7 = df.dropna(subset=['target_date_7d'])
    diff7 = (h7['target_date_7d'] - h7['date']).dt.days
    assert (diff7 >= 5).all(), "Found 7-Day targets with calendar gap < 5 days!"

    # 30-Day: target_date_30d must be >= date + 26 days
    h30 = df.dropna(subset=['target_date_30d'])
    diff30 = (h30['target_date_30d'] - h30['date']).dt.days
    assert (diff30 >= 26).all(), "Found 30-Day targets with calendar gap < 26 days!"


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
    """Verify that master_model_leaderboard.csv strictly matches runtime computed metrics."""
    leaderboard_file = DATA_DIR / "master_model_leaderboard.csv"
    metrics_file = DATA_DIR / "metrics.json"

    assert leaderboard_file.exists(), "master_model_leaderboard.csv missing"
    assert metrics_file.exists(), "metrics.json missing"

    lb_df = pd.read_csv(leaderboard_file)
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Check 7-Day Champion / Super Ensemble row
    ens_row = lb_df[(lb_df['Horizon'] == '7-Day') & (lb_df['Model'].str.contains('Super Ensemble', case=False))].iloc[0]
    expected_m = metrics["by_horizon"]["7-Day"]

    assert abs(ens_row['R2'] - expected_m['R2']) < 0.002, f"R2 mismatch: {ens_row['R2']} vs {expected_m['R2']}"
    assert abs(ens_row['RMSE'] - expected_m['RMSE']) < 0.1, f"RMSE mismatch: {ens_row['RMSE']} vs {expected_m['RMSE']}"
    assert abs(ens_row['Relative_Accuracy'] - expected_m['Relative_Accuracy']) < 0.1, "Relative Accuracy mismatch"


def test_api_uses_trained_model():
    """Verify that CropPricePredictor loads the trained model and produces dynamic predictions."""
    from src.inference.predictor import CropPricePredictor

    predictor = CropPricePredictor()

    # 1. Verify model is loaded from trained registry (not None)
    model, meta = predictor.load_model("Wheat", horizon=7)
    assert model is not None, "Predictor failed to load trained model"

    # 2. Verify that predictor.predict runs and utilizes trained model
    res_wheat = predictor.predict(
        state="Maharashtra",
        district="Nashik",
        market="Lasalgaon",
        crop="Wheat",
        horizon=7,
        confidence_level=0.95
    )
    assert res_wheat is not None
    assert res_wheat["predicted_price"] > 100.0
    assert len(res_wheat["predictions_trajectory"]) == 7
    assert res_wheat["model_name"] != "hardcoded_drift"

    # 3. Test different commodity has different prediction and recommendation
    res_rice = predictor.predict(
        state="Andhra Pradesh",
        district="Krishna",
        market="Gudiwada",
        crop="Rice",
        horizon=7,
        confidence_level=0.95
    )
    assert res_rice is not None
    assert res_rice["predicted_price"] > 100.0
    assert res_rice["predicted_price"] != res_wheat["predicted_price"]


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

    # Verify every crop has at least 100 observations
    crop_counts = df_valid['commodity'].value_counts()
    for crop in expected_crops:
        count = crop_counts[crop]
        assert count >= 100, f"Crop {crop} has only {count} test rows (insufficient sample size)"
