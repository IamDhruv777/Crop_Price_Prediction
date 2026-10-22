"""
Mondrian (Crop-Stratified) Conformal Prediction Calibration.
Calibrates 95% uncertainty interval quantiles strictly on the chronological VALIDATION set (val_unified.csv).
NEVER uses holdout test targets for calibration.

Deployment Model Anchors:
  1-Day: Tuned XGBoost Regressor (anchor: modal_price)
  7-Day: Tuned XGBoost Regressor (anchor: 0.6*modal_price + 0.4*price_rolling_mean_7)
  30-Day: Naive Persistence Baseline (anchor: modal_price)
"""
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "unified_splits" / "val_unified.csv"
MODELS_DIR = REPO_ROOT / "models" / "trained"
OUTPUT_PATH = REPO_ROOT / "data" / "mondrian_conformal_quantiles.json"


def calibrate_mondrian():
    if not VAL_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found at {VAL_SPLIT_PATH}")

    logger.info(f"Loading validation partition from {VAL_SPLIT_PATH}...")
    val_df = pd.read_csv(VAL_SPLIT_PATH, low_memory=False)

    mondrian_quantiles = {
        "_metadata": {
            "source": str(VAL_SPLIT_PATH.relative_to(REPO_ROOT)),
            "method": "Split Conformal Prediction on Validation Residuals (alpha=0.05)",
            "validation_period": f"{val_df['date'].min()} to {val_df['date'].max()}",
            "sample_sizes": {},
            "deployment_models": {
                "1-Day": "Tuned XGBoost Regressor (anchor: modal_price)",
                "7-Day": "Tuned XGBoost Regressor (anchor: 0.6*modal_price + 0.4*price_rolling_mean_7)",
                "30-Day": "Naive Persistence Baseline (anchor: modal_price)"
            }
        }
    }

    # 1. Horizon 1-Day: Tuned XGBoost (anchor = modal_price)
    xgb_1d_path = MODELS_DIR / "xgboost_1d.joblib"
    if not xgb_1d_path.exists():
        raise FileNotFoundError(f"Model file {xgb_1d_path} missing")
    m1 = joblib.load(xgb_1d_path)
    v1 = val_df.dropna(subset=["target_price_1d"]).copy()
    anc1 = v1["modal_price"].values
    feats1 = m1.feature_names_in_
    pred1 = anc1 + m1.predict(v1[feats1].fillna(0))
    v1["res"] = np.abs(v1["target_price_1d"].values - pred1)
    q1 = v1.groupby("commodity")["res"].apply(lambda s: round(float(np.quantile(s, 0.95)), 2)).to_dict()
    mondrian_quantiles["1-Day"] = q1
    mondrian_quantiles["_metadata"]["sample_sizes"]["1-Day"] = len(v1)
    logger.info(f"1-Day Mondrian quantiles calibrated on {len(v1):,} validation samples across {len(q1)} crops.")

    # 2. Horizon 7-Day: Tuned XGBoost with True 7D Anchor (0.6 * modal_price + 0.4 * price_rolling_mean_7)
    xgb_7d_path = MODELS_DIR / "xgboost_7d.joblib"
    if not xgb_7d_path.exists():
        raise FileNotFoundError(f"Model file {xgb_7d_path} missing")
    m7 = joblib.load(xgb_7d_path)
    v7 = val_df.dropna(subset=["target_price_7d"]).copy()
    anc7 = 0.6 * v7["modal_price"].values + 0.4 * v7["price_rolling_mean_7"].fillna(v7["modal_price"]).values
    feats7 = m7.feature_names_in_
    pred7 = anc7 + m7.predict(v7[feats7].fillna(0))
    v7["res"] = np.abs(v7["target_price_7d"].values - pred7)
    q7 = v7.groupby("commodity")["res"].apply(lambda s: round(float(np.quantile(s, 0.95)), 2)).to_dict()
    mondrian_quantiles["7-Day"] = q7
    mondrian_quantiles["_metadata"]["sample_sizes"]["7-Day"] = len(v7)
    logger.info(f"7-Day Mondrian quantiles calibrated on {len(v7):,} validation samples across {len(q7)} crops.")

    # 3. Horizon 30-Day: Naive Persistence Baseline Deployment (anchor = modal_price)
    v30 = val_df.dropna(subset=["target_price_30d"]).copy()
    anc30 = v30["modal_price"].values
    v30["res"] = np.abs(v30["target_price_30d"].values - anc30)
    q30 = v30.groupby("commodity")["res"].apply(lambda s: round(float(np.quantile(s, 0.95)), 2)).to_dict()
    mondrian_quantiles["30-Day"] = q30
    mondrian_quantiles["_metadata"]["sample_sizes"]["30-Day"] = len(v30)
    logger.info(f"30-Day Mondrian persistence quantiles calibrated on {len(v30):,} validation samples across {len(q30)} crops.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mondrian_quantiles, f, indent=2)

    logger.info(f"Successfully saved Mondrian conformal quantiles to {OUTPUT_PATH}")

    # Display calibrated table
    df_summary = pd.DataFrame({
        "1-Day q95 (Rs)": mondrian_quantiles["1-Day"],
        "7-Day q95 (Rs)": mondrian_quantiles["7-Day"],
        "30-Day q95 (Rs)": mondrian_quantiles["30-Day"]
    })
    print("\n" + "=" * 70)
    print("CALIBRATED MONDRIAN CONFORMAL QUANTILES (VALIDATION SET ONLY)")
    print("=" * 70)
    print(df_summary.to_string())


if __name__ == "__main__":
    calibrate_mondrian()
