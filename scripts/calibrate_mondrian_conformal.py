"""
Mondrian (Crop-Stratified) Conformal Prediction Calibration.
Calibrates 95% uncertainty interval quantiles strictly on the chronological VALIDATION set (val_unified.csv)
using the exact production deployment support (complete features) and residual clipping.
NEVER uses holdout test targets for calibration.

Canonical Deployment Pipeline:
  1-Day: Tuned XGBoost Regressor (anchor: modal_price, clip: [-200, 200])
  7-Day: Tuned XGBoost Regressor (anchor: 0.6*modal_price + 0.4*price_rolling_mean_7, clip: [-400, 400])
  30-Day: Naive Persistence Baseline (anchor: modal_price, no clipping)
Support: complete-feature rows (100% of 48 features non-null).
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

    xgb_1d_path = MODELS_DIR / "xgboost_1d.joblib"
    xgb_7d_path = MODELS_DIR / "xgboost_7d.joblib"
    if not xgb_1d_path.exists():
        raise FileNotFoundError(f"Model file {xgb_1d_path} missing")
    if not xgb_7d_path.exists():
        raise FileNotFoundError(f"Model file {xgb_7d_path} missing")

    m1 = joblib.load(xgb_1d_path)
    m7 = joblib.load(xgb_7d_path)
    feats = list(m1.feature_names_in_)

    # Filter strictly to complete-feature rows (production serving contract)
    complete_mask = val_df[feats].notna().all(axis=1)
    logger.info(f"Complete feature validation rows: {complete_mask.sum():,} / {len(val_df):,} ({complete_mask.sum()/len(val_df)*100:.1f}%)")
    val_comp = val_df[complete_mask].copy()

    mondrian_quantiles = {
        "_metadata": {
            "source": str(VAL_SPLIT_PATH.relative_to(REPO_ROOT)),
            "method": "Split Conformal Prediction on Validation Residuals (alpha=0.05)",
            "support": "complete_features",
            "clipped": True,
            "validation_period": f"{val_comp['date'].min()} to {val_comp['date'].max()}",
            "sample_sizes": {},
            "crop_sample_sizes": {},
            "deployment_models": {
                "1-Day": "Tuned XGBoost Regressor (anchor: modal_price, clip: [-200, 200])",
                "7-Day": "Tuned XGBoost Regressor (anchor: 0.6*modal_price + 0.4*price_rolling_mean_7, clip: [-400, 400])",
                "30-Day": "Naive Persistence Baseline (anchor: modal_price)"
            }
        }
    }

    # 1. Horizon 1-Day: Tuned XGBoost with [-200, 200] clip
    v1 = val_comp.dropna(subset=["target_price_1d"]).copy()
    anc1 = v1["modal_price"].values
    raw_delta1 = m1.predict(v1[feats])
    clipped_delta1 = np.clip(raw_delta1, -200.0, 200.0)
    pred1 = anc1 + clipped_delta1
    v1["res"] = np.abs(v1["target_price_1d"].values - pred1)
    q1 = v1.groupby("commodity")["res"].apply(lambda s: round(float(np.quantile(s, 0.95)), 2)).to_dict()
    mondrian_quantiles["1-Day"] = q1
    mondrian_quantiles["_metadata"]["sample_sizes"]["1-Day"] = len(v1)
    mondrian_quantiles["_metadata"]["crop_sample_sizes"]["1-Day"] = v1["commodity"].value_counts().to_dict()
    logger.info(f"1-Day Mondrian quantiles calibrated on {len(v1):,} complete validation samples across {len(q1)} crops.")

    # 2. Horizon 7-Day: Tuned XGBoost with 7D anchor and [-400, 400] clip
    v7 = val_comp.dropna(subset=["target_price_7d"]).copy()
    anc7 = 0.6 * v7["modal_price"].values + 0.4 * v7["price_rolling_mean_7"].values
    raw_delta7 = m7.predict(v7[feats])
    clipped_delta7 = np.clip(raw_delta7, -400.0, 400.0)
    pred7 = anc7 + clipped_delta7
    v7["res"] = np.abs(v7["target_price_7d"].values - pred7)
    q7 = v7.groupby("commodity")["res"].apply(lambda s: round(float(np.quantile(s, 0.95)), 2)).to_dict()
    mondrian_quantiles["7-Day"] = q7
    mondrian_quantiles["_metadata"]["sample_sizes"]["7-Day"] = len(v7)
    mondrian_quantiles["_metadata"]["crop_sample_sizes"]["7-Day"] = v7["commodity"].value_counts().to_dict()
    logger.info(f"7-Day Mondrian quantiles calibrated on {len(v7):,} complete validation samples across {len(q7)} crops.")

    # 3. Horizon 30-Day: Naive Persistence Baseline (flat anchor: modal_price)
    v30 = val_comp.dropna(subset=["target_price_30d"]).copy()
    anc30 = v30["modal_price"].values
    v30["res"] = np.abs(v30["target_price_30d"].values - anc30)
    q30 = v30.groupby("commodity")["res"].apply(lambda s: round(float(np.quantile(s, 0.95)), 2)).to_dict()
    mondrian_quantiles["30-Day"] = q30
    mondrian_quantiles["_metadata"]["sample_sizes"]["30-Day"] = len(v30)
    mondrian_quantiles["_metadata"]["crop_sample_sizes"]["30-Day"] = v30["commodity"].value_counts().to_dict()
    logger.info(f"30-Day Mondrian persistence quantiles calibrated on {len(v30):,} complete validation samples across {len(q30)} crops.")

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
    print("CALIBRATED MONDRIAN CONFORMAL QUANTILES (COMPLETE VALIDATION SUPPORT)")
    print("=" * 70)
    print(df_summary.to_string())
    print("=" * 70)


if __name__ == "__main__":
    calibrate_mondrian()
