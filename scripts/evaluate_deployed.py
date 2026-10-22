"""
Evaluates the Canonical Production Deployed Pipeline on Test Set Complete Support.
Generates:
  - data/deployed_test_predictions.csv
  - data/metrics_deployed.json
  - data/metrics_deployed_by_crop.csv

Production Deployed Pipeline:
  1-Day: Tuned XGBoost Regressor (anchor: modal_price, clip: [-200, 200])
  7-Day: Tuned XGBoost Regressor (anchor: 0.6*modal_price + 0.4*price_rolling_mean_7, clip: [-400, 400])
  30-Day: Naive Persistence Baseline (anchor: modal_price)
Support: Complete-feature rows only (no zero fill).
"""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "unified_splits" / "test_unified.csv"
MODELS_DIR = REPO_ROOT / "models" / "trained"
REGISTRY_PATH = MODELS_DIR / "deployment_registry.json"
MONDRIAN_PATH = REPO_ROOT / "data" / "mondrian_conformal_quantiles.json"

OUT_PREDS = REPO_ROOT / "data" / "deployed_test_predictions.csv"
OUT_METRICS_JSON = REPO_ROOT / "data" / "metrics_deployed.json"
OUT_METRICS_CROP_CSV = REPO_ROOT / "data" / "metrics_deployed_by_crop.csv"


def compute_metrics_slice(y_true, y_pred, y_anchor=None, lower=None, upper=None):
    """Computes comprehensive regression, commercial tolerance, directional, and conformal metrics."""
    res = y_true - y_pred
    mse = float(np.mean(res**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(res)))

    denom = np.where(y_true == 0, 1e-5, y_true)
    mape = float(np.mean(np.abs(res) / denom) * 100.0)
    smape = float(np.mean(2.0 * np.abs(res) / (np.abs(y_true) + np.abs(y_pred) + 1e-5)) * 100.0)

    ss_res = np.sum(res**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-5)))

    rel_acc = float(max(0.0, 100.0 - mape))
    tol_5 = float(np.mean(np.abs(res) / denom <= 0.05) * 100.0)
    tol_10 = float(np.mean(np.abs(res) / denom <= 0.10) * 100.0)
    tol_20 = float(np.mean(np.abs(res) / denom <= 0.20) * 100.0)

    # Directional metrics relative to anchor
    if y_anchor is not None:
        act_dir = (y_true - y_anchor) > 0
        pred_dir = (y_pred - y_anchor) > 0
        dir_acc = float(np.mean(act_dir == pred_dir) * 100.0)
        tp = np.sum(act_dir & pred_dir)
        fp = np.sum(~act_dir & pred_dir)
        fn = np.sum(act_dir & ~pred_dir)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    else:
        dir_acc = prec = rec = f1 = 0.0

    # Conformal coverage and interval width
    if lower is not None and upper is not None:
        in_band = (y_true >= lower) & (y_true <= upper)
        cov_95 = float(np.mean(in_band) * 100.0)
        widths = upper - lower
        mean_width = float(np.mean(widths))
        mean_width_pct = float(np.mean(widths / denom) * 100.0)
    else:
        cov_95 = mean_width = mean_width_pct = None

    return {
        "n_samples": int(len(y_true)),
        "R2": round(r2, 4),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
        "Relative_Accuracy": round(rel_acc, 2),
        "Tolerance_5pct": round(tol_5, 2),
        "Tolerance_10pct": round(tol_10, 2),
        "Tolerance_20pct": round(tol_20, 2),
        "Directional_Accuracy": round(dir_acc, 2),
        "Directional_Precision": round(prec * 100.0, 2),
        "Directional_Recall": round(rec * 100.0, 2),
        "Directional_F1": round(f1, 4),
        "Conformal_Coverage_95": round(cov_95, 2) if cov_95 is not None else None,
        "Mean_Interval_Width": round(mean_width, 2) if mean_width is not None else None,
        "Mean_Interval_Width_Pct": round(mean_width_pct, 2) if mean_width_pct is not None else None
    }


def main():
    logger.info(f"Loading test split from {TEST_SPLIT_PATH}...")
    test_df = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)

    xgb_1d = joblib.load(MODELS_DIR / "xgboost_1d.joblib")
    xgb_7d = joblib.load(MODELS_DIR / "xgboost_7d.joblib")
    feats = list(xgb_1d.feature_names_in_)

    # Filter to complete-feature support
    complete_mask = test_df[feats].notna().all(axis=1)
    logger.info(f"Complete feature test rows: {complete_mask.sum():,} / {len(test_df):,} ({complete_mask.sum()/len(test_df)*100:.1f}%)")
    df_comp = test_df[complete_mask].copy().reset_index(drop=True)

    with open(MONDRIAN_PATH, "r", encoding="utf-8") as f:
        mondrian_q = json.load(f)

    # 1. Generate deployed predictions
    # 1D
    anc1 = df_comp["modal_price"].values
    raw_delta1 = xgb_1d.predict(df_comp[feats])
    clip_delta1 = np.clip(raw_delta1, -200.0, 200.0)
    pred1 = np.maximum(10.0, np.round(anc1 + clip_delta1, 2))
    df_comp["predicted_price_1d"] = pred1

    q1_map = mondrian_q["1-Day"]
    q1_vals = df_comp["commodity"].map(q1_map).values
    df_comp["uncertainty_delta_1d"] = q1_vals
    df_comp["lower_bound_1d"] = np.maximum(0.0, np.round(pred1 - q1_vals, 2))
    df_comp["upper_bound_1d"] = np.round(pred1 + q1_vals, 2)

    # 7D
    anc7 = 0.6 * df_comp["modal_price"].values + 0.4 * df_comp["price_rolling_mean_7"].values
    raw_delta7 = xgb_7d.predict(df_comp[feats])
    clip_delta7 = np.clip(raw_delta7, -400.0, 400.0)
    pred7 = np.maximum(10.0, np.round(anc7 + clip_delta7, 2))
    df_comp["predicted_price_7d"] = pred7

    q7_map = mondrian_q["7-Day"]
    q7_vals = df_comp["commodity"].map(q7_map).values
    df_comp["uncertainty_delta_7d"] = q7_vals
    df_comp["lower_bound_7d"] = np.maximum(0.0, np.round(pred7 - q7_vals, 2))
    df_comp["upper_bound_7d"] = np.round(pred7 + q7_vals, 2)

    # 30D (Naive Persistence)
    pred30 = df_comp["modal_price"].values
    df_comp["predicted_price_30d"] = pred30

    q30_map = mondrian_q["30-Day"]
    q30_vals = df_comp["commodity"].map(q30_map).values
    df_comp["uncertainty_delta_30d"] = q30_vals
    df_comp["lower_bound_30d"] = np.maximum(0.0, np.round(pred30 - q30_vals, 2))
    df_comp["upper_bound_30d"] = np.round(pred30 + q30_vals, 2)

    # Save deployed predictions CSV
    save_cols = [
        "date", "state", "district", "market", "commodity", "modal_price",
        "price_rolling_mean_7", "price_rolling_mean_30",
        "target_price_1d", "predicted_price_1d", "lower_bound_1d", "upper_bound_1d", "uncertainty_delta_1d",
        "target_price_7d", "predicted_price_7d", "lower_bound_7d", "upper_bound_7d", "uncertainty_delta_7d",
        "target_price_30d", "predicted_price_30d", "lower_bound_30d", "upper_bound_30d", "uncertainty_delta_30d"
    ]
    df_comp[save_cols].to_csv(OUT_PREDS, index=False)
    logger.info(f"Saved deployed predictions to {OUT_PREDS}")

    # 2. Evaluate metrics
    results = {
        "_metadata": {
            "evaluation_support": "complete_features",
            "clipping": "1D: [-200, 200], 7D: [-400, 400], 30D: none",
            "models": {
                "1-Day": "Tuned XGBoost Regressor (clipped [-200, 200])",
                "7-Day": "Tuned XGBoost Regressor (clipped [-400, 400])",
                "30-Day": "Naive Persistence Baseline"
            }
        },
        "by_horizon": {},
        "macro_averages": {},
        "mondrian_averages": {},
        "by_crop": []
    }

    crop_rows = []
    horizons = [
        ("1-Day", "target_price_1d", "predicted_price_1d", "modal_price", "lower_bound_1d", "upper_bound_1d", "uncertainty_delta_1d"),
        ("7-Day", "target_price_7d", "predicted_price_7d", "modal_price", "lower_bound_7d", "upper_bound_7d", "uncertainty_delta_7d"),
        ("30-Day", "target_price_30d", "predicted_price_30d", "modal_price", "lower_bound_30d", "upper_bound_30d", "uncertainty_delta_30d")
    ]

    for h_label, act_col, pred_col, base_col, l_col, u_col, delta_col in horizons:
        sub = df_comp.dropna(subset=[act_col, pred_col]).copy()
        act = sub[act_col].values
        pred = sub[pred_col].values
        base = sub[base_col].values
        l_b = sub[l_col].values
        u_b = sub[u_col].values

        pooled = compute_metrics_slice(act, pred, base, l_b, u_b)
        results["by_horizon"][h_label] = pooled

        # Per-crop breakdown
        crops = sorted(sub["commodity"].unique())
        h_crop_metrics = []
        h_crop_cov = []
        h_crop_width_pct = []
        crops_below_95 = []

        for crop in crops:
            c_sub = sub[sub["commodity"] == crop]
            c_act = c_sub[act_col].values
            c_pred = c_sub[pred_col].values
            c_base = c_sub[base_col].values
            c_l = c_sub[l_col].values
            c_u = c_sub[u_col].values

            m_crop = compute_metrics_slice(c_act, c_pred, c_base, c_l, c_u)
            crop_rows.append({
                "Crop": crop,
                "Horizon": h_label,
                "n_samples": m_crop["n_samples"],
                "Model_R2": m_crop["R2"],
                "Model_RMSE": m_crop["RMSE"],
                "Model_MAE": m_crop["MAE"],
                "Model_MAPE": m_crop["MAPE"],
                "Model_RelAcc": m_crop["Relative_Accuracy"],
                "Model_Tol10": m_crop["Tolerance_10pct"],
                "Coverage_95": m_crop["Conformal_Coverage_95"],
                "Mean_Width": m_crop["Mean_Interval_Width"],
                "Mean_Width_Pct": m_crop["Mean_Interval_Width_Pct"]
            })
            h_crop_metrics.append(m_crop)
            if m_crop["Conformal_Coverage_95"] is not None:
                h_crop_cov.append(m_crop["Conformal_Coverage_95"])
                h_crop_width_pct.append(m_crop["Mean_Interval_Width_Pct"])
                if m_crop["Conformal_Coverage_95"] < 95.0:
                    crops_below_95.append({"crop": crop, "coverage": m_crop["Conformal_Coverage_95"]})

        # Macro averages across 11 crops
        macro = {
            "macro_R2": round(float(np.mean([m["R2"] for m in h_crop_metrics])), 4),
            "macro_RMSE": round(float(np.mean([m["RMSE"] for m in h_crop_metrics])), 2),
            "macro_MAE": round(float(np.mean([m["MAE"] for m in h_crop_metrics])), 2),
            "macro_MAPE": round(float(np.mean([m["MAPE"] for m in h_crop_metrics])), 2),
            "macro_Relative_Accuracy": round(float(np.mean([m["Relative_Accuracy"] for m in h_crop_metrics])), 2),
            "macro_Tolerance_10pct": round(float(np.mean([m["Tolerance_10pct"] for m in h_crop_metrics])), 2),
            "macro_Conformal_Coverage_95": round(float(np.mean(h_crop_cov)), 2),
            "macro_Mean_Interval_Width_Pct": round(float(np.mean(h_crop_width_pct)), 2)
        }
        results["macro_averages"][h_label] = macro
        results["mondrian_averages"][h_label] = {
            "macro_coverage": round(float(np.mean(h_crop_cov)), 2),
            "macro_interval_width_pct": round(float(np.mean(h_crop_width_pct)), 2),
            "crops_below_95": crops_below_95
        }

    results["by_crop"] = crop_rows
    crop_df = pd.DataFrame(crop_rows)
    crop_df.to_csv(OUT_METRICS_CROP_CSV, index=False)
    logger.info(f"Saved per-crop metrics to {OUT_METRICS_CROP_CSV}")

    with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved deployed metrics JSON to {OUT_METRICS_JSON}")

    # Print summary
    print("\n" + "=" * 90)
    print("CANONICAL DEPLOYED PIPELINE METRICS (TEST COMPLETE SUPPORT)")
    print("=" * 90)
    for h, m in results["by_horizon"].items():
        print(f"Horizon {h:6s} (n={m['n_samples']:,}): RMSE={m['RMSE']:.2f} | MAE={m['MAE']:.2f} | R2={m['R2']:.4f} | RelAcc={m['Relative_Accuracy']:.2f}% | Tol10={m['Tolerance_10pct']:.2f}% | Cov95={m['Conformal_Coverage_95']:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
