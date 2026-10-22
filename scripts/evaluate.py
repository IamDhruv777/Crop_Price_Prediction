"""
Unified Model Evaluation Engine — Single Source of Truth for Model Metrics.
Computes all statistical, regression, directional, commercial tolerance, and
conformal coverage metrics at runtime directly from ground truth and predictions.
Outputs data/metrics.json and data/metrics_by_crop.csv.
"""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_metrics_slice(y_true, y_pred, y_base=None, lower=None, upper=None):
    """Compute comprehensive metric suite for a single subset."""
    n = len(y_true)
    if n == 0:
        return {}

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 1. Regression Metrics
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Avoid zero division in MAPE
    denom = np.where(np.abs(y_true) < 1e-5, 1e-5, np.abs(y_true))
    pe = np.abs(y_true - y_pred) / denom
    mape = float(np.mean(pe) * 100.0)
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-5)) * 100.0)
    relative_accuracy = float(max(0.0, 100.0 - mape))

    # 2. Tolerance Accuracies
    tol_5 = float(np.mean(pe <= 0.05) * 100.0)
    tol_10 = float(np.mean(pe <= 0.10) * 100.0)
    tol_20 = float(np.mean(pe <= 0.20) * 100.0)

    # 3. Directional Trajectory (Up vs Down/Steady)
    if y_base is not None:
        y_base = np.asarray(y_base, dtype=float)
        act_dir = (y_true >= y_base).astype(int)
        pred_dir = (y_pred >= y_base).astype(int)
        dir_acc = float(accuracy_score(act_dir, pred_dir) * 100.0)
        dir_prec = float(precision_score(act_dir, pred_dir, zero_division=0) * 100.0)
        dir_rec = float(recall_score(act_dir, pred_dir, zero_division=0) * 100.0)
        dir_f1 = float(f1_score(act_dir, pred_dir, zero_division=0))
    else:
        dir_acc = dir_prec = dir_rec = dir_f1 = None

    # 4. Conformal Interval Metrics
    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        in_band = (y_true >= lower) & (y_true <= upper)
        cov = float(np.mean(in_band) * 100.0)
        width = upper - lower
        mean_width = float(np.mean(width))
        mean_width_pct = float(np.mean(width / denom) * 100.0)
    else:
        cov = mean_width = mean_width_pct = None

    return {
        "n_samples": int(n),
        "R2": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
        "Relative_Accuracy": round(relative_accuracy, 2),
        "Tolerance_5pct": round(tol_5, 2),
        "Tolerance_10pct": round(tol_10, 2),
        "Tolerance_20pct": round(tol_20, 2),
        "Directional_Accuracy": round(dir_acc, 2) if dir_acc is not None else None,
        "Directional_Precision": round(dir_prec, 2) if dir_prec is not None else None,
        "Directional_Recall": round(dir_rec, 2) if dir_rec is not None else None,
        "Directional_F1": round(dir_f1, 4) if dir_f1 is not None else None,
        "Conformal_Coverage_95": round(cov, 2) if cov is not None else None,
        "Mean_Interval_Width": round(mean_width, 2) if mean_width is not None else None,
        "Mean_Interval_Width_Pct": round(mean_width_pct, 2) if mean_width_pct is not None else None
    }


def evaluate_predictions_file(pred_file_path: Path):
    """Parse standard or multi-horizon test_predictions.csv and evaluate."""
    df = pd.read_csv(pred_file_path)
    logger.info(f"Loaded predictions: {len(df):,} rows from {pred_file_path}")

    results = {"overall": {}, "by_horizon": {}, "by_crop": {}}
    crop_rows = []

    # Detect format: multi-horizon columns or stacked format
    horizons = []
    for h in ["1d", "7d", "30d"]:
        if f"actual_price_{h}" in df.columns and f"predicted_price_{h}" in df.columns:
            horizons.append(h)

    if horizons:
        for h in horizons:
            h_label = f"{h.replace('d', '')}-Day" if "d" in h else h
            act_col = f"actual_price_{h}"
            pred_col = f"predicted_price_{h}"
            base_col = "modal_price" if "modal_price" in df.columns else None
            l_col = f"lower_bound_95_{h}" if f"lower_bound_95_{h}" in df.columns else "lower_bound_95"
            u_col = f"upper_bound_95_{h}" if f"upper_bound_95_{h}" in df.columns else "upper_bound_95"

            sub = df.dropna(subset=[act_col, pred_col])
            m = compute_metrics_slice(
                y_true=sub[act_col].values,
                y_pred=sub[pred_col].values,
                y_base=sub[base_col].values if base_col else None,
                lower=sub[l_col].values if l_col in sub.columns else None,
                upper=sub[u_col].values if u_col in sub.columns else None
            )
            results["by_horizon"][h_label] = m

            # Per crop breakdown
            if "commodity" in sub.columns:
                for crop in sorted(sub["commodity"].dropna().unique()):
                    c_sub = sub[sub["commodity"] == crop]
                    c_m = compute_metrics_slice(
                        y_true=c_sub[act_col].values,
                        y_pred=c_sub[pred_col].values,
                        y_base=c_sub[base_col].values if base_col else None,
                        lower=c_sub[l_col].values if l_col in c_sub.columns else None,
                        upper=c_sub[u_col].values if u_col in c_sub.columns else None
                    )
                    c_m["Crop"] = crop
                    c_m["Horizon"] = h_label
                    crop_rows.append(c_m)

    # Macro averages across crops for 7-Day horizon (primary benchmark)
    crop_df = pd.DataFrame(crop_rows)
    if not crop_df.empty and "Horizon" in crop_df.columns:
        results["by_crop"] = crop_df.to_dict(orient="records")
        for h_label in crop_df["Horizon"].unique():
            h_crops = crop_df[crop_df["Horizon"] == h_label]
            results[f"macro_average_{h_label}"] = {
                "macro_R2": round(float(h_crops["R2"].mean()), 4),
                "macro_RMSE": round(float(h_crops["RMSE"].mean()), 2),
                "macro_MAE": round(float(h_crops["MAE"].mean()), 2),
                "macro_Relative_Accuracy": round(float(h_crops["Relative_Accuracy"].mean()), 2),
                "macro_Tolerance_10pct": round(float(h_crops["Tolerance_10pct"].mean()), 2),
                "macro_Conformal_Coverage_95": round(float(h_crops["Conformal_Coverage_95"].mean()), 2)
            }

    return results, crop_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--predictions", type=str, default="data/test_predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--output", type=str, default="data/metrics.json", help="Path to output JSON")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error(f"Predictions file not found: {pred_path}")
        return

    results, crop_df = evaluate_predictions_file(pred_path)

    # Save metrics JSON
    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved verified metrics to {out_json}")

    # Save per-crop CSV
    if not crop_df.empty:
        out_crop_csv = out_json.parent / "metrics_by_crop.csv"
        crop_df.to_csv(out_crop_csv, index=False)
        logger.info(f"Saved per-crop metrics to {out_crop_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print("VERIFIED OUT-OF-SAMPLE EVALUATION SUMMARY (DYNAMIC RUNTIME COMPUTATION)")
    print("=" * 80)
    for h, m in results["by_horizon"].items():
        print(f"\nHorizon: {h} (n = {m['n_samples']:,})")
        print(f"  R2 Goodness of Fit              : {m['R2']:.4f}")
        print(f"  RMSE Error                      : Rs. {m['RMSE']:.2f} / qtl")
        print(f"  MAE Error                       : Rs. {m['MAE']:.2f} / qtl")
        print(f"  Relative Accuracy (100 - MAPE)  : {m['Relative_Accuracy']:.2f}%")
        print(f"  +-10% Tolerance Accuracy        : {m['Tolerance_10pct']:.2f}%")
        print(f"  Conformal 95% Band Coverage     : {m['Conformal_Coverage_95']:.2f}%")
        print(f"  Mean Band Width (% of Price)    : {m['Mean_Interval_Width_Pct']:.2f}%")
        print(f"  Directional Accuracy            : {m['Directional_Accuracy']:.2f}%")
        print(f"  Directional F1 Score            : {m['Directional_F1']:.4f}")

    if not crop_df.empty:
        print("\n" + "=" * 80)
        print("PER-CROP BREAKDOWN:")
        print("=" * 80)
        cols_show = ["Crop", "Horizon", "n_samples", "R2", "RMSE", "Relative_Accuracy", "Tolerance_10pct", "Conformal_Coverage_95"]
        print(crop_df[[c for c in cols_show if c in crop_df.columns]].to_string(index=False))

if __name__ == "__main__":
    main()
