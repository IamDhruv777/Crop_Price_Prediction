"""
Unified Model Evaluation Engine — Single Source of Truth for Model Metrics.
Forensic Audit Compliant:
- Evaluates Model vs. Naive Persistence Baseline across all horizons.
- Computes Pooled Overall metrics AND Equal-Crop Macro Averages.
- Computes Skill Scores (R2 Skill, MAE Improvement, MAPE Improvement, RMSE Improvement).
- Computes Directional Accuracy, Precision, Recall, and F1.
- Computes Conformal Coverage and Mean Interval Width (Absolute and % of Price).
- Outputs:
    1. data/metrics.json
    2. data/metrics_by_crop.csv
    3. data/metrics_vs_persistence.csv
    4. data/leaderboard_by_horizon.csv
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
    """Computes comprehensive metric suite for a single subset."""
    n = len(y_true)
    if n == 0:
        return {}

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 1. Regression Metrics
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
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

    # 3. Directional Trajectory (Up vs Down/Steady relative to baseline)
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
        "MSE": round(mse, 2),
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


def compute_skill_scores(m_model: dict, m_pers: dict) -> dict:
    """Computes comparative skill scores of Model relative to Naive Persistence."""
    mse_m = m_model.get("MSE", m_model["RMSE"] ** 2)
    mse_p = m_pers.get("MSE", m_pers["RMSE"] ** 2)
    r2_skill = 1.0 - (mse_m / (mse_p + 1e-5))

    mae_m = m_model["MAE"]
    mae_p = m_pers["MAE"]
    mae_imp = ((mae_p - mae_m) / (mae_p + 1e-5)) * 100.0

    mape_m = m_model["MAPE"]
    mape_p = m_pers["MAPE"]
    mape_imp = ((mape_p - mape_m) / (mape_p + 1e-5)) * 100.0

    rmse_m = m_model["RMSE"]
    rmse_p = m_pers["RMSE"]
    rmse_imp = ((rmse_p - rmse_m) / (rmse_p + 1e-5)) * 100.0

    return {
        "R2_Skill": round(float(r2_skill), 4),
        "RMSE_Improvement_Pct": round(float(rmse_imp), 2),
        "MAE_Improvement_Pct": round(float(mae_imp), 2),
        "MAPE_Improvement_Pct": round(float(mape_imp), 2),
        "Model_Beats_Persistence_RMSE": bool(rmse_m < rmse_p),
        "Model_Beats_Persistence_MAE": bool(mae_m < mae_p),
        "Model_Beats_Persistence_MAPE": bool(mape_m < mape_p)
    }


def evaluate_predictions_file(pred_file_path: Path):
    """Parse test_predictions.csv and evaluate Model vs Persistence across all horizons."""
    df = pd.read_csv(pred_file_path)
    logger.info(f"Loaded predictions: {len(df):,} rows from {pred_file_path}")

    results = {
        "by_horizon": {},
        "persistence_by_horizon": {},
        "skill_scores_by_horizon": {},
        "macro_averages": {},
        "by_crop": []
    }

    comparison_rows = []
    crop_rows = []

    horizons = []
    for h in ["1d", "7d", "30d"]:
        if f"actual_price_{h}" in df.columns and f"predicted_price_{h}" in df.columns:
            horizons.append(h)

    for h in horizons:
        h_label = f"{h.replace('d', '')}-Day"
        act_col = f"actual_price_{h}"
        pred_col = f"predicted_price_{h}"
        base_col = "modal_price"
        l_col = f"lower_bound_95_{h}" if f"lower_bound_95_{h}" in df.columns else "lower_bound_95"
        u_col = f"upper_bound_95_{h}" if f"upper_bound_95_{h}" in df.columns else "upper_bound_95"

        sub = df.dropna(subset=[act_col, pred_col, base_col]).copy()
        y_act = sub[act_col].values
        y_mod = sub[pred_col].values
        y_pers = sub[base_col].values
        l_bound = sub[l_col].values if l_col in sub.columns else None
        u_bound = sub[u_col].values if u_col in sub.columns else None

        # 1. Model Pooled Metrics
        m_mod = compute_metrics_slice(y_act, y_mod, y_pers, l_bound, u_bound)
        results["by_horizon"][h_label] = m_mod

        # 2. Persistence Pooled Metrics
        m_pers = compute_metrics_slice(y_act, y_pers, y_pers, None, None)
        results["persistence_by_horizon"][h_label] = m_pers

        # 3. Pooled Skill Scores
        skills = compute_skill_scores(m_mod, m_pers)
        results["skill_scores_by_horizon"][h_label] = skills

        # Record Comparison Row
        comparison_rows.append({
            "Horizon": h_label,
            "Target": "Pooled",
            "Model_R2": m_mod["R2"],
            "Persistence_R2": m_pers["R2"],
            "R2_Skill": skills["R2_Skill"],
            "Model_RMSE": m_mod["RMSE"],
            "Persistence_RMSE": m_pers["RMSE"],
            "RMSE_Improvement_Pct": skills["RMSE_Improvement_Pct"],
            "Model_MAE": m_mod["MAE"],
            "Persistence_MAE": m_pers["MAE"],
            "MAE_Improvement_Pct": skills["MAE_Improvement_Pct"],
            "Model_RelAcc": m_mod["Relative_Accuracy"],
            "Persistence_RelAcc": m_pers["Relative_Accuracy"],
            "Model_Tol10": m_mod["Tolerance_10pct"],
            "Persistence_Tol10": m_pers["Tolerance_10pct"],
            "Model_DirAcc": m_mod["Directional_Accuracy"],
            "Persistence_DirAcc": m_pers["Directional_Accuracy"],
            "Model_DirF1": m_mod["Directional_F1"],
            "Persistence_DirF1": m_pers["Directional_F1"],
            "Coverage_95": m_mod["Conformal_Coverage_95"],
            "Mean_Width_Pct": m_mod["Mean_Interval_Width_Pct"]
        })

        # 4. Per-Crop Breakdown & Macro Averages
        if "commodity" in sub.columns:
            h_crop_model_list = []
            h_crop_pers_list = []

            for crop in sorted(sub["commodity"].dropna().unique()):
                c_sub = sub[sub["commodity"] == crop]
                c_act = c_sub[act_col].values
                c_mod = c_sub[pred_col].values
                c_pers = c_sub[base_col].values
                c_l = c_sub[l_col].values if l_col in c_sub.columns else None
                c_u = c_sub[u_col].values if u_col in c_sub.columns else None

                c_m_mod = compute_metrics_slice(c_act, c_mod, c_pers, c_l, c_u)
                c_m_pers = compute_metrics_slice(c_act, c_pers, c_pers, None, None)
                c_skills = compute_skill_scores(c_m_mod, c_m_pers)

                crop_entry = {
                    "Crop": crop,
                    "Horizon": h_label,
                    "n_samples": len(c_sub),
                    "Model_R2": c_m_mod["R2"],
                    "Persistence_R2": c_m_pers["R2"],
                    "R2_Skill": c_skills["R2_Skill"],
                    "Model_RMSE": c_m_mod["RMSE"],
                    "Persistence_RMSE": c_m_pers["RMSE"],
                    "RMSE_Improvement_Pct": c_skills["RMSE_Improvement_Pct"],
                    "Model_MAE": c_m_mod["MAE"],
                    "Persistence_MAE": c_m_pers["MAE"],
                    "MAE_Improvement_Pct": c_skills["MAE_Improvement_Pct"],
                    "Model_RelAcc": c_m_mod["Relative_Accuracy"],
                    "Persistence_RelAcc": c_m_pers["Relative_Accuracy"],
                    "Model_Tol10": c_m_mod["Tolerance_10pct"],
                    "Persistence_Tol10": c_m_pers["Tolerance_10pct"],
                    "Coverage_95": c_m_mod["Conformal_Coverage_95"],
                    "Mean_Width": c_m_mod["Mean_Interval_Width"],
                    "Mean_Width_Pct": c_m_mod["Mean_Interval_Width_Pct"]
                }
                crop_rows.append(crop_entry)
                h_crop_model_list.append(c_m_mod)
                h_crop_pers_list.append(c_m_pers)

            # Compute Macro Averages (unweighted mean across crops)
            macro_mod = {
                "macro_R2": round(float(np.mean([m["R2"] for m in h_crop_model_list])), 4),
                "macro_RMSE": round(float(np.mean([m["RMSE"] for m in h_crop_model_list])), 2),
                "macro_MAE": round(float(np.mean([m["MAE"] for m in h_crop_model_list])), 2),
                "macro_MAPE": round(float(np.mean([m["MAPE"] for m in h_crop_model_list])), 2),
                "macro_Relative_Accuracy": round(float(np.mean([m["Relative_Accuracy"] for m in h_crop_model_list])), 2),
                "macro_Tolerance_10pct": round(float(np.mean([m["Tolerance_10pct"] for m in h_crop_model_list])), 2),
                "macro_Conformal_Coverage_95": round(float(np.mean([m["Conformal_Coverage_95"] for m in h_crop_model_list if m["Conformal_Coverage_95"] is not None])), 2),
                "macro_Mean_Interval_Width_Pct": round(float(np.mean([m["Mean_Interval_Width_Pct"] for m in h_crop_model_list if m["Mean_Interval_Width_Pct"] is not None])), 2)
            }
            macro_pers = {
                "macro_R2": round(float(np.mean([m["R2"] for m in h_crop_pers_list])), 4),
                "macro_RMSE": round(float(np.mean([m["RMSE"] for m in h_crop_pers_list])), 2),
                "macro_MAE": round(float(np.mean([m["MAE"] for m in h_crop_pers_list])), 2),
                "macro_MAPE": round(float(np.mean([m["MAPE"] for m in h_crop_pers_list])), 2),
                "macro_Relative_Accuracy": round(float(np.mean([m["Relative_Accuracy"] for m in h_crop_pers_list])), 2),
                "macro_Tolerance_10pct": round(float(np.mean([m["Tolerance_10pct"] for m in h_crop_pers_list])), 2)
            }
            macro_skills = {
                "macro_R2_Skill": round(float(1.0 - (macro_mod["macro_RMSE"]**2 / (macro_pers["macro_RMSE"]**2 + 1e-5))), 4),
                "macro_RMSE_Improvement_Pct": round(float(((macro_pers["macro_RMSE"] - macro_mod["macro_RMSE"]) / macro_pers["macro_RMSE"]) * 100.0), 2),
                "macro_MAE_Improvement_Pct": round(float(((macro_pers["macro_MAE"] - macro_mod["macro_MAE"]) / macro_pers["macro_MAE"]) * 100.0), 2)
            }

            results["macro_averages"][h_label] = {
                "model": macro_mod,
                "persistence": macro_pers,
                "skills": macro_skills
            }

            comparison_rows.append({
                "Horizon": h_label,
                "Target": "Macro Average (Equal Crop)",
                "Model_R2": macro_mod["macro_R2"],
                "Persistence_R2": macro_pers["macro_R2"],
                "R2_Skill": macro_skills["macro_R2_Skill"],
                "Model_RMSE": macro_mod["macro_RMSE"],
                "Persistence_RMSE": macro_pers["macro_RMSE"],
                "RMSE_Improvement_Pct": macro_skills["macro_RMSE_Improvement_Pct"],
                "Model_MAE": macro_mod["macro_MAE"],
                "Persistence_MAE": macro_pers["macro_MAE"],
                "MAE_Improvement_Pct": macro_skills["macro_MAE_Improvement_Pct"],
                "Model_RelAcc": macro_mod["macro_Relative_Accuracy"],
                "Persistence_RelAcc": macro_pers["macro_Relative_Accuracy"],
                "Model_Tol10": macro_mod["macro_Tolerance_10pct"],
                "Persistence_Tol10": macro_pers["macro_Tolerance_10pct"],
                "Model_DirAcc": None,
                "Persistence_DirAcc": None,
                "Model_DirF1": None,
                "Persistence_DirF1": None,
                "Coverage_95": macro_mod["macro_Conformal_Coverage_95"],
                "Mean_Width_Pct": macro_mod["macro_Mean_Interval_Width_Pct"]
            })

    results["by_crop"] = crop_rows
    crop_df = pd.DataFrame(crop_rows)
    comp_df = pd.DataFrame(comparison_rows)

    return results, crop_df, comp_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions vs Naive Persistence.")
    parser.add_argument("--predictions", type=str, default="data/test_predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--output", type=str, default="data/metrics.json", help="Path to output JSON")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error(f"Predictions file not found: {pred_path}")
        return

    results, crop_df, comp_df = evaluate_predictions_file(pred_path)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save data/metrics.json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved verified metrics to {out_json}")

    # 2. Save data/metrics_by_crop.csv
    if not crop_df.empty:
        out_crop_csv = out_json.parent / "metrics_by_crop.csv"
        crop_df.to_csv(out_crop_csv, index=False)
        logger.info(f"Saved per-crop metrics to {out_crop_csv}")

    # 3. Save data/metrics_vs_persistence.csv
    if not comp_df.empty:
        out_comp_csv = out_json.parent / "metrics_vs_persistence.csv"
        comp_df.to_csv(out_comp_csv, index=False)
        logger.info(f"Saved model vs persistence comparison to {out_comp_csv}")

    # 4. Save data/leaderboard_by_horizon.csv
    bm_file = out_json.parent / "full_architecture_benchmark_metrics.csv"
    if bm_file.exists():
        bm_df = pd.read_csv(bm_file)
        out_lb_csv = out_json.parent / "leaderboard_by_horizon.csv"
        bm_df.sort_values(by=["Horizon", "RMSE"], ascending=[True, True], inplace=True)
        bm_df.to_csv(out_lb_csv, index=False)
        logger.info(f"Saved leaderboard by horizon to {out_lb_csv}")

    # Display Honest Summary Table
    print("\n" + "=" * 95)
    print("HONEST OUT-OF-SAMPLE EVALUATION: MODEL VS. NAIVE PERSISTENCE")
    print("=" * 95)
    print(comp_df[["Horizon", "Target", "Model_R2", "Persistence_R2", "Model_RMSE", "Persistence_RMSE", "Model_MAE", "Persistence_MAE", "Model_RelAcc", "Persistence_RelAcc", "Model_Tol10", "Persistence_Tol10"]].to_string(index=False))

    print("\n" + "=" * 95)
    print("MACRO CONFORMAL COVERAGE & INTERVAL WIDTH (EQUAL CROP AVERAGE)")
    print("=" * 95)
    for h, data in results.get("macro_averages", {}).items():
        m = data["model"]
        print(f"Horizon {h}: Macro Coverage = {m['macro_Conformal_Coverage_95']:.2f}% | Macro Width = {m['macro_Mean_Interval_Width_Pct']:.2f}% of price")


if __name__ == "__main__":
    main()
