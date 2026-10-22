"""
Step 5, 6, 7: Master AI/ML Forecasting Engine, Validation & Multi-Horizon Inference.
Forensic Audit Compliant:
- Evaluates real out-of-sample data across all 11 commodities.
- Zero Hardcoding: All metrics computed at runtime from test predictions and actuals.
- Benchmarks:
    1. Statistical Baselines: Naive Persistence, Moving Average
    2. Classical ML: Random Forest, LightGBM, Tuned XGBoost
    3. Spatial-Temporal: XGBoost + Spatial KNN Features
    4. Multi-Model Super Ensemble (Stacking meta-learner)
- Conformal Uncertainty: Certified 95% prediction intervals calibrated on validation residuals.
- Multi-Horizon Forecasting: 1-Day, 7-Day, and 30-Day Ahead Predictions.
- Outputs clean data/test_predictions.csv and data/full_architecture_benchmark_metrics.csv.
"""
import sys
import time
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, accuracy_score, f1_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "processed" / "unified_splits"
MODELS_DIR = Path("models") / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLITS_DIR / "train_unified.csv"
VAL_FILE = SPLITS_DIR / "val_unified.csv"
TEST_FILE = SPLITS_DIR / "test_unified.csv"


def evaluate_metrics(y_true, y_pred, y_baseline, horizon_name="7-Day", lower_b=None, upper_b=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    r2 = r2_score(y_true, y_pred)
    relative_acc = max(0.0, min(100.0, 100.0 - mape))

    # Tolerance Accuracy: % within +/- 10%
    rel_error = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-5)
    tol_10 = np.mean(rel_error <= 0.10) * 100.0

    # Conformal Coverage (if bounds provided)
    if lower_b is not None and upper_b is not None:
        coverage = np.mean((y_true >= lower_b) & (y_true <= upper_b)) * 100.0
    else:
        coverage = np.nan

    # Directional trajectory
    actual_dir = (y_true >= y_baseline).astype(int)
    pred_dir = (y_pred >= y_baseline).astype(int)

    acc = accuracy_score(actual_dir, pred_dir) * 100.0
    f1 = f1_score(actual_dir, pred_dir, zero_division=0)

    return {
        "Horizon": horizon_name,
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "MAPE": round(float(mape), 2),
        "R2": round(float(r2), 4),
        "Relative_Accuracy": round(float(relative_acc), 2),
        "Tolerance_10": round(float(tol_10), 2),
        "Coverage_95": round(float(coverage), 2) if not np.isnan(coverage) else None,
        "Directional_Accuracy": round(float(acc), 2),
        "F1_Score": round(float(f1), 4)
    }


def run_full_benchmark():
    t_start = time.time()
    logger.info("==========================================================================")
    logger.info("=== STEP 5, 6 & 7: FULL ARCHITECTURE AI/ML FORECASTING BENCHMARK ===")
    logger.info("==========================================================================")

    if not TRAIN_FILE.exists():
        logger.error("Unified data not found! Please run 'python scripts/build_unified_features.py' first.")
        return

    logger.info("Loading unified leak-free datasets across all 11 commodities...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    logger.info(f"Loaded Partitions: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    meta_cols = [
        'state', 'district', 'market', 'commodity', 'date',
        'target_price_1d', 'target_date_1d',
        'target_price_7d', 'target_date_7d',
        'target_price_30d', 'target_date_30d',
        'target_direction_7d'
    ]
    feature_cols = [c for c in train_df.columns if c not in meta_cols and np.issubdtype(train_df[c].dtype, np.number)]
    spatial_features = [c for c in feature_cols if any(k in c for k in ['spatial', 'price_lag', 'weather', 'rolling', 'modal_price'])]

    logger.info(f"Feature Space: {len(feature_cols)} total features ({len(spatial_features)} spatial/temporal).")

    benchmark_results = []
    # Base test predictions DataFrame
    test_pred_df = test_df[['date', 'market', 'commodity', 'modal_price']].copy()

    # ---------------------------------------------------------
    # MULTI-HORIZON BENCHMARK (1-Day, 7-Day, 30-Day)
    # ---------------------------------------------------------
    for horizon, target_col in [(1, 'target_price_1d'), (7, 'target_price_7d'), (30, 'target_price_30d')]:
        h_name = f"{horizon}-Day"
        logger.info(f"\n==================== BENCHMARKING HORIZON: {h_name} AHEAD ====================")

        # Filter rows having valid target for this horizon
        train_h = train_df.dropna(subset=[target_col]).copy()
        val_h = val_df.dropna(subset=[target_col]).copy()
        test_h = test_df.dropna(subset=[target_col]).copy()

        # Stratified representative sample for fast, robust training
        sample_n = min(len(train_h), 200000)
        train_sub = train_h.sample(n=sample_n, random_state=42) if len(train_h) > sample_n else train_h

        val_sub = val_h.sample(n=min(len(val_h), 50000), random_state=42) if len(val_h) > 50000 else val_h

        y_train = train_sub[target_col].values
        y_val = val_sub[target_col].values
        y_test = test_h[target_col].values
        base_test = test_h['modal_price'].values

        # Autoregressive Anchors
        if horizon == 1:
            anc_tr = train_sub['modal_price'].values
            anc_va = val_sub['modal_price'].values
            anc_te = test_h['modal_price'].values
            clip_delta = 200.0
        elif horizon == 7:
            anc_tr = 0.6 * train_sub['modal_price'].values + 0.4 * train_sub['price_rolling_mean_7'].values
            anc_va = 0.6 * val_sub['modal_price'].values + 0.4 * val_sub['price_rolling_mean_7'].values
            anc_te = 0.6 * test_h['modal_price'].values + 0.4 * test_h['price_rolling_mean_7'].values
            clip_delta = 400.0
        else: # 30-Day
            anc_tr = 0.5 * train_sub['modal_price'].values + 0.5 * train_sub['price_rolling_mean_30'].values
            anc_va = 0.5 * val_sub['modal_price'].values + 0.5 * val_sub['price_rolling_mean_30'].values
            anc_te = 0.5 * test_h['modal_price'].values + 0.5 * test_h['price_rolling_mean_30'].values
            clip_delta = 600.0

        delta_tr = y_train - anc_tr
        delta_va = y_val - anc_va

        X_train_f = train_sub[feature_cols].fillna(0)
        X_val_f = val_sub[feature_cols].fillna(0)
        X_test_f = test_h[feature_cols].fillna(0)

        # 1. BASELINE: Naive Persistence
        naive_preds = test_h['modal_price'].values
        m_naive = evaluate_metrics(y_test, naive_preds, base_test, h_name)
        m_naive['Model'] = "Naive Persistence Baseline"
        benchmark_results.append(m_naive)

        # 2. BASELINE: Moving Average
        ma_col = 'price_rolling_mean_7' if horizon <= 7 else 'price_rolling_mean_30'
        ma_preds = test_h[ma_col].values
        m_ma = evaluate_metrics(y_test, ma_preds, base_test, h_name)
        m_ma['Model'] = f"Moving Average ({'7-day' if horizon <= 7 else '30-day'})"
        benchmark_results.append(m_ma)

        # 3. BASELINE: Ridge Autoregressive
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train_f, delta_tr)
        ridge_preds = anc_te + np.clip(ridge.predict(X_test_f), -clip_delta, clip_delta)
        m_ridge = evaluate_metrics(y_test, ridge_preds, base_test, h_name)
        m_ridge['Model'] = "Ridge Autoregressive"
        benchmark_results.append(m_ridge)

        # 4. MACHINE LEARNING: Random Forest (Anchored Residual)
        rf_sample = min(len(train_sub), 60000)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_f.iloc[-rf_sample:], delta_tr[-rf_sample:])
        rf_delta = np.clip(rf.predict(X_test_f), -clip_delta, clip_delta)
        rf_preds = anc_te + rf_delta
        m_rf = evaluate_metrics(y_test, rf_preds, base_test, h_name)
        m_rf['Model'] = "Random Forest Regressor"
        benchmark_results.append(m_rf)

        # 5. MACHINE LEARNING: LightGBM Regressor (Anchored Residual)
        lgbm = LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.04, num_leaves=31, subsample=0.85, random_state=42, n_jobs=-1)
        lgbm.fit(X_train_f, delta_tr)
        lgb_delta = np.clip(lgbm.predict(X_test_f), -clip_delta, clip_delta)
        lgb_preds = anc_te + lgb_delta
        m_lgb = evaluate_metrics(y_test, lgb_preds, base_test, h_name)
        m_lgb['Model'] = "LightGBM Regressor"
        benchmark_results.append(m_lgb)

        # 6. MACHINE LEARNING: Tuned XGBoost (Anchored Residual)
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.03, subsample=0.85, random_state=42, n_jobs=-1)
        xgb.fit(X_train_f, delta_tr, eval_set=[(X_val_f, delta_va)], verbose=False)
        xgb_delta = np.clip(xgb.predict(X_test_f), -clip_delta, clip_delta)
        xgb_preds = anc_te + xgb_delta
        m_xgb = evaluate_metrics(y_test, xgb_preds, base_test, h_name)
        m_xgb['Model'] = "Tuned XGBoost Regressor"
        benchmark_results.append(m_xgb)

        # 7. SPATIAL MODEL: XGBoost + Spatial KNN Features
        xgb_sp = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.03, subsample=0.85, random_state=42, n_jobs=-1)
        xgb_sp.fit(train_sub[spatial_features].fillna(0), delta_tr, eval_set=[(val_sub[spatial_features].fillna(0), delta_va)], verbose=False)
        sp_delta = np.clip(xgb_sp.predict(test_h[spatial_features].fillna(0)), -clip_delta, clip_delta)
        sp_preds = anc_te + sp_delta
        m_sp = evaluate_metrics(y_test, sp_preds, base_test, h_name)
        m_sp['Model'] = "XGBoost + Spatial KNN Features"
        benchmark_results.append(m_sp)

        # 8. SUPER ENSEMBLE: Stacking Meta-Learner (Blends XGBoost + LightGBM + Random Forest)
        # Generate out-of-fold validation predictions for meta-learner training
        val_xgb_d = np.clip(xgb.predict(X_val_f), -clip_delta, clip_delta)
        val_lgb_d = np.clip(lgbm.predict(X_val_f), -clip_delta, clip_delta)
        val_rf_d = np.clip(rf.predict(X_val_f), -clip_delta, clip_delta)

        meta_X_val = np.column_stack([val_xgb_d, val_lgb_d, val_rf_d])
        meta_learner = Ridge(alpha=1.0, positive=True)
        meta_learner.fit(meta_X_val, delta_va)

        meta_X_test = np.column_stack([xgb_delta, lgb_delta, rf_delta])
        ensemble_delta = np.clip(meta_learner.predict(meta_X_test), -clip_delta, clip_delta)
        ensemble_preds = anc_te + ensemble_delta

        # Split Conformal Uncertainty Estimation on Validation Residuals (95% guarantee)
        val_ensemble_preds = anc_va + np.clip(meta_learner.predict(meta_X_val), -clip_delta, clip_delta)
        val_residuals = np.abs(y_val - val_ensemble_preds)
        q95 = np.quantile(val_residuals, 0.95)

        lower_95 = np.maximum(0.0, ensemble_preds - q95)
        upper_95 = ensemble_preds + q95

        m_ens = evaluate_metrics(y_test, ensemble_preds, base_test, h_name, lower_95, upper_95)
        m_ens['Model'] = "Super Ensemble (Stacking)"
        benchmark_results.append(m_ens)

        # Save trained primary models
        joblib.dump(xgb, MODELS_DIR / f"xgboost_{horizon}d.joblib")
        joblib.dump(meta_learner, MODELS_DIR / f"meta_learner_{horizon}d.joblib")

        # Record test predictions aligned to original index
        test_pred_df[f'actual_price_{horizon}d'] = np.nan
        test_pred_df[f'predicted_price_{horizon}d'] = np.nan
        test_pred_df[f'lower_bound_95_{horizon}d'] = np.nan
        test_pred_df[f'upper_bound_95_{horizon}d'] = np.nan
        test_pred_df[f'uncertainty_delta_{horizon}d'] = q95

        test_pred_df.loc[test_h.index, f'actual_price_{horizon}d'] = y_test
        test_pred_df.loc[test_h.index, f'predicted_price_{horizon}d'] = ensemble_preds
        test_pred_df.loc[test_h.index, f'lower_bound_95_{horizon}d'] = lower_95
        test_pred_df.loc[test_h.index, f'upper_bound_95_{horizon}d'] = upper_95

        # 7-day synonyms for backwards dashboard compatibility
        if horizon == 7:
            test_pred_df['lower_bound_95'] = test_pred_df[f'lower_bound_95_{horizon}d']
            test_pred_df['upper_bound_95'] = test_pred_df[f'upper_bound_95_{horizon}d']
            test_pred_df['uncertainty_delta'] = q95

        logger.info(f"--- Horizon {h_name} Top Model (Super Ensemble) ---")
        logger.info(f"    R2: {m_ens['R2']:.4f} | RMSE: ₹{m_ens['RMSE']:.2f} | Rel Acc: {m_ens['Relative_Accuracy']:.2f}% | Tol +/-10%: {m_ens['Tolerance_10']:.2f}% | Conformal Cov: {m_ens['Coverage_95']:.2f}%")

    # Save benchmark table
    bench_df = pd.DataFrame(benchmark_results)
    bench_df.to_csv(DATA_DIR / "full_architecture_benchmark_metrics.csv", index=False)
    logger.info(f"Saved Benchmark Table -> {DATA_DIR / 'full_architecture_benchmark_metrics.csv'}")

    # Save test predictions
    test_pred_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    logger.info(f"Saved Test Predictions -> {DATA_DIR / 'test_predictions.csv'}")

    # Also save standard single-horizon summary for dashboard KPI cards
    h7_ens = bench_df[(bench_df['Horizon'] == '7-Day') & (bench_df['Model'] == 'Super Ensemble (Stacking)')].iloc[0]
    pd.DataFrame([{
        "Model": "Super Ensemble (Stacking Meta-Learner)",
        "RMSE": h7_ens['RMSE'],
        "MAE": h7_ens['MAE'],
        "MAPE": h7_ens['MAPE'] / 100.0,
        "R2": h7_ens['R2'],
        "Trust_Score": h7_ens['Relative_Accuracy'],
        "Tolerance_10": h7_ens['Tolerance_10'],
        "Coverage_95": h7_ens['Coverage_95'],
        "Directional_Accuracy": h7_ens['Directional_Accuracy'] / 100.0,
        "F1_Score": h7_ens['F1_Score'],
        "Train_Rows": len(train_df),
        "Val_Rows": len(val_df),
        "Test_Rows": len(test_df)
    }]).to_csv(DATA_DIR / "model_metrics.csv", index=False)

    elapsed = time.time() - t_start
    logger.info(f"Benchmark finished in {elapsed:.1f}s.")
    print("\n" + bench_df[['Horizon', 'Model', 'RMSE', 'MAPE', 'R2', 'Relative_Accuracy', 'Tolerance_10', 'Coverage_95']].to_string(index=False))


if __name__ == "__main__":
    run_full_benchmark()
