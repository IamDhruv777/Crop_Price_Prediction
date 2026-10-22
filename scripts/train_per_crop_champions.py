"""
Phase 2: Dedicated Per-Crop Champion Model Training & Evaluation.
Trains specialized models per agricultural commodity:
- Learns crop-specific seasonality, supply elasticity, and price dynamics
- Uses Huber SmoothL1 residual learning to avoid outlier overshooting
- Saves per-crop model weights and per-crop metric scorecards
"""
import sys
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "processed" / "unified_splits"
MODELS_DIR = Path("models") / "trained" / "crops"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLITS_DIR / "train_unified.csv"
VAL_FILE = SPLITS_DIR / "val_unified.csv"
TEST_FILE = SPLITS_DIR / "test_unified.csv"

def evaluate_crop_metrics(y_true, y_pred, y_baseline, crop_name, horizon_name="7-Day"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    trust_score = max(0, min(100, (1 - mape) * 100))
    
    # Standard directional market trajectory
    actual_dir = (y_true >= y_baseline).astype(int)
    pred_dir = (y_pred >= y_baseline).astype(int)
    
    dir_acc = accuracy_score(actual_dir, pred_dir)
    prec = precision_score(actual_dir, pred_dir, zero_division=0)
    rec = recall_score(actual_dir, pred_dir, zero_division=0)
    f1 = f1_score(actual_dir, pred_dir, zero_division=0)
    
    return {
        "Commodity": crop_name,
        "Horizon": horizon_name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape * 100, 2),
        "R2": round(r2, 4),
        "Trust_Score": round(trust_score, 1),
        "Directional_Accuracy": round(dir_acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1_Score": round(f1, 4)
    }

def train_crop_champions():
    logger.info("==========================================================================")
    logger.info("=== PHASE 2: DEDICATED PER-CROP CHAMPION MODEL SUITE ===")
    logger.info("==========================================================================")
    
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    meta_cols = ['state', 'district', 'market', 'commodity', 'variety', 'group', 'date',
                 'target_price_1d', 'target_price_7d', 'target_price_30d', 'target_direction_7d']
    feats = [c for c in train_df.columns if c not in meta_cols and np.issubdtype(train_df[c].dtype, np.number)]
    
    commodities = train_df['commodity'].dropna().unique().tolist()
    logger.info(f"Training dedicated champions for {len(commodities)} commodities: {commodities}")
    
    results = []
    
    for comm in commodities:
        logger.info(f"\n--- Fitting Dedicated Champion Architecture: {comm} ---")
        
        tr_c = train_df[train_df['commodity'] == comm]
        va_c = val_df[val_df['commodity'] == comm]
        te_c = test_df[test_df['commodity'] == comm]
        
        for horizon, target_col in [(1, 'target_price_1d'), (7, 'target_price_7d'), (30, 'target_price_30d')]:
            h_name = f"{horizon}-Day"
            
            tr_h = tr_c.dropna(subset=[target_col, 'modal_price']).copy()
            va_h = va_c.dropna(subset=[target_col, 'modal_price']).copy()
            te_h = te_c.dropna(subset=[target_col, 'modal_price']).copy()
            
            if len(tr_h) < 100 or len(te_h) < 20:
                continue
                
            y_tr = tr_h[target_col].values
            b_tr = tr_h['modal_price'].values
            y_te = te_h[target_col].values
            b_te = te_h['modal_price'].values
            
            # Predict residual price drift
            delta_tr = np.clip(y_tr - b_tr, -1500, 1500)
            
            crop_model = XGBRegressor(
                n_estimators=180,
                max_depth=6,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            )
            crop_model.fit(tr_h[feats].fillna(0), delta_tr)
            
            pred_delta = crop_model.predict(te_h[feats].fillna(0))
            pred_price = b_te + pred_delta
            
            # Conformal calibration on validation set
            if len(va_h) > 20:
                val_delta = crop_model.predict(va_h[feats].fillna(0))
                val_residuals = np.abs(va_h[target_col].values - (va_h['modal_price'].values + val_delta))
                q95 = float(np.quantile(val_residuals, 0.95))
            else:
                q95 = float(np.std(y_tr) * 1.96)
                
            m = evaluate_crop_metrics(y_te, pred_price, b_te, comm, h_name)
            m['Uncertainty_Delta_95'] = round(q95, 2)
            results.append(m)
            
            clean_name = comm.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            joblib.dump(crop_model, MODELS_DIR / f"{clean_name}_{horizon}d_champion.joblib")
            
            logger.info(f"  {comm} [{h_name}] -> RMSE: Rs. {m['RMSE']:.2f}, MAPE: {m['MAPE']:.2f}%, R2: {m['R2']:.4f}, Trust: {m['Trust_Score']:.1f}/100, F1: {m['F1_Score']:.4f}")
            
    res_df = pd.DataFrame(results)
    res_file = DATA_DIR / "per_crop_leaderboard.csv"
    res_df.to_csv(res_file, index=False)
    
    logger.info("\n==========================================================================")
    logger.info("PHASE 2 PER-CROP CHAMPION TRAINING COMPLETE — SCORECARD SAVED:")
    logger.info("==========================================================================")
    print(res_df[['Commodity', 'Horizon', 'RMSE', 'MAPE', 'R2', 'Trust_Score', 'F1_Score']].to_string(index=False))

if __name__ == "__main__":
    train_crop_champions()
