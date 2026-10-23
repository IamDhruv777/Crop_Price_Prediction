"""
Train Dedicated Directional Classifiers for 1-Day and 7-Day Horizons.
Trains XGBClassifier models on unified train split, calibrates probability thresholds
on the validation split, and saves artifacts for dual-head inference.
"""
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models" / "trained"
TRAIN_FILE = DATA_DIR / "processed" / "unified_splits" / "train_unified.csv"
VAL_FILE = DATA_DIR / "processed" / "unified_splits" / "val_unified.csv"
TEST_FILE = DATA_DIR / "processed" / "unified_splits" / "test_unified.csv"
CONFIG_FILE = MODELS_DIR / "directional_classifier_config.json"


def train_and_calibrate():
    logger.info("Loading unified training and validation splits...")
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    val_df = pd.read_csv(VAL_FILE, low_memory=False)
    test_df = pd.read_csv(TEST_FILE, low_memory=False)

    # Use exact 48 feature names from deployed regressor
    reg_1d = joblib.load(MODELS_DIR / "xgboost_1d.joblib")
    features = list(reg_1d.feature_names_in_)
    logger.info(f"Using {len(features)} features from canonical model.")

    results_config = {}

    for h, target_col in [(1, "target_price_1d"), (7, "target_price_7d")]:
        h_name = f"{h}-Day"
        logger.info(f"\n{'='*30} TRAINING DIRECTIONAL CLASSIFIER: {h_name} {'='*30}")

        # Filter valid targets and complete features
        tr_comp = train_df.dropna(subset=[target_col, "modal_price"])
        va_comp = val_df.dropna(subset=[target_col, "modal_price"])
        te_comp = test_df.dropna(subset=[target_col, "modal_price"])

        tr_sub = tr_comp[tr_comp[features].notna().all(axis=1)]
        va_sub = va_comp[va_comp[features].notna().all(axis=1)]
        te_sub = te_comp[te_comp[features].notna().all(axis=1)]

        # Sample for training efficiency if massive
        if len(tr_sub) > 150000:
            tr_sub = tr_sub.sample(n=150000, random_state=42)
        if len(va_sub) > 40000:
            va_sub = va_sub.sample(n=40000, random_state=42)

        X_tr = tr_sub[features]
        y_tr = (tr_sub[target_col].values > tr_sub["modal_price"].values).astype(int)

        X_va = va_sub[features]
        y_va = (va_sub[target_col].values > va_sub["modal_price"].values).astype(int)

        X_te = te_sub[features]
        y_te = (te_sub[target_col].values > te_sub["modal_price"].values).astype(int)

        logger.info(f"{h_name}: Train={len(y_tr):,} (Pos={y_tr.mean()*100:.1f}%), Val={len(y_va):,}, Test={len(y_te):,}")

        # Train XGBClassifier with logloss
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # Calibrate optimal probability thresholds strictly on VALIDATION split
        val_probs = clf.predict_proba(X_va)[:, 1]

        best_f1_thresh = 0.50
        best_f1_val = 0.0
        high_prec_thresh = 0.55

        threshold_sweep = np.linspace(0.30, 0.70, 41)
        for t in threshold_sweep:
            v_pred = (val_probs >= t).astype(int)
            v_p = precision_score(y_va, v_pred, zero_division=0)
            v_r = recall_score(y_va, v_pred, zero_division=0)
            v_f1 = 2 * v_p * v_r / (v_p + v_r) if (v_p + v_r) > 0 else 0.0
            if v_f1 > best_f1_val:
                best_f1_val = v_f1
                best_f1_thresh = float(round(t, 3))
            if v_p >= 0.55 and high_prec_thresh == 0.55:
                high_prec_thresh = float(round(t, 3))

        logger.info(f"Validation Calibration for {h_name}:")
        logger.info(f"  Best F1 Threshold: {best_f1_thresh} (Val F1: {best_f1_val:.4f})")
        logger.info(f"  High Precision (>=55%) Threshold: {high_prec_thresh}")

        # Out-of-sample Test Evaluation
        test_probs = clf.predict_proba(X_te)[:, 1]

        # Evaluate at default 0.50, best F1 threshold, and high-precision threshold
        test_evals = {}
        for label, t_val in [("balanced_f1", best_f1_thresh), ("standard_half", 0.50), ("high_precision", high_prec_thresh)]:
            t_pred = (test_probs >= t_val).astype(int)
            test_evals[label] = {
                "threshold": t_val,
                "precision": round(float(precision_score(y_te, t_pred, zero_division=0) * 100.0), 2),
                "recall": round(float(recall_score(y_te, t_pred, zero_division=0) * 100.0), 2),
                "f1": round(float(f1_score(y_te, t_pred, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y_te, t_pred) * 100.0), 2)
            }
            logger.info(f"Test Set ({label} @ {t_val}): Prec={test_evals[label]['precision']}%, Rec={test_evals[label]['recall']}%, F1={test_evals[label]['f1']}, Acc={test_evals[label]['accuracy']}%")

        # Save model artifact
        model_out = MODELS_DIR / f"xgb_classifier_{h}d.joblib"
        joblib.dump(clf, model_out)
        logger.info(f"Saved trained classifier to {model_out}")

        results_config[h_name] = {
            "model_file": f"models/trained/xgb_classifier_{h}d.joblib",
            "features": features,
            "best_f1_threshold": best_f1_thresh,
            "high_precision_threshold": high_prec_thresh,
            "test_metrics": test_evals
        }

    # Save configuration
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(results_config, f, indent=2)
    logger.info(f"Saved directional config to {CONFIG_FILE}")


if __name__ == "__main__":
    train_and_calibrate()
