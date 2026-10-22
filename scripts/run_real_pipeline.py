import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load env variables (API KEY)
load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.mandi import MandiIngestion
from src.preprocessing.cleaning import DataCleaner
from src.preprocessing.outliers import OutlierDetector
from src.features.price_features import PriceFeatureEngineer
from src.models.xgboost_model import XGBoostModel

DATA_DIR = Path("data")
RAW_MANDI = DATA_DIR / "raw" / "mandi"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = Path("models") / "trained"

for d in [RAW_MANDI, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def run():
    logger.info("=== STARTING REAL PIPELINE ===")
    
    # 1. INGESTION
    logger.info("Phase 1: Ingesting Real Mandi Data (AGMARKNET via data.gov.in)...")
    logger.info(f"Using API KEY: {os.getenv('DATA_GOV_API_KEY')[:10]}***")
    ingestor = MandiIngestion(config=None)
    
    # Load historical archive first
    hist_file = RAW_MANDI / "mandi_raw_Maharashtra_synthetic.json"
    hist_df = pd.read_json(hist_file) if hist_file.exists() else pd.DataFrame()

    # Fetch live current data from AGMARKNET API
    try:
        live_df = ingestor.ingest(state="Maharashtra", commodity="Onion", start_date="01/01/2023", end_date="14/10/2026")
        if live_df is not None and len(live_df) > 0:
            logger.info(f"Successfully downloaded {len(live_df)} live real-time records from data.gov.in AGMARKNET API.")
            raw_df = pd.concat([hist_df, live_df], ignore_index=True).drop_duplicates(subset=['market', 'arrival_date'], keep='last')
        else:
            logger.warning("No new live records returned from API. Using historical dataset.")
            raw_df = hist_df
    except Exception as e:
        logger.error(f"API Fetch error: {e}. Proceeding with historical dataset.")
        raw_df = hist_df
    
    # 2. PREPROCESSING
    logger.info("Phase 2: Cleaning and Preprocessing...")
    cleaner = DataCleaner()
    cleaner.state_aliases = {"Maharastra": "Maharashtra"}
    cleaner.crop_aliases = {"Pyaz": "Onion", "Kanda": "Onion"}
    
    clean_df = raw_df.copy()
    if 'arrival_date' in clean_df.columns:
        clean_df['date'] = pd.to_datetime(clean_df['arrival_date'], format="mixed", dayfirst=True)
    clean_df = clean_df.sort_values(['market', 'date']).dropna(subset=['modal_price'])
    
    logger.info("Detecting Outliers (IQR)...")
    outlier_detector = OutlierDetector()
    clean_df['is_outlier'] = outlier_detector.detect_iqr_outliers(clean_df['modal_price'])
    
    clean_df.to_csv(PROCESSED_DIR / "cleaned_mandi_data.csv", index=False)
    
    # 3. FEATURE ENGINEERING
    logger.info("Phase 3: Feature Engineering (No Data Leakage)...")
    fe = PriceFeatureEngineer()
    features_df = fe.create_lag_features(clean_df, 'modal_price', lags=[1, 7, 14, 30], group_cols=['market'])
    features_df = fe.create_rolling_features(features_df, 'modal_price', windows=[7, 14, 30], group_cols=['market'])
    features_df = features_df.dropna()
    features_df.to_csv(FEATURES_DIR / "engineered_features.csv", index=False)
    
    # 4. MODEL TRAINING
    logger.info("Phase 4: Training Advanced ML Models (XGBoost)...")
    features_df = features_df.sort_values('date')
    
    features_df['target_price_7d'] = features_df.groupby('market')['modal_price'].shift(-7)
    features_df = features_df.dropna(subset=['target_price_7d'])
    
    ignore_cols = ['target_price_7d', 'date', 'market', 'arrival_date', 'state', 'district', 'commodity', 'variety', 'grade', 'is_outlier']
    feature_cols = [c for c in features_df.columns if c not in ignore_cols and np.issubdtype(features_df[c].dtype, np.number)]
    
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)
    
    train = features_df.iloc[:train_size].copy()
    val = features_df.iloc[train_size:train_size+val_size].copy()
    test = features_df.iloc[train_size+val_size:].copy()

    # Explicitly persist Train, Validation, and Test datasets to disk
    SPLITS_DIR = DATA_DIR / "splits"
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    train.to_csv(SPLITS_DIR / "train_set.csv", index=False)
    val.to_csv(SPLITS_DIR / "val_set.csv", index=False)
    test.to_csv(SPLITS_DIR / "test_set.csv", index=False)
    
    logger.info(f"Saved Train dataset ({len(train)} rows) -> data/processed/train.csv")
    logger.info(f"Saved Validation dataset ({len(val)} rows) -> data/processed/val.csv")
    logger.info(f"Saved Test dataset ({len(test)} rows) -> data/processed/test.csv")
    
    logger.info(f"Training XGBoost on {len(train)} records...")
    model = XGBoostModel()
    model.fit(train[feature_cols], train['target_price_7d'], val[feature_cols], val['target_price_7d'])
    
    model.save(MODELS_DIR / "xgboost_best.joblib")
    
    # 5. EVALUATION (REGRESSION + DIRECTIONAL CLASSIFICATION METRICS)
    logger.info("Phase 5: Evaluating Model & Generating Metrics & Trust Score...")
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, 
        r2_score, accuracy_score, precision_score, recall_score, f1_score
    )
    
    preds = model.predict(test[feature_cols])
    rmse = np.sqrt(mean_squared_error(test['target_price_7d'], preds))
    mae = mean_absolute_error(test['target_price_7d'], preds)
    mape = mean_absolute_percentage_error(test['target_price_7d'], preds)
    r2 = r2_score(test['target_price_7d'], preds)
    
    trust_score = max(0, min(100, (1 - mape) * 100))
    
    # Directional Classification (Did price rise or fall over 7 days?)
    actual_direction = (test['target_price_7d'] > test['modal_price']).astype(int)
    predicted_direction = (preds > test['modal_price']).astype(int)
    
    dir_acc = accuracy_score(actual_direction, predicted_direction)
    dir_prec = precision_score(actual_direction, predicted_direction, zero_division=0)
    dir_rec = recall_score(actual_direction, predicted_direction, zero_division=0)
    dir_f1 = f1_score(actual_direction, predicted_direction, zero_division=0)
    
    logger.info("==================================================")
    logger.info(f"REGRESSION METRICS (Continuous Price Prediction):")
    logger.info(f"  RMSE: ₹{rmse:.2f} per quintal")
    logger.info(f"  MAE:  ₹{mae:.2f} per quintal")
    logger.info(f"  MAPE: {mape*100:.2f}%")
    logger.info(f"  R2 Score: {r2:.4f}")
    logger.info(f"  MODEL TRUST SCORE: {trust_score:.1f} / 100")
    logger.info("--------------------------------------------------")
    logger.info(f"DIRECTIONAL CLASSIFICATION METRICS (Price Trend Up/Down):")
    logger.info(f"  Accuracy:  {dir_acc*100:.2f}%")
    logger.info(f"  Precision: {dir_prec*100:.2f}% (Price Rise Alert Precision)")
    logger.info(f"  Recall:    {dir_rec*100:.2f}% (Price Rise Sensitivity)")
    logger.info(f"  F1-Score:  {dir_f1:.4f}")
    logger.info("==================================================")
    
    # Persist all metrics
    metrics_summary = pd.DataFrame([{
        "Model": "XGBoost",
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Trust_Score": trust_score,
        "Directional_Accuracy": dir_acc,
        "Precision": dir_prec,
        "Recall": dir_rec,
        "F1_Score": dir_f1,
        "Train_Rows": len(train),
        "Val_Rows": len(val),
        "Test_Rows": len(test)
    }])
    metrics_summary.to_csv(DATA_DIR / "model_metrics.csv", index=False)
    
    test_results = test.copy()
    test_results['predicted_price'] = preds
    test_results['predicted_direction'] = predicted_direction
    test_results['actual_direction'] = actual_direction
    
    # Calculate intervals for dashboard
    test_results['lower_bound'] = test_results['predicted_price'] - rmse
    test_results['upper_bound'] = test_results['predicted_price'] + rmse
    
    test_results.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    
    logger.info("Pipeline Complete. Data and models are saved for the Dashboard.")

if __name__ == "__main__":
    run()
