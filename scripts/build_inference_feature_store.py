"""
Builds data/processed/inference_feature_store.csv.
Extracts the latest available feature vector for every (market, commodity) pair
across the dataset splits, preserving all 48 engineered features without data leakage.

Supported Pairs:
  - Total Unique Pairs: 5,218
  - Complete Pairs (All 48 features non-null): 4,554 (87.3%)
  - Incomplete Pairs (<30 trading days lag history): 664 (12.7%) -> returns 422 insufficient_data
"""
from pathlib import Path
import pandas as pd
import logging
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "unified_splits"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "inference_feature_store.csv"
MODEL_PATH = REPO_ROOT / "models" / "trained" / "xgboost_1d.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_cols = list(model.feature_names_in_)
    logger.info(f"Loaded {len(feature_cols)} feature column definitions from {MODEL_PATH}")

    meta_cols = ["market", "commodity", "state", "district", "date"]
    all_needed_cols = meta_cols + feature_cols

    dfs = []
    # Load splits in order: train, val, test so later dates override earlier ones
    for split in ["train", "val", "test"]:
        split_path = SPLITS_DIR / f"{split}_unified.csv"
        if split_path.exists():
            logger.info(f"Loading {split} split from {split_path}...")
            df = pd.read_csv(split_path, usecols=lambda c: c in all_needed_cols, low_memory=False)
            df["_split_order"] = 1 if split == "train" else (2 if split == "val" else 3)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No unified split files found!")

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    logger.info(f"Sorting {len(combined):,} total observations to select latest per market-crop...")
    combined.sort_values(by=["date", "_split_order"], ascending=[True, True], inplace=True)

    # Group by market and commodity and pick the latest available record
    feature_store = combined.groupby(["market", "commodity"], as_index=False).last()
    feature_store.drop(columns=["_split_order"], inplace=True, errors="ignore")

    # Tag feature completeness (must have all 48 model features present and non-null)
    feature_store["is_complete"] = ~feature_store[feature_cols].isnull().any(axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_csv(OUTPUT_PATH, index=False)

    n_total = len(feature_store)
    n_complete = int(feature_store["is_complete"].sum())
    n_incomplete = n_total - n_complete

    logger.info(f"Successfully saved inference feature store with {n_total:,} pairs ({n_complete:,} complete) to {OUTPUT_PATH}")

    # Log summary
    print("\n" + "=" * 80)
    print("INFERENCE FEATURE STORE CREATION SUMMARY")
    print("=" * 80)
    print(f"Total Unique Market-Crop Pairs : {n_total:,}")
    print(f"Complete Supported Pairs (100%) : {n_complete:,} ({n_complete/n_total*100:.1f}%)")
    print(f"Incomplete Pairs (<30d history) : {n_incomplete:,} ({n_incomplete/n_total*100:.1f}%) -> returns insufficient_data")
    print(f"Total Commodities               : {feature_store['commodity'].nunique()}")
    print(f"Total Markets                   : {feature_store['market'].nunique()}")
    print(f"Historical Observation Range    : {feature_store['date'].min().strftime('%Y-%m-%d')} to {feature_store['date'].max().strftime('%Y-%m-%d')}")
    print(f"Columns                         : {len(feature_store.columns)}")


if __name__ == "__main__":
    main()
