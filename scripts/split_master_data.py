import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_MASTER = DATA_DIR / "raw" / "bulk_mandi" / "agmarknet_full_10year_master.csv"
SPLITS_DIR = DATA_DIR / "splits"
TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "validate"
TEST_DIR = SPLITS_DIR / "test"
PROCESSED_DIR = DATA_DIR / "processed"

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def split_master():
    logger.info("Reading 397 MB master dataset...")
    df = pd.read_csv(RAW_MASTER)
    logger.info(f"Loaded {len(df):,} records.")
    
    logger.info("Parsing dates and sorting chronologically...")
    df['date'] = pd.to_datetime(df['date'], format="mixed", dayfirst=True)
    df = df.sort_values('date').dropna(subset=['modal_price'])
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info("Writing train.csv (70%)...")
    train_df.to_csv(TRAIN_DIR / "train.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    
    logger.info("Writing val.csv (15%)...")
    val_df.to_csv(VAL_DIR / "val.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    
    logger.info("Writing test.csv (15%)...")
    test_df.to_csv(TEST_DIR / "test.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    logger.info("============================================================")
    logger.info(f"📁 TRAIN SET:    {len(train_df):,} records ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logger.info(f"📁 VALIDATE SET: {len(val_df):,} records ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    logger.info(f"📁 TEST SET:     {len(test_df):,} records ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    logger.info("============================================================")

if __name__ == "__main__":
    split_master()
