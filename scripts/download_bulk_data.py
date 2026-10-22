"""
Full AGMARKNET 10-Year, 12-Commodity Mega-Dataset Downloader & Splitter.
Downloads millions of rows across all 12 major Indian crops from 2008 to 2017.
Constructs strict Train, Validate, and Test splits with zero data leakage.
"""
import os
import io
import requests
import urllib.parse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "bulk_mandi"
SPLITS_DIR = DATA_DIR / "splits"

TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "validate"
TEST_DIR = SPLITS_DIR / "test"

for d in [RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CROPS = [
    'Rice',
    'Wheat',
    'Soyabean',
    'Maize',
    'Arhar (Tur-Red Gram)',
    'Green Gram (Moong)',
    'Black Gram (Urd Beans)',
    'Lentil (Masur)',
    'Kabuli Chana (Chickpeas-White)',
    'White Peas',
    'Green Peas',
    'Cowpea (Lobia-Karamani)'
]

YEARS = list(range(2008, 2018)) # 2008 to 2017

def download_full_archive():
    cols = ['state', 'district', 'market', 'variety', 'group', 'arrivals_tonnes', 'min_price', 'max_price', 'modal_price', 'date']
    all_frames = []
    total_downloaded_bytes = 0
    
    logger.info("=== STARTING FULL AGMARKNET 10-YEAR MEGA-DATASET DOWNLOAD ===")
    logger.info(f"Targeting {len(CROPS)} commodities across {len(YEARS)} years (2008-2017)...")
    
    for crop in CROPS:
        encoded_crop = urllib.parse.quote(crop)
        logger.info(f"\n--- Downloading Commodity: {crop} ---")
        
        for year in YEARS:
            url = f"https://raw.githubusercontent.com/iancovert/Agmarknet/master/{encoded_crop}/{year}.csv"
            clean_name = crop.split(' ')[0].replace('(', '').lower()
            target_csv = RAW_DIR / f"{clean_name}_{year}.csv"
            
            # If already downloaded, read from disk
            if target_csv.exists() and target_csv.stat().st_size > 1000:
                logger.info(f"  [Cached] {crop} ({year}) -> {target_csv.stat().st_size / (1024*1024):.2f} MB")
                df = pd.read_csv(target_csv)
                all_frames.append(df)
                total_downloaded_bytes += target_csv.stat().st_size
                continue
                
            try:
                resp = requests.get(url, timeout=45)
                if resp.status_code == 200:
                    df = pd.read_csv(io.StringIO(resp.text), header=None, names=cols)
                    df['commodity'] = crop
                    df.to_csv(target_csv, index=False)
                    all_frames.append(df)
                    size = target_csv.stat().st_size
                    total_downloaded_bytes += size
                    logger.info(f"  [Downloaded] {crop} ({year}): {len(df):,} records ({size / (1024*1024):.2f} MB)")
                elif resp.status_code == 404:
                    logger.info(f"  [Skipped] {crop} ({year}): not available in archive")
                else:
                    logger.warning(f"  [Warning] {crop} ({year}): HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"  [Error] {crop} ({year}): {e}")

    if not all_frames:
        logger.error("No data collected!")
        return
        
    logger.info("Concatenating master mega-dataset in memory...")
    master_df = pd.concat(all_frames, ignore_index=True)
    master_csv = RAW_DIR / "agmarknet_full_10year_master.csv"
    master_df.to_csv(master_csv, index=False)
    
    total_mb = master_csv.stat().st_size / (1024*1024)
    logger.info(f"Successfully generated Master Dataset: {len(master_df):,} records ({total_mb:.1f} MB on disk)")
    
    # Chronological Splitting
    logger.info("Applying strict Chronological Time Splits (Train / Validate / Test)...")
    master_df['date'] = pd.to_datetime(master_df['date'], format="mixed", dayfirst=True)
    master_df = master_df.sort_values('date').dropna(subset=['modal_price'])
    
    n = len(master_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = master_df.iloc[:train_end]
    val_df = master_df.iloc[train_end:val_end]
    test_df = master_df.iloc[val_end:]
    
    # Save into dedicated folders
    logger.info("Saving split files...")
    train_df.to_csv(TRAIN_DIR / "train.csv", index=False)
    val_df.to_csv(VAL_DIR / "val.csv", index=False)
    test_df.to_csv(TEST_DIR / "test.csv", index=False)
    
    # Mirror into data/processed
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    logger.info("=====================================================================")
    logger.info("MEGA-DATASET GENERATION COMPLETE:")
    logger.info(f"TOTAL ROWS: {len(master_df):,} Mandi records across India")
    logger.info(f"TOTAL FILE SIZE: {total_mb:.1f} MB on disk")
    logger.info(f"📁 TRAIN FOLDER    ({TRAIN_DIR / 'train.csv'}): {len(train_df):,} records ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logger.info(f"📁 VALIDATE FOLDER ({VAL_DIR / 'val.csv'}):   {len(val_df):,} records ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    logger.info(f"📁 TEST FOLDER     ({TEST_DIR / 'test.csv'}):  {len(test_df):,} records ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    logger.info("=====================================================================")

if __name__ == "__main__":
    download_full_archive()
