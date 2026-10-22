import pandas as pd
import numpy as np
from pathlib import Path

MASTER_PATH = Path("data/raw/bulk_mandi/agmarknet_full_10year_master.csv")
OUT_PATH = Path("data/processed/historical_multi_crop_sample.csv")

def prepare_historical_sample():
    print(f"Reading from {MASTER_PATH} in chunks...")
    chunks = []
    # Collect balanced samples across all 11 commodities
    cols = ['state', 'district', 'market', 'commodity', 'date', 'modal_price', 'min_price', 'max_price', 'arrivals_tonnes']
    
    # Read in chunks of 200,000 to manage memory
    chunk_iter = pd.read_csv(MASTER_PATH, usecols=cols, chunksize=200000, low_memory=False)
    
    commodity_counts = {}
    sampled_dfs = []
    target_per_crop = 15000
    
    for i, chunk in enumerate(chunk_iter):
        chunk = chunk.dropna(subset=['date', 'modal_price', 'commodity', 'market'])
        chunk['modal_price'] = pd.to_numeric(chunk['modal_price'], errors='coerce')
        chunk = chunk[chunk['modal_price'] > 50]
        
        for crop, group in chunk.groupby('commodity'):
            current_count = commodity_counts.get(crop, 0)
            if current_count < target_per_crop:
                needed = target_per_crop - current_count
                sample = group.head(needed)
                sampled_dfs.append(sample)
                commodity_counts[crop] = current_count + len(sample)
        
        if all(commodity_counts.get(c, 0) >= target_per_crop for c in [
            'Wheat', 'Rice', 'Green Gram (Moong)', 'Maize', 'Soyabean',
            'Arhar (Tur-Red Gram)', 'Black Gram (Urd Beans)', 'Lentil (Masur)',
            'Cowpea (Lobia-Karamani)', 'Kabuli Chana (Chickpeas-White)'
        ]):
            print(f"Reached target samples at chunk {i}!")
            break

    df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Total sampled rows: {len(df):,}")
    print("Crop distribution:")
    print(df['commodity'].value_counts())
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(by=['commodity', 'market', 'date'])
    
    # Calculate rolling features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['price_rolling_mean_7'] = df.groupby(['commodity', 'market'])['modal_price'].transform(lambda s: s.rolling(7, min_periods=1).mean()).round(2)
    df['price_rolling_mean_14'] = df.groupby(['commodity', 'market'])['modal_price'].transform(lambda s: s.rolling(14, min_periods=1).mean()).round(2)
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Successfully saved {len(df):,} multi-crop historical records to {OUT_PATH}")

if __name__ == '__main__':
    prepare_historical_sample()
