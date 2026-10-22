"""
Step 2 & Step 3: Unified Fusion & Full Feature Engineering Pipeline.
Fuses:
  - Mandi Prices & Arrivals (AGMARKNET)
  - Daily Weather Time-Series (NASA Meteorological)
  - Spatial Neighbor Markets (Graph Adjacency & Haversine Distance)
Generates:
  - Price Lags (1, 7, 14, 30) & Rolling Stats (Mean, Std, Momentum)
  - Arrival Dynamics (Lag 1, 7, Rolling Volume, Anomaly Score)
  - Temporal & Agronomic Encodings (Month, Week, Kharif/Rabi, Cyclic sin/cos)
  - Weather Signals (Rainfall, Temp, Humidity, Weather Anomalies, Lagged Weather)
  - Spatial Market Signals (Mean Neighbor Price at t-1 from Graph Edges)
Outputs:
  - Clean, leak-free Chronological Train, Validation, and Test Sets.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_MASTER = DATA_DIR / "raw" / "bulk_mandi" / "agmarknet_full_10year_master.csv"
WEATHER_FILE = DATA_DIR / "raw" / "weather" / "mandi_daily_weather_aligned.csv"
GEO_FILE = DATA_DIR / "raw" / "geography" / "market_master.csv"
SPATIAL_GRAPH = DATA_DIR / "spatial" / "mandi_adjacency_graph.json"

SPLITS_DIR = DATA_DIR / "processed" / "unified_splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

def generate_unified_features(sample_records=300000):
    logger.info("==========================================================================")
    logger.info("=== STEP 2 & 3: FULL MULTI-SOURCE DATA FUSION & FEATURE ENGINEERING ===")
    logger.info("==========================================================================")
    
    # 1. Load Raw Mandi Prices & Arrivals
    logger.info("Loading AGMARKNET Mandi Price and Arrival records...")
    df = pd.read_csv(RAW_MASTER, low_memory=False)
    if sample_records and len(df) > sample_records:
        df = df.iloc[-sample_records:].copy()
        
    df['date'] = pd.to_datetime(df['date'], format="mixed", dayfirst=True, errors='coerce')
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df['min_price'] = pd.to_numeric(df['min_price'], errors='coerce')
    df['max_price'] = pd.to_numeric(df['max_price'], errors='coerce')
    df['arrivals_tonnes'] = pd.to_numeric(df['arrivals_tonnes'], errors='coerce').fillna(0)
    
    # Data Cleaning & Outlier Removal (Phase 2)
    df = df.dropna(subset=['date', 'modal_price', 'market', 'commodity'])
    df = df[(df['modal_price'] >= 100) & (df['modal_price'] <= 35000)].copy()
    
    # 2. Load and Merge Daily Weather Data (Step 1.B)
    logger.info("Merging Daily Meteorological Weather Data (Rainfall, Temp, Humidity)...")
    weather_df = pd.read_csv(WEATHER_FILE)
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    # Merge on market_name and date
    df = pd.merge(df, weather_df[['market_name', 'date', 'temperature_mean', 'temperature_max', 'temperature_min', 'rainfall_mm', 'relative_humidity', 'wind_speed_ms']],
                  left_on=['market', 'date'], right_on=['market_name', 'date'], how='left')
    df = df.drop(columns=['market_name'], errors='ignore')
    
    # Fill remaining weather for markets outside the primary 15 with state-level seasonal averages
    df['temperature_mean'] = df['temperature_mean'].fillna(28.0)
    df['temperature_max'] = df['temperature_max'].fillna(34.0)
    df['temperature_min'] = df['temperature_min'].fillna(22.0)
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0.0)
    df['relative_humidity'] = df['relative_humidity'].fillna(55.0)
    df['wind_speed_ms'] = df['wind_speed_ms'].fillna(4.0)
    
    # 3. Step 3.A: Price Feature Engineering
    logger.info("Engineering Price Features (Lags 1, 7, 14, 30, Rolling Mean/Std, Momentum)...")
    df = df.sort_values(['market', 'commodity', 'date']).reset_index(drop=True)
    
    # Multi-Horizon Targets
    df['target_price_1d'] = df.groupby(['market', 'commodity'])['modal_price'].shift(-1)
    df['target_price_7d'] = df.groupby(['market', 'commodity'])['modal_price'].shift(-7)
    df['target_price_30d'] = df.groupby(['market', 'commodity'])['modal_price'].shift(-30)
    
    # Directional Target (7-day trend >= +1.5%)
    df['target_direction_7d'] = (((df['target_price_7d'] - df['modal_price']) / (df['modal_price'] + 1e-5)) >= 0.015).astype(int)
    
    # Strict Causal Price Lags (shift 1 ensures zero future leakage)
    for lag in [1, 7, 14, 30]:
        df[f'price_lag_{lag}'] = df.groupby(['market', 'commodity'])['modal_price'].shift(lag)
        
    for w in [7, 14, 30]:
        df[f'price_rolling_mean_{w}'] = df.groupby(['market', 'commodity'])['modal_price'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'price_rolling_std_{w}'] = df.groupby(['market', 'commodity'])['modal_price'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        ).fillna(0)
        
    df['price_momentum'] = (df['price_lag_1'] - df['price_rolling_mean_30']) / (df['price_rolling_mean_30'] + 1e-5)
    df['price_spread'] = df['max_price'] - df['min_price']
    
    # 4. Step 3.B: Arrival Feature Engineering
    logger.info("Engineering Arrival Features (Lags, Growth, Rolling, Anomalies)...")
    df['arrival_lag_1'] = df.groupby(['market', 'commodity'])['arrivals_tonnes'].shift(1)
    df['arrival_lag_7'] = df.groupby(['market', 'commodity'])['arrivals_tonnes'].shift(7)
    df['arrival_rolling_7'] = df.groupby(['market', 'commodity'])['arrivals_tonnes'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df['arrival_rolling_30'] = df.groupby(['market', 'commodity'])['arrivals_tonnes'].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )
    df['arrival_growth_7d'] = (df['arrival_lag_1'] - df['arrival_lag_7']) / (df['arrival_lag_7'] + 1.0)
    df['arrival_anomaly'] = (df['arrival_lag_1'] - df['arrival_rolling_30']) / (df['arrival_rolling_30'] + 1.0)
    
    # 5. Step 3.C: Temporal & Agronomic Features
    logger.info("Engineering Temporal Features (Month, Week, Season, Cyclic Encodings)...")
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year
    
    # Cyclic Encodings
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Agricultural Seasons
    df['is_kharif'] = df['month'].isin([6, 7, 8, 9, 10]).astype(int)
    df['is_rabi'] = df['month'].isin([11, 12, 1, 2, 3, 4]).astype(int)
    
    # 6. Step 3.D: Weather Features & Anomalies
    logger.info("Engineering Weather Features (Rainfall, Temp, Humidity, Weather Anomalies)...")
    df['weather_rain_lag_1'] = df.groupby(['market'])['rainfall_mm'].shift(1)
    df['weather_rain_lag_7'] = df.groupby(['market'])['rainfall_mm'].shift(7)
    df['weather_rain_sum_14'] = df.groupby(['market'])['rainfall_mm'].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).sum()
    )
    df['weather_temp_mean_lag_1'] = df.groupby(['market'])['temperature_mean'].shift(1)
    df['weather_humidity_lag_1'] = df.groupby(['market'])['relative_humidity'].shift(1)
    
    # Weather Anomaly: Temperature deviation from 30-day baseline
    df['weather_temp_rolling_30'] = df.groupby(['market'])['temperature_mean'].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )
    df['weather_temp_anomaly'] = df['weather_temp_mean_lag_1'] - df['weather_temp_rolling_30']
    
    # Growing Degree Days (GDD)
    df['growing_degree_days'] = np.maximum(0, df['weather_temp_mean_lag_1'] - 10.0)
    
    # 7. Step 4: Spatial Graph Information (Neighbor Market Prices at t-1)
    logger.info("Integrating Step 4: Region-Wise Spatial Neighbor Features from Graph Edges...")
    with open(SPATIAL_GRAPH, "r") as f:
        adj_graph = json.load(f)
        
    # Pre-compute lag_1 price table for fast spatial lookup
    mandi_price_lookup = df.set_index(['market', 'commodity', 'date'])['price_lag_1'].to_dict()
    
    def get_spatial_neighbor_price(row):
        m = row['market']
        c = row['commodity']
        d = row['date']
        neighbors = adj_graph.get(m, [])
        if not neighbors:
            return row['price_lag_1']
        prices = []
        for n_info in neighbors:
            n_name = n_info['neighbor']
            p = mandi_price_lookup.get((n_name, c, d))
            if p is not None and not np.isnan(p):
                prices.append(p)
        return np.mean(prices) if prices else row['price_lag_1']
        
    df['spatial_neighbor_price_mean'] = df.apply(get_spatial_neighbor_price, axis=1)
    df['spatial_price_ratio'] = df['price_lag_1'] / (df['spatial_neighbor_price_mean'] + 1e-5)
    
    # Clean nulls from lag creation
    df = df.dropna(subset=['target_price_7d', 'price_lag_30', 'price_rolling_mean_30']).reset_index(drop=True)
    logger.info(f"Total fully featured and aligned records: {len(df):,}")
    
    # 8. Strict Chronological Split (Phase 6: No Data Leakage)
    logger.info("Executing Chronological Train / Validate / Test Partitioning...")
    df = df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Save splits
    train_df.to_csv(SPLITS_DIR / "train_unified.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val_unified.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test_unified.csv", index=False)
    
    logger.info("==========================================================================")
    logger.info("UNIFIED DATA FUSION & FEATURE ENGINEERING COMPLETE:")
    logger.info(f"📁 TRAIN UNIFIED:    {len(train_df):,} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logger.info(f"📁 VAL UNIFIED:      {len(val_df):,} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    logger.info(f"📁 TEST UNIFIED:     {len(test_df):,} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    logger.info(f"Total Feature Columns: {len(train_df.columns)}")
    logger.info(f"Output Directory:    {SPLITS_DIR}")
    logger.info("==========================================================================")

if __name__ == "__main__":
    generate_unified_features()
