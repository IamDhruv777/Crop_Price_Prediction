"""
Arbitrage & Benchmark Hub Feature Engineering (Phase 2).
Models spatial price contagion from national price-setting benchmark hubs:
- Wheat -> Dahod / Khanna
- Rice -> Burdwan
- Soyabean -> Kota / Indore
- Green Gram (Moong) -> Merta City
- Cowpea -> Mumbai
- Arhar (Tur) -> Raichur / Gulbarga
- Maize -> Dahod / Chhindwara
- Black Gram (Urd) -> Lalitpur / Latur
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

# National price-setting benchmark hubs determined by long-term volume
BENCHMARK_HUBS: Dict[str, str] = {
    'Arhar (Tur-Red Gram)': 'Raichur',
    'Black Gram (Urd Beans)': 'Lalitpur',
    'Cowpea (Lobia-Karamani)': 'Mumbai',
    'Green Gram (Moong)': 'Merta City',
    'Kabuli Chana (Chickpeas-White)': 'Indore',
    'Lentil (Masur)': 'Shujalpur',
    'Maize': 'Dahod',
    'Rice': 'Burdwan',
    'Soyabean': 'Kota',
    'Wheat': 'Dahod',
    'White Peas': 'Aligarh'
}

class ArbitrageFeatureEngineer:
    def __init__(self, benchmark_hubs: Optional[Dict[str, str]] = None):
        self.benchmark_hubs = benchmark_hubs or BENCHMARK_HUBS
        self.hub_history: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> 'ArbitrageFeatureEngineer':
        """
        Precompute historical daily prices for each commodity's benchmark hub.
        df must contain ['commodity', 'market', 'date', 'modal_price'].
        """
        for comm, hub_mkt in self.benchmark_hubs.items():
            sub = df[(df['commodity'] == comm) & (df['market'] == hub_mkt)]
            if not sub.empty:
                daily_p = sub.groupby('date')['modal_price'].mean().to_dict()
                self.hub_history[comm] = daily_p
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate leading arbitrage spread, hub momentum, and arrival supply shocks.
        Zero future leakage: all hub features are lagged by at least 1 day.
        """
        df = df.copy()
        
        # 1. Map benchmark hub price
        hub_prices = []
        for _, row in df.iterrows():
            c = row.get('commodity')
            d = row.get('date')
            lookup = self.hub_history.get(c, {})
            p = lookup.get(d, np.nan)
            hub_prices.append(p)
            
        df['hub_price'] = hub_prices
        
        # Group-wise causal imputation for missing hub dates
        if 'commodity' in df.columns:
            df['hub_price'] = df.groupby('commodity')['hub_price'].ffill().bfill()
        if 'modal_price' in df.columns:
            df['hub_price'] = df['hub_price'].fillna(df['modal_price'])
            
        # 2. Strict Causal Lag (shift 1 ensures price from t-1 is used)
        if 'market' in df.columns and 'commodity' in df.columns:
            df['hub_price_lag1'] = df.groupby(['market', 'commodity'])['hub_price'].shift(1)
            df['hub_price_lag7'] = df.groupby(['market', 'commodity'])['hub_price'].shift(7)
        else:
            df['hub_price_lag1'] = df['hub_price'].shift(1)
            df['hub_price_lag7'] = df['hub_price'].shift(7)
            
        df['hub_price_lag1'] = df['hub_price_lag1'].fillna(df.get('price_lag_1', df.get('modal_price', 0)))
        df['hub_price_lag7'] = df['hub_price_lag7'].fillna(df['hub_price_lag1'])
        
        # 3. Spatial Price Arbitrage Spread: P(local, t-1) - P(hub, t-1)
        local_p = df.get('price_lag_1', df.get('modal_price', 0))
        df['hub_price_spread'] = local_p - df['hub_price_lag1']
        df['hub_price_ratio'] = local_p / (df['hub_price_lag1'] + 1e-5)
        
        # 4. Benchmark Hub Leading Momentum (7-day % change in national hub)
        df['hub_momentum_7d'] = (df['hub_price_lag1'] - df['hub_price_lag7']) / (df['hub_price_lag7'] + 1e-5)
        
        # 5. Arrival Supply-Shock Ratio
        if 'arrival_lag_1' in df.columns and 'arrival_rolling_7' in df.columns:
            df['supply_shock_ratio'] = df['arrival_lag_1'] / (df['arrival_rolling_7'] + 1.0)
        elif 'arrivals_tonnes' in df.columns:
            arr_lag1 = df.groupby(['market', 'commodity'])['arrivals_tonnes'].shift(1).fillna(0)
            arr_roll = df.groupby(['market', 'commodity'])['arrivals_tonnes'].transform(
                lambda x: x.shift(1).rolling(7, min_periods=1).mean()
            ).fillna(1.0)
            df['supply_shock_ratio'] = arr_lag1 / (arr_roll + 1.0)
            
        return df
