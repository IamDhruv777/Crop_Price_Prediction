"""
Price feature engineering.
CRITICAL: All lag and rolling features use ONLY past/current data.
No future prices are ever used to compute features at time t.
"""

import pandas as pd
import numpy as np

class PriceFeatureEngineer:
    def __init__(self):
        pass

    def create_lag_features(self, df: pd.DataFrame, price_col: str, lags: list = [1, 7, 14, 30], group_cols: list = ['market_id', 'commodity']) -> pd.DataFrame:
        df_out = df.copy()
        for lag in lags:
            df_out[f'{price_col}_lag_{lag}'] = df_out.groupby(group_cols)[price_col].shift(lag)
        return df_out

    def create_rolling_features(self, df: pd.DataFrame, price_col: str, windows: list = [7, 14, 30], group_cols: list = ['market_id', 'commodity']) -> pd.DataFrame:
        df_out = df.copy()
        for w in windows:
            df_out[f'{price_col}_rolling_mean_{w}'] = df_out.groupby(group_cols)[price_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df_out[f'{price_col}_rolling_std_{w}'] = df_out.groupby(group_cols)[price_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
            df_out[f'{price_col}_rolling_min_{w}'] = df_out.groupby(group_cols)[price_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())
            df_out[f'{price_col}_rolling_max_{w}'] = df_out.groupby(group_cols)[price_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        return df_out

    def create_price_change_features(self, df: pd.DataFrame, price_col: str, group_cols: list) -> pd.DataFrame:
        df_out = df.copy()
        if f'{price_col}_lag_1' not in df_out.columns:
            df_out = self.create_lag_features(df_out, price_col, lags=[1, 2, 7, 30], group_cols=group_cols)
        
        lag_1 = df_out[f'{price_col}_lag_1']
        lag_2 = df_out[f'{price_col}_lag_2'] if f'{price_col}_lag_2' in df_out.columns else df_out.groupby(group_cols)[price_col].shift(2)
        lag_7 = df_out[f'{price_col}_lag_7'] if f'{price_col}_lag_7' in df_out.columns else df_out.groupby(group_cols)[price_col].shift(7)
        lag_30 = df_out[f'{price_col}_lag_30'] if f'{price_col}_lag_30' in df_out.columns else df_out.groupby(group_cols)[price_col].shift(30)
        
        df_out['price_change_1d'] = lag_1 - lag_2
        df_out['price_change_7d'] = lag_1 - lag_7
        df_out['price_change_30d'] = lag_1 - lag_30
        
        df_out['price_pct_change_1d'] = (lag_1 - lag_2) / lag_2.replace(0, np.nan)
        df_out['price_pct_change_7d'] = (lag_1 - lag_7) / lag_7.replace(0, np.nan)
        df_out['price_pct_change_30d'] = (lag_1 - lag_30) / lag_30.replace(0, np.nan)
        
        return df_out

    def create_momentum_features(self, df: pd.DataFrame, price_col: str, group_cols: list) -> pd.DataFrame:
        df_out = df.copy()
        if f'{price_col}_lag_1' not in df_out.columns:
            df_out = self.create_lag_features(df_out, price_col, lags=[1], group_cols=group_cols)
        if f'{price_col}_rolling_mean_30' not in df_out.columns:
            df_out = self.create_rolling_features(df_out, price_col, windows=[14, 30], group_cols=group_cols)
            
        df_out['price_momentum'] = df_out[f'{price_col}_lag_1'] / df_out[f'{price_col}_rolling_mean_30'].replace(0, np.nan) - 1
        df_out['price_volatility'] = df_out[f'{price_col}_rolling_std_14'] / df_out[f'{price_col}_rolling_mean_14'].replace(0, np.nan)
        
        return df_out

    def create_all_price_features(self, df: pd.DataFrame, price_col: str = 'modal_price', group_cols: list = ['market_id', 'commodity']) -> pd.DataFrame:
        df_out = df.copy()
        df_out = self.create_lag_features(df_out, price_col, lags=[1, 2, 7, 14, 30], group_cols=group_cols)
        df_out = self.create_rolling_features(df_out, price_col, windows=[7, 14, 30], group_cols=group_cols)
        df_out = self.create_price_change_features(df_out, price_col, group_cols=group_cols)
        df_out = self.create_momentum_features(df_out, price_col, group_cols=group_cols)
        return df_out

    def verify_no_leakage(self, df: pd.DataFrame, price_col: str, feature_cols: list) -> bool:
        df_temp = df.copy()
        df_temp['future_price'] = df_temp[price_col].shift(-1)
        df_temp['past_price'] = df_temp[price_col].shift(1)
        
        for col in feature_cols:
            if col in df_temp.columns:
                corr_future = df_temp[col].corr(df_temp['future_price'])
                corr_past = df_temp[col].corr(df_temp['past_price'])
                
                if pd.isna(corr_future) or pd.isna(corr_past):
                    continue
                    
                if corr_future > corr_past + 0.05: 
                    raise ValueError(f"Data Leakage Detected: Feature {col} correlates more with future price ({corr_future:.4f}) than past price ({corr_past:.4f})")
        return True
