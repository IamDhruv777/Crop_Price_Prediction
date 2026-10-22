import pandas as pd
import numpy as np

class ArrivalFeatureEngineer:
    def __init__(self):
        pass

    def create_arrival_lag_features(self, df: pd.DataFrame, arrival_col: str, lags: list = [1, 7, 30], group_cols: list = ['market_id', 'commodity']) -> pd.DataFrame:
        df_out = df.copy()
        for lag in lags:
            df_out[f'{arrival_col}_lag_{lag}'] = df_out.groupby(group_cols)[arrival_col].shift(lag)
        return df_out

    def create_arrival_rolling_features(self, df: pd.DataFrame, arrival_col: str, windows: list = [7, 30], group_cols: list = ['market_id', 'commodity']) -> pd.DataFrame:
        df_out = df.copy()
        for w in windows:
            df_out[f'{arrival_col}_rolling_mean_{w}'] = df_out.groupby(group_cols)[arrival_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df_out[f'{arrival_col}_rolling_std_{w}'] = df_out.groupby(group_cols)[arrival_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        return df_out

    def create_arrival_growth_feature(self, df: pd.DataFrame, arrival_col: str, group_cols: list) -> pd.DataFrame:
        df_out = df.copy()
        if f'{arrival_col}_lag_1' not in df_out.columns or f'{arrival_col}_lag_7' not in df_out.columns:
            df_out = self.create_arrival_lag_features(df_out, arrival_col, lags=[1, 7], group_cols=group_cols)
        
        lag_1 = df_out[f'{arrival_col}_lag_1']
        lag_7 = df_out[f'{arrival_col}_lag_7']
        df_out[f'{arrival_col}_growth'] = (lag_1 - lag_7) / lag_7.replace(0, np.nan)
        return df_out

    def create_arrival_pct_change(self, df: pd.DataFrame, arrival_col: str, group_cols: list) -> pd.DataFrame:
        df_out = df.copy()
        if f'{arrival_col}_lag_1' not in df_out.columns or f'{arrival_col}_lag_2' not in df_out.columns:
            df_out = self.create_arrival_lag_features(df_out, arrival_col, lags=[1, 2], group_cols=group_cols)
        
        lag_1 = df_out[f'{arrival_col}_lag_1']
        lag_2 = df_out[f'{arrival_col}_lag_2']
        df_out[f'{arrival_col}_pct_change'] = (lag_1 - lag_2) / lag_2.replace(0, np.nan)
        return df_out

    def create_arrival_anomaly(self, df: pd.DataFrame, arrival_col: str, group_cols: list, window: int = 30) -> pd.DataFrame:
        df_out = df.copy()
        if f'{arrival_col}_rolling_mean_{window}' not in df_out.columns or f'{arrival_col}_rolling_std_{window}' not in df_out.columns:
            df_out = self.create_arrival_rolling_features(df_out, arrival_col, windows=[window], group_cols=group_cols)
            
        lag_1 = df_out.groupby(group_cols)[arrival_col].shift(1)
        mean = df_out[f'{arrival_col}_rolling_mean_{window}']
        std = df_out[f'{arrival_col}_rolling_std_{window}'].replace(0, np.nan)
        
        df_out[f'{arrival_col}_anomaly_{window}'] = (lag_1 - mean) / std
        return df_out

    def create_arrival_volatility(self, df: pd.DataFrame, arrival_col: str, group_cols: list, window: int = 14) -> pd.DataFrame:
        df_out = df.copy()
        if f'{arrival_col}_rolling_mean_{window}' not in df_out.columns or f'{arrival_col}_rolling_std_{window}' not in df_out.columns:
            df_out = self.create_arrival_rolling_features(df_out, arrival_col, windows=[window], group_cols=group_cols)
            
        mean = df_out[f'{arrival_col}_rolling_mean_{window}'].replace(0, np.nan)
        std = df_out[f'{arrival_col}_rolling_std_{window}']
        
        df_out[f'{arrival_col}_volatility_{window}'] = std / mean
        return df_out

    def create_missing_arrival_indicators(self, df: pd.DataFrame, arrival_col: str) -> pd.DataFrame:
        df_out = df.copy()
        df_out[f'{arrival_col}_is_missing'] = df_out[arrival_col].isna()
        return df_out

    def create_all_arrival_features(self, df: pd.DataFrame, arrival_col: str = 'arrival_quantity', group_cols: list = ['market_id', 'commodity']) -> pd.DataFrame:
        df_out = df.copy()
        df_out = self.create_missing_arrival_indicators(df_out, arrival_col)
        df_out = self.create_arrival_lag_features(df_out, arrival_col, lags=[1, 2, 7, 30], group_cols=group_cols)
        df_out = self.create_arrival_rolling_features(df_out, arrival_col, windows=[7, 14, 30], group_cols=group_cols)
        df_out = self.create_arrival_growth_feature(df_out, arrival_col, group_cols=group_cols)
        df_out = self.create_arrival_pct_change(df_out, arrival_col, group_cols=group_cols)
        df_out = self.create_arrival_anomaly(df_out, arrival_col, group_cols=group_cols, window=30)
        df_out = self.create_arrival_volatility(df_out, arrival_col, group_cols=group_cols, window=14)
        return df_out
