import pandas as pd
import numpy as np

class WeatherFeatureEngineer:
    def __init__(self):
        pass

    def merge_weather_with_prices(self, price_df: pd.DataFrame, weather_df: pd.DataFrame, market_id_col: str, date_col: str) -> pd.DataFrame:
        price_df[date_col] = pd.to_datetime(price_df[date_col]).dt.tz_localize(None)
        weather_df[date_col] = pd.to_datetime(weather_df[date_col]).dt.tz_localize(None)
        
        merged_df = pd.merge(price_df, weather_df, on=[market_id_col, date_col], how='left')
        return merged_df

    def create_weather_lag_features(self, df: pd.DataFrame, weather_cols: list, lags: list = [1, 7, 14], group_cols: list = ['market_id']) -> pd.DataFrame:
        df_out = df.copy()
        for col in weather_cols:
            if col in df_out.columns:
                for lag in lags:
                    df_out[f'{col}_lag_{lag}'] = df_out.groupby(group_cols)[col].shift(lag)
        return df_out

    def create_weather_rolling_features(self, df: pd.DataFrame, weather_cols: list, windows: list = [7, 14, 30], group_cols: list = ['market_id']) -> pd.DataFrame:
        df_out = df.copy()
        for col in weather_cols:
            if col in df_out.columns:
                for w in windows:
                    df_out[f'{col}_rolling_mean_{w}'] = df_out.groupby(group_cols)[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    df_out[f'{col}_rolling_std_{w}'] = df_out.groupby(group_cols)[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        return df_out

    def create_weather_anomaly_features(self, df: pd.DataFrame, weather_cols: list, window: int = 30, group_cols: list = ['market_id']) -> pd.DataFrame:
        df_out = df.copy()
        for col in weather_cols:
            if col in df_out.columns:
                if f'{col}_rolling_mean_{window}' not in df_out.columns or f'{col}_rolling_std_{window}' not in df_out.columns:
                    df_out = self.create_weather_rolling_features(df_out, [col], windows=[window], group_cols=group_cols)
                
                val = df_out[col] 
                mean = df_out[f'{col}_rolling_mean_{window}']
                std = df_out[f'{col}_rolling_std_{window}'].replace(0, np.nan)
                
                df_out[f'{col}_anomaly_{window}'] = (val - mean) / std
        return df_out

    def create_growing_degree_days(self, df: pd.DataFrame, base_temp: float = 10.0, temp_col: str = 'temp_mean') -> pd.Series:
        if temp_col in df.columns:
            return np.maximum(0, df[temp_col] - base_temp)
        return pd.Series(np.nan, index=df.index)

    def create_all_weather_features(self, df: pd.DataFrame, group_cols: list = ['market_id']) -> pd.DataFrame:
        df_out = df.copy()
        
        skip_cols = group_cols + ['price_date', 'modal_price', 'commodity', 'arrival_quantity']
        weather_cols = [c for c in df_out.columns if c not in skip_cols and not c.endswith('_lag') and '_rolling_' not in c]
        
        df_out = self.create_weather_lag_features(df_out, weather_cols, lags=[1, 7, 14], group_cols=group_cols)
        df_out = self.create_weather_rolling_features(df_out, weather_cols, windows=[7, 14, 30], group_cols=group_cols)
        df_out = self.create_weather_anomaly_features(df_out, weather_cols, window=30, group_cols=group_cols)
        
        temp_cols = [c for c in weather_cols if 'temp' in c.lower() and 'mean' in c.lower()]
        if temp_cols:
            df_out['gdd'] = self.create_growing_degree_days(df_out, temp_col=temp_cols[0])
            
        return df_out
