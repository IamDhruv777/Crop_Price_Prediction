import pandas as pd
import numpy as np

class OutlierDetector:
    def detect_iqr_outliers(self, series: pd.Series, multiplier: float = 3.0) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)

    def detect_mad_outliers(self, series: pd.Series, threshold: float = 3.5) -> pd.Series:
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return pd.Series(False, index=series.index)
        
        modified_z_score = 0.6745 * (series - median) / mad
        return modified_z_score.abs() > threshold

    def detect_rolling_zscore_outliers(self, series: pd.Series, window: int = 30, threshold: float = 3.0) -> pd.Series:
        rolling_mean = series.shift(1).rolling(window=window, min_periods=1).mean()
        rolling_std = series.shift(1).rolling(window=window, min_periods=1).std()
        
        z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        return z_scores.abs() > threshold

    def detect_price_outliers(self, df: pd.DataFrame, price_col: str = 'modal_price', 
                              market_col: str = 'market_id', commodity_col: str = 'commodity') -> pd.DataFrame:
        result_df = df.copy()
        
        if price_col not in result_df.columns:
            return result_df
            
        result_df['is_outlier_iqr'] = False
        result_df['is_outlier_mad'] = False
        result_df['is_outlier_rolling_z'] = False
        
        if all(col in result_df.columns for col in [market_col, commodity_col]):
            grouped = result_df.groupby([market_col, commodity_col])[price_col]
            
            result_df['is_outlier_iqr'] = grouped.transform(self.detect_iqr_outliers)
            result_df['is_outlier_mad'] = grouped.transform(self.detect_mad_outliers)
            result_df['is_outlier_rolling_z'] = grouped.transform(self.detect_rolling_zscore_outliers)

        result_df['is_outlier_consensus'] = (
            result_df['is_outlier_iqr'].fillna(False).astype(int) + 
            result_df['is_outlier_mad'].fillna(False).astype(int) + 
            result_df['is_outlier_rolling_z'].fillna(False).astype(int)
        ) >= 2
        
        return result_df

    def classify_price_event(self, series: pd.Series, window: int = 30, n_std: float = 2.0) -> pd.Series:
        rolling_mean = series.shift(1).rolling(window=window, min_periods=1).mean()
        rolling_std = series.shift(1).rolling(window=window, min_periods=1).std()
        
        upper_bound = rolling_mean + (n_std * rolling_std)
        lower_bound = rolling_mean - (n_std * rolling_std)
        
        event = pd.Series('normal', index=series.index)
        event[series > upper_bound] = 'spike'
        event[series < lower_bound] = 'drop'
        
        volatility_mask = rolling_std > (rolling_mean * 0.5)
        event[(event == 'normal') & volatility_mask] = 'high_volatility'
        
        return event
