import pandas as pd
import numpy as np

class MissingValueHandler:
    def analyze_missingness(self, df: pd.DataFrame, target_cols: list) -> dict:
        report = {}
        for col in target_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                total_count = len(df)
                report[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_count / total_count * 100) if total_count > 0 else 0.0
                }
        return report

    def interpolate_price_gaps(self, series: pd.Series, max_gap: int = 14) -> pd.DataFrame:
        name = series.name if series.name else 'price'
        was_missing = series.isna()
        
        # Uses backward-fill-safe approach (forward direction only)
        interpolated = series.interpolate(method='linear', limit=max_gap, limit_direction='forward')
        is_interpolated = was_missing & ~interpolated.isna()
        
        return pd.DataFrame({
            name: interpolated,
            f"{name}_interpolated": is_interpolated.astype(int)
        })

    def fill_weather_gaps(self, df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
        filled_df = df.copy()
        numeric_cols = filled_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            filled_df[col] = filled_df[col].interpolate(method='linear', limit=max_gap, limit_direction='forward')
            
        return filled_df

    def add_missing_indicators(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        out_df = df.copy()
        for col in cols:
            if col in out_df.columns:
                indicator_col = f'is_missing_{col}'
                out_df[indicator_col] = out_df[col].isna().astype(int)
        return out_df

    def compute_missingness_threshold(self, df: pd.DataFrame, col: str, threshold: float = 0.20) -> bool:
        if col not in df.columns:
            return True
        missing_ratio = df[col].isna().sum() / len(df)
        return bool(missing_ratio > threshold)
