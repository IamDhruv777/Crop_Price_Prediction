import pandas as pd
import numpy as np
from typing import List, Tuple

"""
Expanding-window walk-forward cross-validation for time series.
NEVER shuffles data. Always respects chronological order.
"""

class WalkForwardValidator:
    def __init__(self, n_folds=5, min_train_days=180, step_days=30):
        self.n_folds = n_folds
        self.min_train_days = min_train_days
        self.step_days = step_days
        
    def split(self, df: pd.DataFrame, date_col='price_date') -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        df = df.sort_values(date_col)
        dates = df[date_col].dt.normalize().unique()
        dates = np.sort(dates)
        
        if len(dates) < self.min_train_days + self.n_folds * self.step_days:
            raise ValueError("Not enough dates for the requested number of folds and step days.")
            
        splits = []
        
        for i in range(self.n_folds):
            # Calculate the end of the training window
            train_end_idx = self.min_train_days + i * self.step_days
            if train_end_idx >= len(dates):
                break
                
            train_dates = dates[:train_end_idx]
            
            # The validation window starts immediately after the training window
            val_end_idx = min(train_end_idx + self.step_days, len(dates))
            val_dates = dates[train_end_idx:val_end_idx]
            
            if len(val_dates) == 0:
                break
                
            # Validate chronological ordering and no overlap
            assert train_dates.max() < val_dates.min(), "Validation dates must strictly follow training dates."
            assert len(np.intersect1d(train_dates, val_dates)) == 0, "Training and validation dates must not overlap."
            
            splits.append((train_dates, val_dates))
            
        return splits
        
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, dates: pd.Series, group_cols=None) -> pd.DataFrame:
        df_dates = pd.DataFrame({'price_date': dates})
        splits = self.split(df_dates, date_col='price_date')
        
        results = []
        for fold_idx, (train_dates, val_dates) in enumerate(splits):
            train_mask = dates.isin(train_dates)
            val_mask = dates.isin(val_dates)
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            from .evaluation import compute_all_metrics
            metrics = compute_all_metrics(y_val, y_pred)
            metrics['fold'] = fold_idx
            metrics['train_start'] = train_dates.min()
            metrics['train_end'] = train_dates.max()
            metrics['val_start'] = val_dates.min()
            metrics['val_end'] = val_dates.max()
            
            results.append(metrics)
            
        return pd.DataFrame(results)

    def get_fold_summary(self, fold_results: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = fold_results.select_dtypes(include=[np.number]).columns
        # Drop fold specific metrics to just get means
        cols_to_mean = [c for c in numeric_cols if c != 'fold']
        
        summary = fold_results[cols_to_mean].mean().to_frame(name='mean_metrics').T
        return summary

def chronological_train_val_test_split(df: pd.DataFrame, date_col: str, train_ratio=0.70, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(date_col).copy()
    
    n_samples = len(df_sorted)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    
    assert train[date_col].max() <= val[date_col].min(), "Train end must be <= val start"
    assert val[date_col].max() <= test[date_col].min(), "Val end must be <= test start"
    
    return train, val, test
