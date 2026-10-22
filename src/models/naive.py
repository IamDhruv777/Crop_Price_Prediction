"""
Naive forecast baseline.
prediction(t+h) = price(t) for h=1
For multi-step: recursive application or seasonal naive fallback.
"""
import pandas as pd
import numpy as np

class NaiveForecast:
    def __init__(self):
        self.last_value = None

    def fit(self, series: pd.Series) -> None:
        if len(series) == 0:
            raise ValueError("Empty series provided for training.")
        self.last_value = series.iloc[-1]

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.last_value is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return np.repeat(self.last_value, steps)
    
    def predict_horizon(self, series: pd.Series, horizon: int) -> dict:
        if len(series) == 0:
            raise ValueError("Empty series.")
        last_val = series.iloc[-1]
        preds = np.repeat(last_val, horizon)
        if isinstance(series.index, pd.DatetimeIndex):
            future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon)
            return dict(zip(future_dates, preds))
        else:
            return {f"step_{i+1}": float(p) for i, p in enumerate(preds)}

class SeasonalNaiveForecast:
    def __init__(self):
        self.seasonality = None
        self.historical_values = None

    def fit(self, series: pd.Series, seasonality: int = 7) -> None:
        if len(series) == 0:
            raise ValueError("Empty series.")
        if len(series) < seasonality:
            raise ValueError(f"Series length must be at least the seasonality ({seasonality}).")
        self.seasonality = seasonality
        self.historical_values = series.iloc[-seasonality:].values

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.historical_values is None:
            raise RuntimeError("Model must be fitted before prediction.")
        
        preds = []
        for i in range(steps):
            preds.append(self.historical_values[i % self.seasonality])
        return np.array(preds)
