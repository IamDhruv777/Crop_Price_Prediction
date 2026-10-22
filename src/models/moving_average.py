import numpy as np
import pandas as pd

class MovingAverageForecast:
    def __init__(self):
        self.window = None
        self.last_window_mean = None

    def fit(self, series: pd.Series, window: int = 7) -> None:
        if len(series) < window:
            raise ValueError("Series length must be at least the window size.")
        self.window = window
        self.last_window_mean = series.iloc[-window:].mean()

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.last_window_mean is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return np.repeat(self.last_window_mean, steps)

    def select_window_by_validation(self, series: pd.Series, val_size: float = 0.2, windows: list = [3, 7, 14, 30]) -> int:
        n = len(series)
        val_len = int(n * val_size)
        train_len = n - val_len
        
        train_series = series.iloc[:train_len]
        val_series = series.iloc[train_len:]
        
        best_window = None
        best_mse = float('inf')
        
        for w in windows:
            if len(train_series) < w:
                continue
            
            mse = 0
            for i in range(len(val_series)):
                if i == 0:
                    hist = train_series.iloc[-w:]
                else:
                    hist = pd.concat([train_series, val_series.iloc[:i]]).iloc[-w:]
                pred = hist.mean()
                mse += (pred - val_series.iloc[i]) ** 2
            
            mse /= len(val_series)
            if mse < best_mse:
                best_mse = mse
                best_window = w
                
        return best_window
