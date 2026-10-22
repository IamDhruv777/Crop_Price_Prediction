"""
ARIMA model wrapper using statsmodels.
Tuning uses validation data only — NEVER test data.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import itertools

class ARIMAModel:
    def __init__(self):
        self.model = None
        self.fit_model = None
        self.order = None

    def fit(self, train_series: pd.Series, order: tuple = None) -> None:
        if order is None:
            try:
                from pmdarima import auto_arima
                pmd_model = auto_arima(train_series, suppress_warnings=True, seasonal=False)
                order = pmd_model.order
            except ImportError:
                raise ImportError("pmdarima not installed. Please provide 'order' manually or install pmdarima.")
        
        self.order = order
        self.model = ARIMA(train_series, order=self.order)
        self.fit_model = self.model.fit()

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.fit_model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        preds = self.fit_model.forecast(steps=steps)
        return preds.values

    def tune(self, train_series: pd.Series, val_series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> tuple:
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        
        best_score, best_cfg = float("inf"), None
        
        for p, d, q in itertools.product(p_values, d_values, q_values):
            order = (p, d, q)
            try:
                model = ARIMA(train_series, order=order)
                fit_model = model.fit()
                # predict validation
                preds = fit_model.forecast(steps=len(val_series))
                mse = np.mean((preds.values - val_series.values) ** 2)
                if mse < best_score:
                    best_score = mse
                    best_cfg = order
            except:
                continue
                
        self.order = best_cfg
        return best_cfg

    def get_residuals(self) -> pd.Series:
        if self.fit_model is None:
            raise RuntimeError("Model not fitted.")
        return self.fit_model.resid

    def get_aic(self) -> float:
        if self.fit_model is None:
            raise RuntimeError("Model not fitted.")
        return self.fit_model.aic
