import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

class SARIMAModel:
    def __init__(self):
        self.model = None
        self.fit_model = None
        self.order = None
        self.seasonal_order = None

    def fit(self, train_series: pd.Series, order: tuple, seasonal_order: tuple) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        self.fit_model = self.model.fit(disp=False)

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.fit_model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        preds = self.fit_model.forecast(steps=steps)
        return preds.values

    def tune(self, train_series: pd.Series, val_series: pd.Series, max_p=2, max_d=1, max_q=2, seasonal_periods=[7, 30]) -> tuple:
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        
        P_values = range(0, 2)
        D_values = range(0, 2)
        Q_values = range(0, 2)
        
        best_score, best_cfg = float("inf"), None
        
        for p, d, q in itertools.product(p_values, d_values, q_values):
            for P, D, Q in itertools.product(P_values, D_values, Q_values):
                for s in seasonal_periods:
                    order = (p, d, q)
                    seasonal_order = (P, D, Q, s)
                    try:
                        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                        fit_model = model.fit(disp=False)
                        preds = fit_model.forecast(steps=len(val_series))
                        mse = np.mean((preds.values - val_series.values) ** 2)
                        if mse < best_score:
                            best_score = mse
                            best_cfg = (order, seasonal_order)
                    except:
                        continue
                        
        if best_cfg is not None:
            self.order = best_cfg[0]
            self.seasonal_order = best_cfg[1]
        return best_cfg

    def get_residuals(self) -> pd.Series:
        if self.fit_model is None:
            raise RuntimeError("Model not fitted.")
        return self.fit_model.resid
