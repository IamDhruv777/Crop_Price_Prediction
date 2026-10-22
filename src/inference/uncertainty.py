"""
Prediction interval estimation.
Primary method: Conformal Prediction (split conformal)
Fallback: Bootstrap / residual-based intervals
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

class ConformalPredictor:
    def __init__(self):
        self.q_hat = None

    def calibrate(self, y_cal: Union[np.ndarray, list, pd.Series], 
                  y_pred_cal: Union[np.ndarray, list, pd.Series], 
                  alpha: float = 0.05) -> None:
        """
        Calibrates the conformal predictor using a calibration set.
        """
        y_cal = np.array(y_cal)
        y_pred_cal = np.array(y_pred_cal)
        
        residuals = np.abs(y_cal - y_pred_cal)
        n = len(residuals)
        
        # Calculate the quantile
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0) # Ensure it doesn't exceed 1.0
        
        self.q_hat = np.quantile(residuals, q_level, method='higher')

    def predict_interval(self, y_pred: float, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Predicts an interval for a single point.
        """
        if self.q_hat is None:
            raise ValueError("ConformalPredictor must be calibrated before use.")
        
        lower = float(y_pred - self.q_hat)
        upper = float(y_pred + self.q_hat)
        return (lower, upper)

    def batch_predict_intervals(self, y_preds: Union[np.ndarray, list, pd.Series]) -> pd.DataFrame:
        """
        Predicts intervals for a batch of predictions.
        """
        if self.q_hat is None:
            raise ValueError("ConformalPredictor must be calibrated before use.")
        
        y_preds = np.array(y_preds)
        lower = y_preds - self.q_hat
        upper = y_preds + self.q_hat
        
        return pd.DataFrame({
            'y_pred': y_preds,
            'lower_bound': lower,
            'upper_bound': upper
        })

class BootstrapUncertainty:
    def __init__(self):
        self.residuals = None

    def fit(self, residuals: Union[np.ndarray, list, pd.Series]) -> None:
        """
        Fits the residual distribution.
        """
        self.residuals = np.array(residuals)

    def predict_interval(self, y_pred: float, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Predicts an interval using the fitted residual distribution.
        """
        if self.residuals is None:
            raise ValueError("BootstrapUncertainty must be fitted before use.")
        
        alpha = 1.0 - confidence_level
        lower_q = alpha / 2.0
        upper_q = 1.0 - lower_q
        
        lower_bound = float(y_pred + np.quantile(self.residuals, lower_q))
        upper_bound = float(y_pred + np.quantile(self.residuals, upper_q))
        
        return (lower_bound, upper_bound)
