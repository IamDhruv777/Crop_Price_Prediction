"""
Random Forest regressor for crop price forecasting.
Uses engineered features (price lags, rolling stats, weather, arrivals, temporal).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import joblib

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None) -> None:
        if self.model is None:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 30) -> dict:
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter tuning.")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, preds)
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.model = RandomForestRegressor(**best_params, random_state=42)
        return best_params

    def get_feature_importance(self) -> pd.Series:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model must be fitted with feature names to get feature importance.")
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        state = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str):
        state = joblib.load(path)
        instance = cls()
        instance.model = state['model']
        instance.feature_names = state['feature_names']
        return instance
