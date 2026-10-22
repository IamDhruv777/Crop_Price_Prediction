"""
XGBoost model for crop price forecasting.
Primary classical ML benchmark.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import joblib

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: list = None) -> None:
        if self.model is None:
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.feature_names = feature_names

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50) -> dict:
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter tuning.")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            preds = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, preds)
            return rmse

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.model = xgb.XGBRegressor(**best_params, random_state=42)
        return best_params

    def get_feature_importance(self) -> pd.Series:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model must be fitted with feature names.")
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)

    def explain_with_shap(self, X_sample: np.ndarray):
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required for explaining the model.")
        if self.model is None:
            raise RuntimeError("Model must be fitted before explaining.")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X_sample)
        return shap_values

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
