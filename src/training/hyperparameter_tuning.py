import os
import json
import logging
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class HyperparameterTuner:
    def __init__(self, config, direction='minimize', metric='rmse'):
        self.config = config
        self.direction = direction
        self.metric = metric
        self.logger = logging.getLogger(__name__)

    def _get_metric_func(self):
        if self.metric == 'rmse':
            return lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
        elif self.metric == 'mae':
            return mean_absolute_error
        else:
            raise ValueError(f"Unknown metric {self.metric}")

    def tune_random_forest(self, X_train, y_train, X_val, y_val, n_trials=30) -> dict:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            metric_val = self._get_metric_func()(y_val, preds)
            return metric_val

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def tune_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50) -> dict:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            preds = model.predict(X_val)
            metric_val = self._get_metric_func()(y_val, preds)
            return metric_val

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def tune_lstm(self, train_data, val_data, n_trials=20) -> dict:
        # Placeholder for LSTM tuning
        self.logger.info("Tuning LSTM (placeholder)")
        return {'hidden_size': 64, 'num_layers': 2, 'learning_rate': 0.001}

    def tune_gru(self, train_data, val_data, n_trials=20) -> dict:
        # Placeholder for GRU tuning
        self.logger.info("Tuning GRU (placeholder)")
        return {'hidden_size': 64, 'num_layers': 2, 'learning_rate': 0.001}

    def save_study_results(self, study: optuna.Study, model_name: str, path: str):
        os.makedirs(path, exist_ok=True)
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': len(study.trials)
        }
        file_path = os.path.join(path, f"{model_name}_optuna_study.json")
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Saved {model_name} study results to {file_path}")

    def load_best_params(self, model_name: str, path: str) -> dict:
        file_path = os.path.join(path, f"{model_name}_optuna_study.json")
        if not os.path.exists(file_path):
            self.logger.warning(f"No study results found at {file_path}")
            return {}
        with open(file_path, 'r') as f:
            results = json.load(f)
        return results.get('best_params', {})
