import numpy as np
from typing import Tuple, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

class DataNormalizer:
    def _get_scaler_instance(self, scaler_type: str):
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def fit_scaler(self, X_train: np.ndarray, scaler_type: str = 'standard') -> Any:
        scaler = self._get_scaler_instance(scaler_type)
        scaler.fit(X_train)
        return scaler

    def transform(self, X: np.ndarray, scaler: Any) -> np.ndarray:
        return scaler.transform(X)

    def fit_transform_train(self, X_train: np.ndarray, scaler_type: str = 'standard') -> Tuple[np.ndarray, Any]:
        scaler = self._get_scaler_instance(scaler_type)
        X_scaled = scaler.fit_transform(X_train)
        return X_scaled, scaler

    def inverse_transform(self, X_scaled: np.ndarray, scaler: Any) -> np.ndarray:
        return scaler.inverse_transform(X_scaled)

    def save_scaler(self, scaler: Any, path: str):
        joblib.dump(scaler, path)

    def load_scaler(self, path: str) -> Any:
        return joblib.load(path)

    def get_feature_names(self, scaler: Any, original_names: List[str]) -> List[str]:
        if hasattr(scaler, 'get_feature_names_out'):
            return list(scaler.get_feature_names_out(original_names))
        return original_names
