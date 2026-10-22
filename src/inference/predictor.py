"""
Inference engine for crop price prediction.
Loads trained models and generates predictions with uncertainty.
Never retrains during inference.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class CropPricePredictor:
    def __init__(self, config: Dict[str, Any], db_session: Any):
        self.config = config
        self.db_session = db_session
        self.models_cache = {}

    def load_model(self, crop: str, market_id: str, horizon: int) -> Tuple[Any, Dict]:
        """
        Loads best trained model from registry, with fallback.
        """
        model_key = f"{crop}_{market_id}_{horizon}"
        if model_key in self.models_cache:
            return self.models_cache[model_key]
        
        # Simulate loading a model and metadata
        class DummyModel:
            def predict(self, X):
                return np.random.uniform(1000, 3000, len(X))
        
        primary_model = DummyModel()
        metadata = {
            "model_version": "1.0.0",
            "model_type": "xgboost",
            "rmse": 150.0,
            "mae": 110.0,
            "q_hat": 200.0
        }
        
        self.models_cache[model_key] = (primary_model, metadata)
        return primary_model, metadata

    def get_model_fallback(self, primary_model: str, crop: str, market_id: str, horizon: int) -> str:
        """
        Gets fallback model if primary is not available.
        """
        return f"fallback_model_{crop}_{horizon}"

    def _prepare_features(self, market_id: str, crop: str, as_of_date: str) -> pd.DataFrame:
        """
        Loads and prepares features.
        """
        return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

    def _compute_uncertainty(self, y_pred_array: np.ndarray, model_run: Dict) -> List[Dict]:
        """
        Computes uncertainty intervals.
        """
        q_hat = model_run.get("q_hat", 150.0)
        uncertainty = []
        for pred in y_pred_array:
            uncertainty.append({
                "lower_bound": float(pred - q_hat),
                "upper_bound": float(pred + q_hat)
            })
        return uncertainty

    def _format_response(self, predictions: List[float], uncertainty: List[Dict], metadata: Dict) -> Dict:
        """
        Formats the prediction response.
        """
        return {
            "model": metadata.get("model_type", "unknown"),
            "model_version": metadata.get("model_version", "unknown"),
            "rmse": metadata.get("rmse", 0.0),
            "mae": metadata.get("mae", 0.0),
            "predictions": [
                {
                    "predicted_price": pred,
                    "lower_bound": unc["lower_bound"],
                    "upper_bound": unc["upper_bound"]
                }
                for pred, unc in zip(predictions, uncertainty)
            ]
        }

    def predict(self, state: str, district: str, market: str, crop: str, horizon: int, variety: Optional[str] = None) -> Dict:
        """
        Main predict method.
        """
        if horizon not in [1, 7, 30]:
            raise ValueError("Horizon must be 1, 7, or 30.")
            
        X = self._prepare_features(market, crop, "today")
        model, metadata = self.load_model(crop, market, horizon)
        
        X_horizon = X.iloc[:horizon] if len(X) >= horizon else pd.concat([X]*horizon).iloc[:horizon]
        y_pred = model.predict(X_horizon)
        
        uncertainty = self._compute_uncertainty(y_pred, metadata)
        response = self._format_response(list(y_pred), uncertainty, metadata)
        
        response.update({
            "state": state,
            "district": district,
            "market": market,
            "crop": crop,
            "horizon": horizon,
            "confidence_level": 0.95
        })
        
        return response
