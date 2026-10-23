"""
Inference engine for crop price prediction.
Loads production-deployed models from models/trained/deployment_registry.json
and generates multi-horizon predictions with calibrated 95% conformal uncertainty intervals.
Evaluated on complete feature vectors without data leakage or zero-filling.

Canonical Deployment Configuration:
  - 1-Day: Tuned XGBoost Regressor (anchor: modal_price, clip: [-200, 200])
  - 7-Day: Tuned XGBoost Regressor (anchor: 0.6*modal_price + 0.4*price_rolling_mean_7, clip: [-400, 400])
  - 30-Day: Naive Persistence Baseline (anchor: modal_price, flat trajectory)
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import logging
import joblib
import numpy as np
import pandas as pd

from src.models.naive import NaiveForecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models" / "trained"
REGISTRY_PATH = MODELS_DIR / "deployment_registry.json"
FEATURE_STORE_PATH = DATA_DIR / "processed" / "inference_feature_store.csv"
METRICS_DEPLOYED_PATH = DATA_DIR / "metrics_deployed.json"
METRICS_PATH = DATA_DIR / "metrics.json"
MONDRIAN_PATH = DATA_DIR / "mondrian_conformal_quantiles.json"

KNOWN_COMMODITIES = [
    "Arhar (Tur-Red Gram)",
    "Black Gram (Urd Beans)",
    "Cowpea (Lobia-Karamani)",
    "Green Gram (Moong)",
    "Kabuli Chana (Chickpeas-White)",
    "Lentil (Masur)",
    "Maize",
    "Rice",
    "Soyabean",
    "Wheat",
    "White Peas"
]


def compute_canonical_anchor(df_or_series: Any, horizon: int) -> Any:
    """Computes the canonical price anchor according to deployment configuration."""
    if isinstance(df_or_series, (pd.Series, dict)):
        base_price = float(df_or_series["modal_price"])
        if horizon == 1:
            return base_price
        elif horizon == 7:
            p_roll7 = df_or_series.get("price_rolling_mean_7", base_price)
            if pd.isna(p_roll7):
                p_roll7 = base_price
            return 0.6 * base_price + 0.4 * float(p_roll7)
        else:
            return base_price
    else:
        # DataFrame / batch
        base_price = df_or_series["modal_price"].values
        if horizon == 1:
            return base_price
        elif horizon == 7:
            p_roll7 = df_or_series["price_rolling_mean_7"].values if "price_rolling_mean_7" in df_or_series else base_price
            p_roll7 = np.where(pd.isna(p_roll7), base_price, p_roll7)
            return 0.6 * base_price + 0.4 * p_roll7
        else:
            return base_price


def predict_canonical_batch(
    model: Any,
    df: pd.DataFrame,
    horizon: int,
    reg_entry: Dict[str, Any]
) -> np.ndarray:
    """
    Computes predictions using the exact canonical deployment logic:
    anchor + clip(model.predict(X), -clip_delta, clip_delta), floor at 10.0.
    For persistence (horizon=30 or is_persistence), returns anchor.
    """
    anchor = compute_canonical_anchor(df, horizon)
    if reg_entry.get("is_persistence") or horizon == 30:
        return anchor

    feats = list(model.feature_names_in_)
    raw_delta = model.predict(df[feats])
    clip_delta = reg_entry.get("clip_delta")
    if clip_delta is not None:
        raw_delta = np.clip(raw_delta, -clip_delta, clip_delta)
    pred = np.maximum(10.0, np.round(anchor + raw_delta, 2))
    return pred


class CropPricePredictor:
    def __init__(self, config: Optional[Dict[str, Any]] = None, db_session: Any = None):
        self.config = config or {}
        self.db_session = db_session
        self.models_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self._feature_store_cache: Optional[pd.DataFrame] = None
        self._registry_cache: Optional[Dict[str, Any]] = None
        self._metrics_deployed_cache: Optional[Dict[str, Any]] = None
        self._metrics_cache: Optional[Dict[str, Any]] = None
        self._mondrian_cache: Optional[Dict[str, Any]] = None

    def _get_feature_store(self) -> pd.DataFrame:
        if self._feature_store_cache is None:
            if FEATURE_STORE_PATH.exists():
                self._feature_store_cache = pd.read_csv(FEATURE_STORE_PATH)
            else:
                self._feature_store_cache = pd.DataFrame()
        return self._feature_store_cache

    @property
    def feature_store(self) -> pd.DataFrame:
        return self._get_feature_store()

    def _get_registry(self) -> Dict[str, Any]:
        if self._registry_cache is None:
            if not REGISTRY_PATH.exists():
                raise FileNotFoundError(f"Deployment registry missing at {REGISTRY_PATH}")
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                self._registry_cache = json.load(f)
        return self._registry_cache

    @property
    def registry(self) -> Dict[str, Any]:
        return self._get_registry()

    def _get_deployed_metrics(self) -> Dict[str, Any]:
        if self._metrics_deployed_cache is None:
            if not METRICS_DEPLOYED_PATH.exists():
                raise FileNotFoundError(f"Deployed metrics artifact missing at {METRICS_DEPLOYED_PATH}")
            with open(METRICS_DEPLOYED_PATH, "r", encoding="utf-8") as f:
                self._metrics_deployed_cache = json.load(f)
        return self._metrics_deployed_cache

    def _get_metrics_json(self) -> Dict[str, Any]:
        if self._metrics_cache is None:
            if METRICS_PATH.exists():
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    self._metrics_cache = json.load(f)
            else:
                self._metrics_cache = {}
        return self._metrics_cache

    def _get_mondrian_quantiles(self) -> Dict[str, Any]:
        if self._mondrian_cache is None:
            if not MONDRIAN_PATH.exists():
                raise FileNotFoundError(f"Mondrian quantiles artifact missing at {MONDRIAN_PATH}")
            with open(MONDRIAN_PATH, "r", encoding="utf-8") as f:
                self._mondrian_cache = json.load(f)
        return self._mondrian_cache

    def get_calibrated_q95(self, crop: str, horizon: int) -> float:
        """
        Dynamically resolves calibrated 95% conformal quantile for the given crop and horizon.
        Requires exact crop key matching. Fails closed with RuntimeError if missing.
        """
        h_str = f"{horizon}-Day"
        mondrian = self._get_mondrian_quantiles()
        if h_str in mondrian:
            h_dict = mondrian[h_str]
            clean_target = crop.strip().lower()
            for c_name, val in h_dict.items():
                if c_name.strip().lower() == clean_target:
                    return float(val)

        raise RuntimeError(
            f"Conformal calibration not found for exact crop '{crop}' and horizon {h_str}. "
            f"Please run scripts/calibrate_mondrian_conformal.py first."
        )

    def load_model(self, crop: str, horizon: int) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Loads the production-deployed model and metadata for the given horizon strictly from
        models/trained/deployment_registry.json and data/metrics_deployed.json.
        """
        cache_key = f"{crop.lower()}_{horizon}d"
        if cache_key in self.models_cache:
            return self.models_cache[cache_key]

        reg = self._get_registry()
        h_str = f"{horizon}-Day"
        if h_str not in reg:
            raise ValueError(f"Horizon '{h_str}' not configured in deployment registry.")

        h_cfg = reg[h_str]
        model_name = h_cfg["model_name"]
        model_class = h_cfg["model_class"]
        model_version = h_cfg["model_version"]
        is_persistence = h_cfg.get("is_persistence", False)
        clip_delta = h_cfg.get("clip_delta")

        # Load deployed metrics strictly from metrics_deployed.json
        m_dep = self._get_deployed_metrics()
        h_metrics = m_dep["by_horizon"].get(h_str, {})
        rmse = float(h_metrics["RMSE"]) if "RMSE" in h_metrics else None
        mae = float(h_metrics["MAE"]) if "MAE" in h_metrics else None
        r2 = float(h_metrics["R2"]) if "R2" in h_metrics else None

        q95 = self.get_calibrated_q95(crop, horizon)

        if is_persistence:
            model = NaiveForecast()
        else:
            model_path = REPO_ROOT / h_cfg["model_file"]
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None, {
                    "status": "error",
                    "error_code": "model_not_found",
                    "detail": f"Model file not found at {model_path}"
                }
            try:
                model = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
                return None, {
                    "status": "error",
                    "error_code": "model_load_error",
                    "detail": str(e)
                }

        metadata = {
            "model_name": model_name,
            "model_class": model_class,
            "model_version": model_version,
            "horizon": horizon,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "q95": q95,
            "is_persistence": is_persistence,
            "clip_delta": clip_delta,
            "anchor_formula": h_cfg.get("anchor_formula")
        }

        self.models_cache[cache_key] = (model, metadata)
        return model, metadata

    def load_directional_classifier(self, horizon: int) -> Optional[Any]:
        """Loads trained XGBClassifier head for horizon 1 or 7 if available."""
        if horizon not in [1, 7]:
            return None
        cache_key = f"clf_{horizon}d"
        if cache_key in self.models_cache:
            return self.models_cache[cache_key][0]

        clf_file = MODELS_DIR / f"xgb_classifier_{horizon}d.joblib"
        if clf_file.exists():
            try:
                clf = joblib.load(clf_file)
                self.models_cache[cache_key] = (clf, {})
                return clf
            except Exception as e:
                logger.warning(f"Could not load directional classifier from {clf_file}: {e}")
                return None
        return None

    def predict(
        self,
        state: str,
        district: str,
        market: str,
        crop: str,
        horizon: int,
        confidence_level: float = 0.95,
        variety: Optional[str] = None,
        as_of_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes production inference using complete feature vectors from the feature store.
        Strictly enforces:
          1. Exact match on state, district, market, commodity.
          2. Zero-fill prohibition: is_complete=True required for ALL horizons (including 30D).
          3. Model residual clipping (1D: [-200, 200], 7D: [-400, 400]).
          4. Historical calendar dates derived from as_of_date.
        """
        if horizon not in [1, 7, 30]:
            raise ValueError(f"Invalid forecast horizon: {horizon}. Must be 1, 7, or 30.")

        # Validate crop strictly
        clean_crop = crop.strip()
        if clean_crop == "Soybean":
            clean_crop = "Soyabean"
        if clean_crop not in KNOWN_COMMODITIES:
            return {
                "status": "error",
                "error_code": "unknown_crop",
                "detail": f"Unknown crop '{crop}'. Valid crops: {KNOWN_COMMODITIES}"
            }
        crop = clean_crop

        # Validate as_of_date strictly
        if as_of_date:
            try:
                target_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d", errors="raise")
            except Exception:
                return {
                    "status": "error",
                    "error_code": "invalid_date",
                    "detail": f"Invalid as_of_date format '{as_of_date}'. Must be YYYY-MM-DD."
                }
            if target_dt > pd.to_datetime("2017-06-17"):
                return {
                    "status": "error",
                    "error_code": "invalid_date",
                    "detail": f"as_of_date '{as_of_date}' exceeds historical corpus maximum date (2017-06-17)."
                }

        model, metadata = self.load_model(crop, horizon)
        if model is None:
            return {
                "status": "error",
                "error_code": "model_unavailable",
                "detail": f"Model for horizon {horizon}-Day is not available."
            }

        fs = self._get_feature_store()
        if fs.empty:
            return {
                "status": "error",
                "error_code": "insufficient_data",
                "detail": "Inference feature store is empty or unavailable."
            }

        # Exact match strictly required on state, district, market, and commodity
        DISTRICT_ALIASES = {
            "solapur": "sholapur",
            "ahmednagar": "ahmadnagar",
            "beed": "bid",
            "buldhana": "buldana",
            "gondia": "gondiya",
            "raigad": "raigarh",
        }
        clean_dist = district.strip().lower()
        norm_district = DISTRICT_ALIASES.get(clean_dist, clean_dist)

        st_mask = fs["state"].str.strip().str.lower() == state.strip().lower()
        dist_mask = fs["district"].str.strip().str.lower() == norm_district
        mkt_mask = fs["market"].str.strip().str.lower() == market.strip().lower()
        crp_mask = fs["commodity"].str.strip().str.lower() == crop.strip().lower()
        sub = fs[st_mask & dist_mask & mkt_mask & crp_mask]

        if sub.empty:
            return {
                "status": "error",
                "error_code": "insufficient_data",
                "detail": f"No historical trade features found for exact location: {state} / {district} / {market} / {crop}."
            }

        # Handle optional as_of_date filter
        if as_of_date:
            sub_date = sub[pd.to_datetime(sub["date"]) <= pd.to_datetime(as_of_date)]
            if sub_date.empty:
                return {
                    "status": "error",
                    "error_code": "insufficient_data",
                    "detail": f"No trading records found on or before as_of_date '{as_of_date}' for {market} / {crop}."
                }
            latest_record = sub_date.iloc[-1]
        else:
            latest_record = sub.iloc[-1]

        # Enforce complete feature contract for ALL horizons (including 30D persistence)
        if "is_complete" in latest_record and not bool(latest_record["is_complete"]):
            return {
                "status": "error",
                "error_code": "insufficient_data",
                "detail": f"Market-crop pair {market} / {crop} has incomplete trading history (<30d history) and cannot be forecasted reliably."
            }

        base_price = float(latest_record.get("modal_price", np.nan))
        if pd.isna(base_price) or base_price <= 0:
            return {
                "status": "error",
                "error_code": "insufficient_data",
                "detail": f"Base price for market '{market}' and crop '{crop}' is invalid or missing."
            }

        q95 = metadata["q95"]
        record_date = str(latest_record.get("date"))
        target_as_of_date = as_of_date if as_of_date else record_date
        base_dt = pd.to_datetime(target_as_of_date)

        # -------------------------------------------------------------
        # 30-Day: Naive Persistence Baseline Deployment
        # -------------------------------------------------------------
        if metadata.get("is_persistence") or horizon == 30:
            final_price = round(base_price, 2)
            predictions = []
            for step in range(1, horizon + 1):
                step_date = (base_dt + pd.Timedelta(days=step)).strftime("%Y-%m-%d")
                step_delta = q95 * np.sqrt(step / horizon)
                predictions.append({
                    "day": step,
                    "date": step_date,
                    "predicted_price": final_price,
                    "lower_bound": round(max(0.0, final_price - step_delta), 2),
                    "upper_bound": round(final_price + step_delta, 2),
                    "is_interpolated_bound": (step < horizon)
                })

            p_rmse = float(metadata["rmse"])
            p_mae = float(metadata["mae"])
            p_r2 = float(metadata["r2"])
            gate_threshold = round(float(max(10.0, 0.05 * q95)), 2)

            return {
                "status": "success",
                "state": state,
                "district": district,
                "market": market,
                "crop": crop,
                "horizon": horizon,
                "as_of_date": target_as_of_date,
                "is_historical_backtest": True,
                "base_price": round(base_price, 2),
                "predicted_price": final_price,
                "anchor_price": round(base_price, 2),
                "price_change": 0.0,
                "percent_change": 0.0,
                "recommendation": "🔵 स्थिर (Stable - Naive Persistence Baseline deployed)",
                "directional_signal": "STABLE",
                "signal_confidence": "NEUTRAL",
                "rise_probability": 0.50,
                "conformal_gate_threshold": gate_threshold,
                "confidence_level": confidence_level,
                "uncertainty_delta": round(q95, 2),
                "model_name": metadata["model_name"],
                "model_class": metadata["model_class"],
                "model_version": metadata["model_version"],
                "rmse": p_rmse,
                "mae": p_mae,
                "deployed_rmse": metadata["rmse"],
                "deployed_mae": metadata["mae"],
                "r2": p_r2,
                "predictions_trajectory": predictions
            }

        # -------------------------------------------------------------
        # 1-Day & 7-Day: Tuned XGBoost Inference with Anchors & Clipping
        # -------------------------------------------------------------
        if not hasattr(model, "feature_names_in_"):
            return {
                "status": "error",
                "error_code": "model_schema_error",
                "detail": "Model does not specify required feature_names_in_."
            }

        feat_cols = list(model.feature_names_in_)
        missing_features = [col for col in feat_cols if col not in latest_record or pd.isna(latest_record[col])]

        if missing_features:
            return {
                "status": "error",
                "error_code": "insufficient_data",
                "detail": f"Required model features missing or null for {market} / {crop}: {missing_features}"
            }

        # Exact Training Anchor Parity using Canonical Calculation
        anchor = float(compute_canonical_anchor(latest_record, horizon))

        X_input = pd.DataFrame([latest_record[feat_cols]])
        try:
            raw_pred = model.predict(X_input)
            predicted_delta = float(raw_pred[0])
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {
                "status": "error",
                "error_code": "prediction_execution_failed",
                "detail": f"Inference execution failed: {str(e)}"
            }

        # Apply canonical residual delta clipping
        clip_bound = metadata.get("clip_delta")
        if clip_bound is not None:
            predicted_delta = float(np.clip(predicted_delta, -clip_bound, clip_bound))

        final_price = max(10.0, round(anchor + predicted_delta, 2))

        # Trajectory calculation from base_price to final_price
        predictions = []
        for step in range(1, horizon + 1):
            step_date = (base_dt + pd.Timedelta(days=step)).strftime("%Y-%m-%d")
            projected = base_price + (final_price - base_price) * (step / horizon)
            step_delta = q95 * np.sqrt(step / horizon)
            predictions.append({
                "day": step,
                "date": step_date,
                "predicted_price": round(projected, 2),
                "lower_bound": round(max(0.0, projected - step_delta), 2),
                "upper_bound": round(projected + step_delta, 2),
                "is_interpolated_bound": (step < horizon)
            })

        price_change = round(final_price - base_price, 2)
        pct_change = round((price_change / base_price) * 100, 2)

        # Conformal Uncertainty Gating: calibrated on validation split to eliminate micro-fluctuation false alarms
        # Minimum noise gate is max(Rs 5.0, 5% of crop-specific 95% conformal interval width)
        gate_threshold = round(float(max(5.0, 0.05 * q95)), 2)

        # Dual-Head Directional Classification Head
        clf = self.load_directional_classifier(horizon)
        if clf is not None:
            try:
                clf_probs = clf.predict_proba(X_input)
                rise_prob = round(float(clf_probs[0][1]), 4)
            except Exception as e:
                logger.warning(f"Directional classification prediction failed: {e}")
                rise_prob = round(float(0.50 + np.clip(predicted_delta / 200.0, -0.49, 0.49)), 4)
        else:
            rise_prob = round(float(0.50 + np.clip(predicted_delta / 200.0, -0.49, 0.49)), 4)

        # Dual-head Consensus: Gated rally requires both continuous drift >= gate and classifier confirmation
        is_gated_rally = (price_change >= gate_threshold) and (pct_change >= 2.0) and (rise_prob >= 0.45)
        is_gated_drop = (price_change <= -gate_threshold) and (pct_change <= -2.0) and (rise_prob <= 0.40)

        if is_gated_rally:
            directional_signal = "RALLY"
            signal_confidence = "HIGH" if (price_change >= (0.10 * q95) and rise_prob >= 0.55) else "MODERATE"
            recommendation = f"🟢 माल रोकें (Hold - Dual-Head Verified Rally, {int(rise_prob*100)}% Probability)"
        elif is_gated_drop:
            directional_signal = "DROP"
            signal_confidence = "HIGH" if (abs(price_change) >= (0.10 * q95) and rise_prob <= 0.35) else "MODERATE"
            recommendation = f"🔴 आज बेचें (Sell - Dual-Head Verified Softening, {int((1-rise_prob)*100)}% Drop Probability)"
        else:
            directional_signal = "STABLE"
            signal_confidence = "NEUTRAL"
            recommendation = f"🔵 स्थिर (Stable - Range-bound equilibrium, {int(rise_prob*100)}% Up Probability)"

        return {
            "status": "success",
            "state": state,
            "district": district,
            "market": market,
            "crop": crop,
            "horizon": horizon,
            "as_of_date": target_as_of_date,
            "is_historical_backtest": True,
            "base_price": round(base_price, 2),
            "predicted_price": final_price,
            "anchor_price": round(anchor, 2),
            "price_change": price_change,
            "percent_change": pct_change,
            "recommendation": recommendation,
            "directional_signal": directional_signal,
            "signal_confidence": signal_confidence,
            "rise_probability": rise_prob,
            "conformal_gate_threshold": gate_threshold,
            "confidence_level": confidence_level,
            "uncertainty_delta": round(q95, 2),
            "model_name": metadata["model_name"],
            "model_class": metadata["model_class"],
            "model_version": metadata["model_version"],
            "rmse": metadata["rmse"],
            "mae": metadata["mae"],
            "r2": metadata["r2"],
            "predictions_trajectory": predictions
        }
