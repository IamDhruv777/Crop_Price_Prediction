"""
Configuration loader — reads config.yaml, crops.yaml, regions.yaml, models.yaml
and merges with environment variables.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load .env on import
load_dotenv()


CONFIG_DIR = Path(__file__).parents[2] / "config"


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from the config directory."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Config:
    """Centralised configuration object.
    
    Merges YAML files and environment variables.
    Environment variables take precedence.
    """

    def __init__(self) -> None:
        self._main = _load_yaml("config.yaml")
        self._crops = _load_yaml("crops.yaml")
        self._regions = _load_yaml("regions.yaml")
        self._models = _load_yaml("models.yaml")

    # --- Project ----------------------------------------------------------
    @property
    def project_name(self) -> str:
        return self._main["project"]["name"]

    @property
    def timezone(self) -> str:
        return self._main["project"]["timezone"]

    @property
    def random_seed(self) -> int:
        return int(os.getenv("RANDOM_SEED", self._main["project"]["random_seed"]))

    # --- API Keys ---------------------------------------------------------
    @property
    def data_gov_api_key(self) -> str:
        key = os.getenv("DATA_GOV_API_KEY", "")
        if not key:
            raise ValueError("DATA_GOV_API_KEY not set in environment / .env")
        return key

    @property
    def mandi_resource_id(self) -> str:
        return os.getenv(
            "MANDI_RESOURCE_ID",
            self._main["data_sources"]["mandi_prices"]["resource_id"],
        )

    # --- Database ---------------------------------------------------------
    @property
    def database_url(self) -> str:
        url = os.getenv("DATABASE_URL")
        if url:
            return url
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "crop_price_prediction")
        user = os.getenv("POSTGRES_USER", "cropuser")
        password = os.getenv("POSTGRES_PASSWORD", "")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    # --- Paths ------------------------------------------------------------
    @property
    def data_raw(self) -> Path:
        return Path(os.getenv("DATA_RAW_PATH", self._main["paths"]["data_raw"]))

    @property
    def data_interim(self) -> Path:
        return Path(os.getenv("DATA_INTERIM_PATH", self._main["paths"]["data_interim"]))

    @property
    def data_processed(self) -> Path:
        return Path(os.getenv("DATA_PROCESSED_PATH", self._main["paths"]["data_processed"]))

    @property
    def data_features(self) -> Path:
        return Path(os.getenv("DATA_FEATURES_PATH", self._main["paths"]["data_features"]))

    @property
    def models_path(self) -> Path:
        return Path(os.getenv("MODEL_REGISTRY_PATH", self._main["paths"]["models"]))

    @property
    def logs_path(self) -> Path:
        return Path(self._main["paths"]["logs"])

    @property
    def reports_path(self) -> Path:
        return Path(self._main["paths"]["reports"])

    # --- Data sources -----------------------------------------------------
    @property
    def nasa_power_url(self) -> str:
        return os.getenv(
            "NASA_POWER_BASE_URL",
            self._main["data_sources"]["weather"]["source_url"],
        )

    @property
    def weather_parameters(self) -> list:
        return self._main["data_sources"]["weather"]["parameters"]

    # --- Spatial ----------------------------------------------------------
    @property
    def k_neighbors(self) -> int:
        return self._main["spatial"]["k_neighbors"]

    @property
    def edge_weight_sigma(self) -> float:
        return self._main["spatial"]["edge_weight_sigma"]

    @property
    def max_distance_km(self) -> float:
        return self._main["spatial"]["max_distance_km"]

    # --- Forecast ---------------------------------------------------------
    @property
    def forecast_horizons(self) -> list:
        return self._main["forecast"]["horizons"]

    @property
    def confidence_level(self) -> float:
        return self._main["forecast"]["confidence_level"]

    # --- Crops ------------------------------------------------------------
    @property
    def default_crops(self) -> list:
        return self._main["crops"]["default"]

    @property
    def crops_config(self) -> dict:
        return self._crops["crops"]

    def get_crop_aliases(self, crop_name: str) -> list:
        """Return all known aliases for a crop."""
        cfg = self._crops["crops"].get(crop_name, {})
        aliases = cfg.get("aliases", [])
        return [crop_name] + aliases

    # --- Regions ----------------------------------------------------------
    @property
    def default_state(self) -> str:
        return self._main["region"]["default_state"]

    @property
    def regions_config(self) -> dict:
        return self._regions["regions"]

    # --- Models -----------------------------------------------------------
    @property
    def models_config(self) -> dict:
        return self._models

    @property
    def enabled_models(self) -> dict:
        return self._models["enabled_models"]

    @property
    def selection_metric(self) -> str:
        return self._models["selection_metric"]

    # --- Walk-forward validation ------------------------------------------
    @property
    def walk_forward_config(self) -> dict:
        return self._main["walk_forward"]

    # --- Data split -------------------------------------------------------
    @property
    def data_split(self) -> dict:
        return self._main["data_split"]

    # --- Features ---------------------------------------------------------
    @property
    def price_lags(self) -> list:
        return self._main["features"]["price_lags"]

    @property
    def rolling_windows(self) -> list:
        return self._main["features"]["rolling_windows"]

    @property
    def arrival_lags(self) -> list:
        return self._main["features"]["arrival_lags"]

    @property
    def weather_lags(self) -> list:
        return self._main["features"]["weather_lags"]

    # --- Outlier detection ------------------------------------------------
    @property
    def outlier_config(self) -> dict:
        return self._main["outlier_detection"]

    # --- Alerts -----------------------------------------------------------
    @property
    def alert_config(self) -> dict:
        return self._main["alerts"]

    # --- Generic access ---------------------------------------------------
    def get(self, *keys, default=None):
        """Dot-path access into the main config dict."""
        obj = self._main
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k, default)
            else:
                return default
        return obj


# Singleton
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the singleton Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
