import os
import json
import time
import logging
import hashlib
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class WeatherIngestion:
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.parameters = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS10M"]
        self.raw_dir = Path("data/raw/weather")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0.0
        
        self.param_mapping = {
            "T2M": "temperature",
            "T2M_MAX": "temperature_max",
            "T2M_MIN": "temperature_min",
            "PRECTOTCORR": "rainfall",
            "RH2M": "humidity",
            "WS10M": "wind_speed"
        }

    def _build_cache_key(self, lat: float, lon: float, start_date: str, end_date: str) -> str:
        key_str = f"{lat:.3f}_{lon:.3f}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_filename(self, lat: float, lon: float, start_date: str, end_date: str) -> Path:
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        return self.raw_dir / f"weather_{lat:.3f}_{lon:.3f}_{start_clean}_{end_clean}.json"

    def _load_cache(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        filepath = self._get_filename(lat, lon, start_date, end_date)
        if filepath.exists():
            logger.info(f"Cache hit for weather data: {filepath}")
            with open(filepath, 'r') as f:
                data = json.load(f)
            return self._parse_response(data, lat, lon)
        return None

    def _save_cache(self, data: dict, lat: float, lon: float, start_date: str, end_date: str):
        filepath = self._get_filename(lat, lon, start_date, end_date)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved weather data to {filepath}")

    def _rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < 2.0:
            sleep_time = 2.0 - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _call_api(self, lat: float, lon: float, start_date: str, end_date: str) -> dict:
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        
        params = {
            "parameters": ",".join(self.parameters),
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start_clean,
            "end": end_clean,
            "format": "JSON"
        }
        
        retries = 3
        backoff = 2
        
        for attempt in range(retries):
            self._rate_limit()
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"NASA POWER API request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(backoff ** attempt)
                else:
                    logger.error("Max retries reached for NASA POWER API")
                    raise

    def _parse_response(self, response_json: dict, lat: float, lon: float) -> pd.DataFrame:
        if "properties" not in response_json or "parameter" not in response_json["properties"]:
            logger.error("Invalid response format from NASA POWER API")
            return pd.DataFrame()
            
        parameters_data = response_json["properties"]["parameter"]
        
        records = []
        first_param = list(parameters_data.keys())[0]
        dates = list(parameters_data[first_param].keys())
        
        for date_str in dates:
            record = {
                "date": datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d"),
                "latitude": lat,
                "longitude": lon
            }
            
            for nasa_param, mapped_col in self.param_mapping.items():
                val = parameters_data.get(nasa_param, {}).get(date_str)
                if val == -999 or val == -999.0:
                    val = None
                record[mapped_col] = val
                
            records.append(record)
            
        df = pd.DataFrame(records)
        return df

    def fetch_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching weather for lat={lat}, lon={lon}, dates={start_date} to {end_date}")
        
        cached_df = self._load_cache(lat, lon, start_date, end_date)
        if cached_df is not None:
            return cached_df
            
        response = self._call_api(lat, lon, start_date, end_date)
        self._save_cache(response, lat, lon, start_date, end_date)
        
        return self._parse_response(response, lat, lon)

    def fetch_weather_for_market(self, market_id: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.fetch_weather(lat, lon, start_date, end_date)
        if not df.empty:
            df['market_id'] = market_id
        return df
