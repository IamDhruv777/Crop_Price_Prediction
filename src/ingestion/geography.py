import os
import re
import json
import time
import logging
import pandas as pd
import requests
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

logger = logging.getLogger(__name__)

class GeographyIngestion:
    def __init__(self):
        self.raw_dir = Path("data/raw/geography")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = Path("config")
        self.cache_file = self.raw_dir / "geocoding_cache.json"
        self.master_file = self.raw_dir / "market_master.csv"
        self.markets_csv = self.raw_dir / "markets.csv"
        self.regions_yaml = self.config_dir / "regions.yaml"
        
        self.geocache = self._load_geocache()
        self.last_geocode_time = 0.0

    def _load_geocache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_geocache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.geocache, f, indent=2)

    def _rate_limit_geocoding(self):
        elapsed = time.time() - self.last_geocode_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self.last_geocode_time = time.time()

    def generate_market_id(self, state: str, district: str, market_name: str) -> str:
        state_code = re.sub(r'[^A-Z0-9]', '_', str(state).upper().strip())
        district_code = re.sub(r'[^A-Z0-9]', '_', str(district).upper().strip())
        market_code = re.sub(r'[^A-Z0-9]', '_', str(market_name).upper().strip())
        return f"{state_code[:5]}_{district_code[:5]}_{market_code}"

    def geocode_market(self, market_name: str, district: str, state: str) -> Dict[str, float]:
        query = f"{market_name}, {district}, {state}, India"
        if query in self.geocache:
            logger.info(f"Geocoding cache hit for: {query}")
            return self.geocache[query]

        self._rate_limit_geocoding()
        logger.info(f"Geocoding via Nominatim: {query}")
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "CropPricePredictionBot/1.0"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            result = {"lat": None, "lon": None}
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                if (6.0 <= lat <= 36.0) and (68.0 <= lon <= 98.0):
                    result = {"lat": lat, "lon": lon}
                else:
                    logger.warning(f"Coordinates outside India for {query}: {lat}, {lon}")
                    
            self.geocache[query] = result
            self._save_geocache()
            return result
            
        except Exception as e:
            logger.error(f"Geocoding failed for {query}: {e}")
            return {"lat": None, "lon": None}

    def build_market_master(self) -> pd.DataFrame:
        markets_data = []
        
        if self.regions_yaml.exists():
            with open(self.regions_yaml, 'r') as f:
                regions = yaml.safe_load(f)
                if regions and 'markets' in regions:
                    for m in regions['markets']:
                        markets_data.append(m)
        
        if self.markets_csv.exists():
            df_csv = pd.read_csv(self.markets_csv)
            markets_data.extend(df_csv.to_dict('records'))
            
        if not markets_data:
            logger.warning("No market data sources found.")
            return pd.DataFrame()
            
        df = pd.DataFrame(markets_data)
        
        processed_records = []
        for _, row in df.iterrows():
            state = row.get('state', '')
            district = row.get('district', '')
            market = row.get('market_name', '')
            
            market_id = row.get('market_id') or self.generate_market_id(state, district, market)
            lat = row.get('latitude')
            lon = row.get('longitude')
            
            if pd.isna(lat) or pd.isna(lon):
                coords = self.geocode_market(market, district, state)
                lat = coords['lat']
                lon = coords['lon']
                
            processed_records.append({
                "market_id": market_id,
                "market_name": market,
                "district": district,
                "state": state,
                "latitude": lat,
                "longitude": lon,
                "region_id": row.get('region_id', 'UNKNOWN')
            })
            
        master_df = pd.DataFrame(processed_records)
        master_df.to_csv(self.master_file, index=False)
        logger.info(f"Saved market master to {self.master_file}")
        
        if HAS_GEOPANDAS and not master_df.empty:
            valid_coords = master_df.dropna(subset=['latitude', 'longitude'])
            if not valid_coords.empty:
                gdf = gpd.GeoDataFrame(
                    master_df, 
                    geometry=gpd.points_from_xy(master_df.longitude, master_df.latitude),
                    crs="EPSG:4326"
                )
                return gdf
                
        return master_df
