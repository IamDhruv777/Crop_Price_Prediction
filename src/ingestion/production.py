import os
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProductionIngestion:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = os.environ.get('DATA_GOV_API_KEY')
        self.resource_id = self.config.get('resource_id', '37233eb4-97c9-467f-945f-4d642e1cc714')
        self.base_url = f"https://api.data.gov.in/resource/{self.resource_id}"
        self.raw_dir = Path("data/raw/production")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, state: Optional[str] = None, crop: Optional[str] = None) -> pd.DataFrame:
        if not self.api_key:
            logger.warning("DATA_GOV_API_KEY not set. Cannot fetch production data.")
            return pd.DataFrame()

        params = {
            'api-key': self.api_key,
            'format': 'json',
            'limit': 1000,
            'offset': 0
        }
        
        if state:
            params['filters[state_name]'] = state
        if crop:
            params['filters[crop_name]'] = crop
            
        all_records = []
        total_records = float('inf')
        
        while params['offset'] < total_records:
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                records = data.get('records', [])
                if not records:
                    break
                    
                all_records.extend(records)
                total_records = data.get('total', 0)
                params['offset'] += params['limit']
                
                logger.info(f"Fetched {len(all_records)} / {total_records} production records")
                
            except Exception as e:
                logger.error(f"Failed to fetch production data: {e}")
                break
                
        df = pd.DataFrame(all_records)
        return df

    def process_and_validate(self, df: pd.DataFrame, prediction_year: int) -> pd.DataFrame:
        if df.empty:
            return df
            
        col_map = {
            'state_name': 'state',
            'district_name': 'district',
            'crop_name': 'crop',
            'season_name': 'season',
            'crop_year': 'year',
            'area': 'area',
            'production': 'production'
        }
        
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        required = ['state', 'district', 'crop', 'season', 'year', 'area', 'production']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column {col} in production data")
                df[col] = None
                
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        df['production'] = pd.to_numeric(df['production'], errors='coerce')
        
        original_len = len(df)
        df = df[df['year'] <= prediction_year].copy()
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} records from future years (leakage prevention).")
            
        df['yield_proxy'] = None
        valid_mask = (df['area'] > 0) & (df['production'] > 0) & df['area'].notna() & df['production'].notna()
        df.loc[valid_mask, 'yield_proxy'] = df.loc[valid_mask, 'production'] / df.loc[valid_mask, 'area']
        
        return df

    def ingest(self, prediction_year: int, state: Optional[str] = None, crop: Optional[str] = None) -> pd.DataFrame:
        df = self.fetch_data(state, crop)
        processed_df = self.process_and_validate(df, prediction_year)
        
        if not processed_df.empty:
            filename = f"production_{state or 'all'}_{crop or 'all'}.csv"
            filepath = self.raw_dir / filename
            processed_df.to_csv(filepath, index=False)
            logger.info(f"Saved production data to {filepath}")
            
        return processed_df
