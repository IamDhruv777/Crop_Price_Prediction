import os
import json
import time
import logging
import hashlib
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class MandiIngestion:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_key = os.environ.get('DATA_GOV_API_KEY')
        if not self.api_key:
            logger.warning("DATA_GOV_API_KEY environment variable not set. API calls will fail.")
        self.resource_id = self.config.get('resource_id', '9ef84268-d588-465a-a308-a864a43d0070') # default mandi resource id
        self.base_url = f"https://api.data.gov.in/resource/{self.resource_id}"
        self.raw_dir = Path("data/raw/mandi")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.required_fields = ['state', 'district', 'market', 'commodity', 'arrival_date', 'min_price', 'max_price', 'modal_price']

    def _get_checksum(self, filepath: Path) -> str:
        if not filepath.exists():
            return ""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _fetch_page(self, params: Dict[str, Any]) -> dict:
        retries = 3
        backoff_factor = 2
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        for attempt in range(retries):
            try:
                response = requests.get(self.base_url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    sleep_time = backoff_factor ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries exceeded.")
                    raise

    def _validate_record(self, record: dict) -> bool:
        is_valid = True
        for field in self.required_fields:
            if field not in record or record[field] is None:
                logger.warning(f"Missing required field '{field}' in record: {record}")
                record[field] = None
                is_valid = False
        return is_valid

    def _save_raw(self, data: List[dict], state: str, commodity: str, date_str: str) -> Path:
        state_clean = state.replace(" ", "_").lower() if state else "all"
        filename = f"mandi_raw_{state_clean}_{date_str}.json"
        filepath = self.raw_dir / filename
        
        metadata = {
            "source": "data.gov.in",
            "download_time": datetime.now().isoformat(),
            "date_range": date_str,
            "record_count": len(data),
            "state": state,
            "commodity": commodity
        }
        
        output = {
            "metadata": metadata,
            "records": data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        # Update metadata with checksum
        metadata['checksum'] = self._get_checksum(filepath)
        
        # Save again with checksum
        output['metadata'] = metadata
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(data)} records to {filepath}")
        return filepath

    def _load_cached(self, state: str, commodity: str, date_str: str) -> Optional[pd.DataFrame]:
        state_clean = state.replace(" ", "_").lower() if state else "all"
        filename = f"mandi_raw_{state_clean}_{date_str}.json"
        filepath = self.raw_dir / filename
        
        if filepath.exists():
            logger.info(f"Found cached data at {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            records = data.get('records', [])
            return pd.DataFrame(records)
        return None

    def fetch_data(self, state: str, commodity: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("API key is required to fetch data.")
            
        all_records = []
        limit = 500
        offset = 0
        total_records = float('inf')
        
        params = {
            'api-key': self.api_key,
            'format': 'json',
            'limit': limit,
            'offset': offset
        }
        
        if state:
            params['filters[state]'] = state
        if commodity:
            params['filters[commodity]'] = commodity
            
        logger.info(f"Fetching data for State: {state}, Commodity: {commodity}")
        
        while offset < total_records:
            params['offset'] = offset
            data = self._fetch_page(params)
            
            records = data.get('records', [])
            if not records:
                break
                
            for record in records:
                self._validate_record(record)
                all_records.append(record)
                
            total_records = data.get('total', 0)
            offset += limit
            
            logger.info(f"Fetched {len(all_records)} / {total_records} records")
            
        df = pd.DataFrame(all_records)
        if not df.empty and start_date and end_date:
            # Filter dates if available
            try:
                df['arrival_date_parsed'] = pd.to_datetime(df['arrival_date'], dayfirst=True, format='mixed', errors='coerce')
                s_dt = pd.to_datetime(start_date, dayfirst=True, format='mixed')
                e_dt = pd.to_datetime(end_date, dayfirst=True, format='mixed')
                mask = (df['arrival_date_parsed'] >= s_dt) & (df['arrival_date_parsed'] <= e_dt)
                if mask.any():
                    df = df.loc[mask].drop(columns=['arrival_date_parsed'])
                else:
                    df = df.drop(columns=['arrival_date_parsed'])
            except Exception:
                pass
            
        return df

    def ingest(self, state: str, commodity: str, start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
        clean_s = str(start_date).replace('-', '').replace('/', '')
        clean_e = str(end_date).replace('-', '').replace('/', '')
        date_str = f"{clean_s}_{clean_e}"
        
        if not force_refresh:
            cached_df = self._load_cached(state, commodity, date_str)
            if cached_df is not None:
                return cached_df
                
        df = self.fetch_data(state, commodity, start_date, end_date)
        if not df.empty:
            records = df.to_dict(orient='records')
            self._save_raw(records, state, commodity, date_str)
        else:
            logger.warning("No data fetched, nothing to save.")
            
        return df
