import os
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class ArrivalIngestion:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.raw_dir = Path("data/raw/arrivals")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.csv_fallback_path = self.config.get('csv_path', self.raw_dir / "manual_arrivals.csv")

    def _normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'unit' not in df.columns or 'arrival_quantity' not in df.columns:
            return df
            
        def convert_to_quintals(row):
            qty = row['arrival_quantity']
            unit = str(row['unit']).lower().strip()
            
            if pd.isna(qty):
                return qty, unit
                
            if unit in ['tonnes', 'tonne', 't']:
                return qty * 10, 'Quintal'
            elif unit in ['kg', 'kilograms']:
                return qty / 100, 'Quintal'
            elif unit in ['quintals', 'quintal', 'q']:
                return qty, 'Quintal'
            else:
                return qty, unit
                
        if not df.empty:
            converted = df.apply(convert_to_quintals, axis=1, result_type='expand')
            df['arrival_quantity'] = converted[0]
            df['unit'] = converted[1]
            
        return df

    def ingest_from_mandi(self, mandi_df: pd.DataFrame) -> pd.DataFrame:
        if mandi_df.empty:
            return pd.DataFrame()
            
        df = pd.DataFrame()
        arrival_col = next((c for c in mandi_df.columns if 'arrival' in c.lower() and ('qty' in c.lower() or 'quantity' in c.lower())), None)
        
        df['market_id'] = mandi_df.get('market_id', mandi_df.get('market', 'UNKNOWN'))
        df['commodity'] = mandi_df.get('commodity', 'UNKNOWN')
        df['arrival_date'] = mandi_df.get('arrival_date')
        
        if arrival_col:
            df['arrival_quantity'] = pd.to_numeric(mandi_df[arrival_col], errors='coerce')
            df['unit'] = mandi_df.get('unit', 'Tonnes')
            df['data_source'] = 'mandi_primary'
            df['is_from_primary'] = True
            df['arrival_missing'] = df['arrival_quantity'].isna()
        else:
            logger.info("Primary mandi data does not contain arrival quantities.")
            df['arrival_quantity'] = None
            df['unit'] = None
            df['data_source'] = 'none'
            df['is_from_primary'] = False
            df['arrival_missing'] = True
            
        return self._normalize_units(df)

    def ingest_fallback(self) -> pd.DataFrame:
        if isinstance(self.csv_fallback_path, (str, Path)) and Path(self.csv_fallback_path).exists():
            logger.info(f"Loading arrivals from fallback CSV: {self.csv_fallback_path}")
            df = pd.read_csv(self.csv_fallback_path)
            
            required = ['market_id', 'commodity', 'arrival_date', 'arrival_quantity']
            for col in required:
                if col not in df.columns:
                    logger.warning(f"Fallback CSV missing required column: {col}")
                    
            if 'unit' not in df.columns:
                df['unit'] = 'Tonnes'
                
            df['data_source'] = 'fallback_csv'
            df['is_from_primary'] = False
            df['arrival_missing'] = df['arrival_quantity'].isna()
            
            return self._normalize_units(df)
        return pd.DataFrame()

    def ingest(self, mandi_df: pd.DataFrame) -> pd.DataFrame:
        arrivals_df = self.ingest_from_mandi(mandi_df)
        
        if arrivals_df.empty or arrivals_df['arrival_missing'].all():
            logger.info("Attempting to load from fallback source.")
            fallback_df = self.ingest_fallback()
            if not fallback_df.empty:
                return fallback_df
                
        if arrivals_df.empty:
            logger.warning("No arrival data available from any source.")
            
        return arrivals_df
