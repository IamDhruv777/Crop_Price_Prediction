import pandas as pd
import re
from typing import Optional
from datetime import date

STATE_ALIASES = {
    'uttar pradesh': 'uttar pradesh', 'up': 'uttar pradesh', 'u.p.': 'uttar pradesh',
    'madhya pradesh': 'madhya pradesh', 'mp': 'madhya pradesh', 'm.p.': 'madhya pradesh',
    'andhra pradesh': 'andhra pradesh', 'ap': 'andhra pradesh',
    'maharashtra': 'maharashtra', 'maha': 'maharashtra',
    'karnataka': 'karnataka', 'kar': 'karnataka',
    'gujarat': 'gujarat', 'guj': 'gujarat',
    'rajasthan': 'rajasthan', 'raj': 'rajasthan',
    'tamil nadu': 'tamil nadu', 'tn': 'tamil nadu', 'tamilnadu': 'tamil nadu',
    'west bengal': 'west bengal', 'wb': 'west bengal',
    'punjab': 'punjab', 'pun': 'punjab',
    'haryana': 'haryana', 'har': 'haryana',
    'bihar': 'bihar',
    'odisha': 'odisha', 'orissa': 'odisha',
    'kerala': 'kerala', 'ker': 'kerala',
    'telangana': 'telangana', 'ts': 'telangana',
    'assam': 'assam',
    'chhattisgarh': 'chhattisgarh', 'cg': 'chhattisgarh',
    'jharkhand': 'jharkhand', 'jh': 'jharkhand',
    'uttarakhand': 'uttarakhand', 'uk': 'uttarakhand',
    'himachal pradesh': 'himachal pradesh', 'hp': 'himachal pradesh'
}

class DataCleaner:
    def clean_mandi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        
        # Remove exact duplicate rows
        cleaned_df = cleaned_df.drop_duplicates()
        
        if 'market' in cleaned_df.columns:
            cleaned_df['market'] = cleaned_df['market'].apply(self.normalize_market_name)
            
        if 'state' in cleaned_df.columns:
            cleaned_df['state'] = cleaned_df['state'].apply(self.normalize_state_name)
            
        if 'district' in cleaned_df.columns:
            cleaned_df['district'] = cleaned_df['district'].apply(self.normalize_market_name)
            
        if 'commodity' in cleaned_df.columns:
            cleaned_df['commodity'] = cleaned_df['commodity'].apply(lambda x: self.normalize_commodity_name(x, {}))
            
        if 'variety' in cleaned_df.columns:
            cleaned_df['variety'] = cleaned_df['variety'].apply(self.normalize_market_name)
            
        price_cols = ['min_price', 'max_price', 'modal_price']
        for col in price_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
        if 'price_date' in cleaned_df.columns:
            cleaned_df['price_date'] = cleaned_df['price_date'].apply(self.parse_mandi_date)
            cleaned_df['price_date'] = pd.to_datetime(cleaned_df['price_date'])
            
        if all(c in cleaned_df.columns for c in ['state', 'district', 'market']):
            cleaned_df['market_id'] = cleaned_df['state'] + '_' + cleaned_df['district'] + '_' + cleaned_df['market']
            cleaned_df['market_id'] = cleaned_df['market_id'].astype(str).str.replace(' ', '_').str.lower()
            
        return cleaned_df

    def normalize_market_name(self, name: str) -> str:
        if pd.isna(name):
            return ""
        name = str(name).strip()
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        return name.title()

    def normalize_state_name(self, name: str) -> str:
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        return STATE_ALIASES.get(name, name.title())

    def normalize_commodity_name(self, name: str, crop_aliases: dict) -> str:
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        return crop_aliases.get(name, name).title()

    def parse_mandi_date(self, date_str: str) -> Optional[date]:
        if pd.isna(date_str):
            return None
        try:
            dt = pd.to_datetime(date_str, format='mixed', dayfirst=True).date()
            return dt
        except Exception:
            return None
