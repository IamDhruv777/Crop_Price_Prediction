import pandas as pd
import numpy as np
import pickle

from .price_features import PriceFeatureEngineer
from .arrival_features import ArrivalFeatureEngineer
from .temporal_features import TemporalFeatureEngineer
from .weather_features import WeatherFeatureEngineer
from .spatial_features import SpatialFeatureEngineer

class FeaturePipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.price_engineer = PriceFeatureEngineer()
        self.arrival_engineer = ArrivalFeatureEngineer()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.weather_engineer = WeatherFeatureEngineer()
        self.spatial_engineer = SpatialFeatureEngineer()
        
        self.scalers = {}
        self.encoders = {}
        self.neighbors_df = None
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, price_df: pd.DataFrame, weather_df: pd.DataFrame, arrival_df: pd.DataFrame, market_df: pd.DataFrame, train_end_date: str):
        price_train = price_df[price_df['price_date'] <= train_end_date].copy()
        
        if market_df is not None and not market_df.empty:
            self.neighbors_df = self.spatial_engineer.find_k_nearest_markets(market_df, k=5, max_distance_km=200)
        
        self.is_fitted = True

    def transform(self, price_df: pd.DataFrame, weather_df: pd.DataFrame, arrival_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
            
        df = price_df.sort_values(by=['market_id', 'commodity', 'price_date']).copy()
        
        df = self.price_engineer.create_all_price_features(df, price_col='modal_price', group_cols=['market_id', 'commodity'])
        
        if arrival_df is not None and not arrival_df.empty:
            arr = arrival_df.sort_values(by=['market_id', 'commodity', 'price_date']).copy()
            arr_feats = self.arrival_engineer.create_all_arrival_features(arr, arrival_col='arrival_quantity', group_cols=['market_id', 'commodity'])
            df = pd.merge(df, arr_feats, on=['market_id', 'commodity', 'price_date'], how='left')
        
        if weather_df is not None and not weather_df.empty:
            weather_df = weather_df.sort_values(by=['market_id', 'price_date']).copy()
            weather_feats = self.weather_engineer.create_all_weather_features(weather_df, group_cols=['market_id'])
            df = self.weather_engineer.merge_weather_with_prices(df, weather_feats, market_id_col='market_id', date_col='price_date')
            
        df = self.temporal_engineer.create_all_temporal_features(df, date_col='price_date')
        
        if market_df is not None and not market_df.empty and self.neighbors_df is not None:
            df = self.spatial_engineer.create_all_spatial_features(
                price_df=df, 
                market_df=market_df, 
                neighbors_df=self.neighbors_df, 
                date_col='price_date', 
                price_col='modal_price'
            )
            
        exclude_cols = ['market_id', 'commodity', 'price_date', 'modal_price']
        self.feature_names = [c for c in df.columns if c not in exclude_cols]
        
        return df

    def fit_transform(self, price_df: pd.DataFrame, weather_df: pd.DataFrame, arrival_df: pd.DataFrame, market_df: pd.DataFrame, train_end_date: str) -> pd.DataFrame:
        self.fit(price_df, weather_df, arrival_df, market_df, train_end_date)
        return self.transform(price_df, weather_df, arrival_df, market_df)

    def get_feature_names(self) -> list:
        return self.feature_names

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        with open(path, 'rb') as f:
            return pickle.load(f)
