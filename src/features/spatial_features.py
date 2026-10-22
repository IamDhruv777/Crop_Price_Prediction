import pandas as pd
import numpy as np

class SpatialFeatureEngineer:
    def __init__(self):
        pass

    def compute_haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        return distance

    def find_k_nearest_markets(self, market_df: pd.DataFrame, k: int = 5, max_distance_km: float = 200.0) -> pd.DataFrame:
        records = []
        for i, row1 in market_df.iterrows():
            market1 = row1['market_id']
            lat1, lon1 = row1['latitude'], row1['longitude']
            
            distances = []
            for j, row2 in market_df.iterrows():
                market2 = row2['market_id']
                if market1 == market2:
                    continue
                lat2, lon2 = row2['latitude'], row2['longitude']
                dist = self.compute_haversine_distance(lat1, lon1, lat2, lon2)
                
                if dist <= max_distance_km:
                    distances.append((market2, dist))
                    
            distances.sort(key=lambda x: x[1])
            
            for rank, (market2, dist) in enumerate(distances[:k], 1):
                records.append({
                    'source_market_id': market1,
                    'neighbor_market_id': market2,
                    'distance_km': dist,
                    'rank': rank
                })
                
        return pd.DataFrame(records)

    def create_neighbor_price_features(self, price_df: pd.DataFrame, neighbors_df: pd.DataFrame, date_col: str = 'price_date', price_col: str = 'modal_price') -> pd.DataFrame:
        lagged_prices = price_df.copy()
        lagged_prices = lagged_prices.sort_values(by=['market_id', 'commodity', date_col])
        lagged_prices[f'{price_col}_lag_1'] = lagged_prices.groupby(['market_id', 'commodity'])[price_col].shift(1)
        
        merged = pd.merge(
            neighbors_df, 
            lagged_prices[['market_id', 'commodity', date_col, f'{price_col}_lag_1']], 
            left_on='neighbor_market_id', 
            right_on='market_id'
        )
        
        stats = merged.groupby(['source_market_id', 'commodity', date_col])[f'{price_col}_lag_1'].agg(
            mean_neighbor_price='mean',
            median_neighbor_price='median',
            max_neighbor_price='max',
            min_neighbor_price='min'
        ).reset_index()
        
        nearest = merged[merged['rank'] == 1].copy()
        nearest = nearest.rename(columns={f'{price_col}_lag_1': 'nearest_neighbor_price'})
        
        stats = pd.merge(stats, nearest[['source_market_id', 'commodity', date_col, 'nearest_neighbor_price']], on=['source_market_id', 'commodity', date_col], how='left')
        
        result = pd.merge(
            price_df, 
            stats, 
            left_on=['market_id', 'commodity', date_col], 
            right_on=['source_market_id', 'commodity', date_col], 
            how='left'
        )
        result.drop(columns=['source_market_id'], inplace=True, errors='ignore')
        return result

    def create_regional_features(self, price_df: pd.DataFrame, market_df: pd.DataFrame, date_col: str, price_col: str, group_col: str = 'state') -> pd.DataFrame:
        df = pd.merge(price_df, market_df[['market_id', group_col]], on='market_id', how='left')
        df = df.sort_values(by=[group_col, 'commodity', date_col])
        
        regional_stats = df.groupby([group_col, 'commodity', date_col])[price_col].agg(
            regional_average_price='mean',
            regional_median_price='median',
            regional_price_std='std'
        ).reset_index()
        
        regional_stats = regional_stats.sort_values(by=[group_col, 'commodity', date_col])
        for col in ['regional_average_price', 'regional_median_price', 'regional_price_std']:
            regional_stats[col] = regional_stats.groupby([group_col, 'commodity'])[col].shift(1)
            
        result = pd.merge(df, regional_stats, on=[group_col, 'commodity', date_col], how='left')
        
        if 'mean_neighbor_price' in result.columns:
            result = result.sort_values(by=['market_id', 'commodity', date_col])
            result['neighbor_price_change'] = result['mean_neighbor_price'] - result.groupby(['market_id', 'commodity'])['mean_neighbor_price'].shift(1)
            
        result.drop(columns=[group_col], inplace=True)
        return result

    def create_geographic_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        df_out = market_df.copy()
        
        if 'latitude' in df_out.columns and 'longitude' in df_out.columns:
            df_out['latitude_norm'] = (df_out['latitude'] - df_out['latitude'].mean()) / df_out['latitude'].std()
            df_out['longitude_norm'] = (df_out['longitude'] - df_out['longitude'].mean()) / df_out['longitude'].std()
            
            distances = []
            counts = []
            
            for i, row1 in df_out.iterrows():
                lat1, lon1 = row1['latitude'], row1['longitude']
                min_dist = float('inf')
                count_100km = 0
                
                for j, row2 in df_out.iterrows():
                    if i == j:
                        continue
                    lat2, lon2 = row2['latitude'], row2['longitude']
                    dist = self.compute_haversine_distance(lat1, lon1, lat2, lon2)
                    
                    if dist < min_dist:
                        min_dist = dist
                    if dist <= 100.0:
                        count_100km += 1
                        
                distances.append(min_dist if min_dist != float('inf') else np.nan)
                counts.append(count_100km)
                
            df_out['distance_to_nearest_market'] = distances
            df_out['number_of_nearby_markets'] = counts
            
        return df_out

    def create_all_spatial_features(self, price_df: pd.DataFrame, market_df: pd.DataFrame, neighbors_df: pd.DataFrame, date_col: str, price_col: str) -> pd.DataFrame:
        df_out = self.create_neighbor_price_features(price_df, neighbors_df, date_col, price_col)
        df_out = self.create_regional_features(df_out, market_df, date_col, price_col)
        
        geo_features = self.create_geographic_features(market_df)
        df_out = pd.merge(df_out, geo_features[['market_id', 'latitude_norm', 'longitude_norm', 'distance_to_nearest_market', 'number_of_nearby_markets']], on='market_id', how='left')
        
        return df_out
