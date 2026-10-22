import pandas as pd
import numpy as np

class TemporalFeatureEngineer:
    def __init__(self):
        pass

    def create_calendar_features(self, df: pd.DataFrame, date_col: str = 'price_date') -> pd.DataFrame:
        df_out = df.copy()
        df_out[date_col] = pd.to_datetime(df_out[date_col])
        
        df_out['day_of_week'] = df_out[date_col].dt.dayofweek
        df_out['day_of_month'] = df_out[date_col].dt.day
        df_out['week_of_year'] = df_out[date_col].dt.isocalendar().week.astype(int)
        df_out['month'] = df_out[date_col].dt.month
        df_out['quarter'] = df_out[date_col].dt.quarter
        df_out['year'] = df_out[date_col].dt.year
        df_out['is_weekend'] = df_out['day_of_week'].isin([5, 6]).astype(int)
        df_out['is_month_start'] = df_out[date_col].dt.is_month_start.astype(int)
        df_out['is_month_end'] = df_out[date_col].dt.is_month_end.astype(int)
        
        return df_out

    def create_cyclic_features(self, df: pd.DataFrame, date_col: str = 'price_date') -> pd.DataFrame:
        df_out = df.copy()
        if 'month' not in df_out.columns:
            df_out = self.create_calendar_features(df_out, date_col)
            
        df_out['sin_month'] = np.sin(2 * np.pi * df_out['month'] / 12)
        df_out['cos_month'] = np.cos(2 * np.pi * df_out['month'] / 12)
        
        df_out['sin_week'] = np.sin(2 * np.pi * df_out['week_of_year'] / 52)
        df_out['cos_week'] = np.cos(2 * np.pi * df_out['week_of_year'] / 52)
        
        df_out['sin_dow'] = np.sin(2 * np.pi * df_out['day_of_week'] / 7)
        df_out['cos_dow'] = np.cos(2 * np.pi * df_out['day_of_week'] / 7)
        
        return df_out

    def create_season_features(self, df: pd.DataFrame, date_col: str = 'price_date') -> pd.DataFrame:
        df_out = df.copy()
        if 'month' not in df_out.columns:
            df_out = self.create_calendar_features(df_out, date_col)
            
        def get_season(month):
            if month in [7, 8, 9, 10]:
                return 'Kharif'
            elif month in [11, 12, 1, 2, 3]:
                return 'Rabi'
            else:
                return 'Summer'
                
        df_out['season'] = df_out['month'].apply(get_season)
        
        season_map = {'Kharif': 0, 'Rabi': 1, 'Summer': 2}
        df_out['season_encoded'] = df_out['season'].map(season_map)
        
        df_out = pd.get_dummies(df_out, columns=['season'], prefix='season')
        
        def get_season_start_month(month):
            if month in [7, 8, 9, 10]:
                return 7
            elif month in [11, 12, 1, 2, 3]:
                return 11
            else:
                return 4
                
        df_out['season_start_month'] = df_out['month'].apply(get_season_start_month)
        
        dates = pd.to_datetime(df_out[date_col])
        season_start_dates = dates.apply(
            lambda x: pd.Timestamp(year=x.year if x.month >= get_season_start_month(x.month) else x.year - 1, 
                                   month=get_season_start_month(x.month), 
                                   day=1)
        )
        df_out['days_since_season_start'] = (dates - season_start_dates).dt.days
        df_out.drop(columns=['season_start_month'], inplace=True)
        
        return df_out

    def create_holiday_features(self, df: pd.DataFrame, date_col: str = 'price_date') -> pd.DataFrame:
        df_out = df.copy()
        dates = pd.to_datetime(df_out[date_col])
        
        holidays_fixed = [(1, 26), (8, 15), (10, 2)] 
        
        def is_holiday(d):
            return (d.month, d.day) in holidays_fixed
            
        df_out['is_public_holiday'] = dates.apply(is_holiday).astype(int)
        
        unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
        holiday_dates = unique_dates[unique_dates.apply(is_holiday)]
        
        def days_to_next(d):
            future = holiday_dates[holiday_dates >= d]
            if len(future) > 0:
                return (future.iloc[0] - d).days
            return 365 
            
        def days_since_last(d):
            past = holiday_dates[holiday_dates <= d]
            if len(past) > 0:
                return (d - past.iloc[-1]).days
            return 365 
            
        df_out['days_to_next_holiday'] = dates.apply(days_to_next)
        df_out['days_since_last_holiday'] = dates.apply(days_since_last)
        
        return df_out

    def create_all_temporal_features(self, df: pd.DataFrame, date_col: str = 'price_date') -> pd.DataFrame:
        df_out = df.copy()
        df_out = self.create_calendar_features(df_out, date_col)
        df_out = self.create_cyclic_features(df_out, date_col)
        df_out = self.create_season_features(df_out, date_col)
        df_out = self.create_holiday_features(df_out, date_col)
        return df_out
