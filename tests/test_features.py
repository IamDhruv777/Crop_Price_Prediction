import pytest
import pandas as pd
import numpy as np
from src.features.arbitrage_features import ArbitrageFeatureEngineer
from src.features.price_features import PriceFeatureEngineer

def test_arbitrage_feature_engineer():
    df = pd.DataFrame({
        'commodity': ['Green Gram (Moong)', 'Green Gram (Moong)', 'Green Gram (Moong)'],
        'market': ['Merta City', 'Merta City', 'Jaipur'],
        'date': ['2026-01-01', '2026-01-02', '2026-01-02'],
        'modal_price': [7200.0, 7250.0, 7100.0],
        'price_lag_1': [7180.0, 7200.0, 7050.0]
    })
    
    fe = ArbitrageFeatureEngineer()
    fe.fit(df)
    transformed = fe.transform(df)
    
    assert 'hub_price' in transformed.columns
    assert 'hub_price_spread' in transformed.columns
    assert 'hub_price_ratio' in transformed.columns
    assert not transformed['hub_price_spread'].isnull().all()

def test_price_features_no_leakage():
    df = pd.DataFrame({
        'market_id': ['MKT_1'] * 10,
        'commodity': ['Wheat'] * 10,
        'price_date': pd.date_range('2026-01-01', periods=10),
        'modal_price': [2000 + i * 50 for i in range(10)]
    })
    
    pfe = PriceFeatureEngineer()
    lagged = pfe.create_lag_features(df, price_col='modal_price', lags=[1, 7], group_cols=['market_id', 'commodity'])
    
    # Verify lag 1 is strictly past price (at index 1, lag 1 must be index 0's price 2000)
    assert lagged.loc[1, 'modal_price_lag_1'] == 2000
    assert np.isnan(lagged.loc[0, 'modal_price_lag_1'])
