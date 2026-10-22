from fastapi import APIRouter
import pandas as pd
from pathlib import Path

router = APIRouter(prefix='/prices', tags=['prices'])

PREDS_PATH = Path("data/test_predictions.csv")

@router.get('/current')
async def get_current_prices(limit: int = 50):
    """
    Latest observed wholesale clearing prices across mandis.
    """
    if PREDS_PATH.exists():
        df = pd.read_csv(PREDS_PATH)
        latest = df.groupby('market').last().reset_index()
        return {
            "total_markets": len(latest),
            "prices": latest[['market', 'commodity', 'date', 'modal_price', 'predicted_price_7d']].head(limit).to_dict(orient='records')
        }
    return {"prices": []}

@router.get('/history')
async def get_price_history(market_id: str = None, commodity: str = None, limit: int = 50):
    """
    Historical wholesale price observations.
    """
    if PREDS_PATH.exists():
        df = pd.read_csv(PREDS_PATH)
        if market_id:
            df = df[df['market'].str.lower() == market_id.lower()]
        if commodity:
            df = df[df['commodity'].str.lower() == commodity.lower()]
        return {"records": df.tail(limit).to_dict(orient='records')}
    return {"records": []}

@router.get('/markets')
async def list_price_markets():
    """
    List markets with price history.
    """
    if PREDS_PATH.exists():
        df = pd.read_csv(PREDS_PATH)
        return {"markets": sorted(df['market'].dropna().unique().tolist())}
    return {"markets": []}
