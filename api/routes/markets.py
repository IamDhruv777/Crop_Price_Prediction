from fastapi import APIRouter
import pandas as pd
from pathlib import Path

router = APIRouter(prefix='/markets', tags=['markets'])

GEO_PATH = Path("data/raw/geography/market_master.csv")
EDGES_PATH = Path("data/spatial/spatial_graph_edges.csv")

@router.get('/')
async def list_markets(state: str = None, limit: int = 50):
    """
    List markets across India with geographic coordinates.
    """
    if GEO_PATH.exists():
        df = pd.read_csv(GEO_PATH)
        if state:
            df = df[df['state'].str.lower() == state.lower()]
        return {
            "total_count": len(df),
            "markets": df.head(limit).to_dict(orient='records')
        }
    return {"markets": []}

@router.get('/{market_id}')
async def get_market_detail(market_id: str):
    """
    Market detail and agro-climatic zone.
    """
    if GEO_PATH.exists():
        df = pd.read_csv(GEO_PATH)
        match = df[(df['market_id'].str.lower() == market_id.lower()) | (df['market_name'].str.lower() == market_id.lower())]
        if not match.empty:
            return {"market": match.iloc[0].to_dict()}
    return {"market_id": market_id, "details": {}}

@router.get('/{market_id}/neighbors')
async def get_market_neighbors(market_id: str):
    """
    K-nearest neighbor mandis from GraphSAGE spatial graph.
    """
    if EDGES_PATH.exists():
        edges = pd.read_csv(EDGES_PATH)
        match = edges[(edges['source_market_id'].str.lower() == market_id.lower()) | (edges['source_name'].str.lower() == market_id.lower())]
        if not match.empty:
            return {
                "source_market": market_id,
                "neighbors": match.to_dict(orient='records')
            }
    return {"market_id": market_id, "neighbors": []}
