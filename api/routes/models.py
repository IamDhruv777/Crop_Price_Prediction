from fastapi import APIRouter
import pandas as pd
from pathlib import Path

router = APIRouter(prefix='/models', tags=['models'])

LEADERBOARD_PATH = Path("data/master_model_leaderboard.csv")

@router.get('/')
async def list_models():
    """
    List trained models and architectures.
    """
    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        models = df[['Architecture_Type', 'Model', 'Horizon', 'RMSE', 'Trust_Score']].to_dict(orient='records')
        return {"total_models": len(models), "models": models}
    return {"models": []}

@router.get('/comparison')
async def compare_models():
    """
    Master model comparison leaderboard table.
    """
    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        return {"comparison": df.to_dict(orient='records')}
    return {"comparison": []}

@router.get('/best')
async def get_best_model(crop: str = "Onion", horizon: str = "7-Day"):
    """
    Return champion model based on benchmark evaluation.
    """
    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        sub = df[df['Horizon'] == horizon].sort_values(by="RMSE")
        if not sub.empty:
            best = sub.iloc[0].to_dict()
            return {"crop": crop, "horizon": horizon, "champion_model": best}
    return {"crop": crop, "horizon": horizon, "champion_model": "Spatial Graph (GraphSAGE + XGBoost)"}

@router.get('/{model_id}')
async def get_model_details(model_id: str):
    """
    Model details + metrics.
    """
    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        match = df[df['Model'].str.lower().str.contains(model_id.lower())]
        if not match.empty:
            return {"model_id": model_id, "details": match.to_dict(orient='records')}
    return {"model_id": model_id, "details": {}}
