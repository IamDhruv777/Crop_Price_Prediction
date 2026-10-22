from fastapi import APIRouter, HTTPException
import json
import pandas as pd
from pathlib import Path

router = APIRouter(prefix='/models', tags=['models'])

LEADERBOARD_PATH = Path("data/master_model_leaderboard.csv")
DEPLOYMENT_REGISTRY_PATH = Path("models/trained/deployment_registry.json")

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
    Return champion model based on benchmark evaluation and deployment registry.
    """
    norm_horizon = horizon if "-Day" in horizon else f"{horizon}-Day"

    if DEPLOYMENT_REGISTRY_PATH.exists():
        with open(DEPLOYMENT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        if norm_horizon in registry:
            dep_info = registry[norm_horizon]
            return {
                "crop": crop,
                "horizon": norm_horizon,
                "champion_model": dep_info["model_name"],
                "model_version": dep_info.get("model_version", "2.1.0-production"),
                "artifact": dep_info.get("model_path")
            }

    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        sub = df[df['Horizon'] == norm_horizon].sort_values(by="RMSE")
        if not sub.empty:
            best = sub.iloc[0].to_dict()
            return {"crop": crop, "horizon": norm_horizon, "champion_model": best}

    raise HTTPException(status_code=404, detail=f"Champion model not found for crop '{crop}' and horizon '{norm_horizon}'")

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
