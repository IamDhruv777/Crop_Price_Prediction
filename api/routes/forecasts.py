from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from ..schemas import ForecastRequest, ForecastResponse, PricePrediction
from src.inference.predictor import CropPricePredictor

router = APIRouter(prefix='/forecasts', tags=['forecasts'])

PREDS_PATH = Path("data/deployed_test_predictions.csv")
LEADERBOARD_PATH = Path("data/master_model_leaderboard.csv")

predictor = CropPricePredictor()

@router.post('/predict', response_model=ForecastResponse)
async def predict_forecast(request: ForecastRequest):
    """
    Main multi-horizon forecast endpoint with calibrated 95% conformal prediction bounds.
    Serves Tuned XGBoost Regressor for 1-Day and 7-Day horizons, and Naive Persistence Baseline for 30-Day horizon.
    """
    try:
        kwargs = {
            "state": request.state,
            "district": request.district,
            "market": request.market,
            "crop": request.crop,
            "horizon": request.horizon,
            "confidence_level": request.confidence_level,
            "variety": request.variety
        }
        if request.as_of_date:
            kwargs["as_of_date"] = request.as_of_date

        res = predictor.predict(**kwargs)
        if res.get("status") == "error":
            error_code = res.get("error_code")
            if error_code in ["insufficient_data", "unknown_crop", "invalid_date"]:
                raise HTTPException(status_code=422, detail=res.get("detail", "Insufficient data or validation error"))
            elif error_code == "model_unavailable":
                raise HTTPException(status_code=503, detail=res.get("detail", "Model unavailable"))
            else:
                raise HTTPException(status_code=500, detail=res.get("detail", "Prediction error"))
        
        predictions = []
        for p in res["predictions_trajectory"]:
            predictions.append(PricePrediction(
                date=p["date"],
                predicted_price=p["predicted_price"],
                lower_bound=p["lower_bound"],
                upper_bound=p["upper_bound"],
                is_interpolated_bound=p.get("is_interpolated_bound", False)
            ))
            
        if not res.get("model_version"):
            raise HTTPException(status_code=500, detail="Missing model_version in prediction output")
        if not res.get("as_of_date"):
            raise HTTPException(status_code=500, detail="Missing as_of_date in prediction output")
        if res.get("is_historical_backtest") is None:
            raise HTTPException(status_code=500, detail="Missing is_historical_backtest in prediction output")

        return ForecastResponse(
            state=res["state"],
            district=res["district"],
            market=res["market"],
            crop=res["crop"],
            horizon=res["horizon"],
            model=res["model_name"],
            model_version=res["model_version"],
            predictions=predictions,
            confidence_level=res["confidence_level"],
            rmse=res["rmse"],
            mae=res["mae"],
            r2=res.get("r2"),
            as_of_date=res["as_of_date"],
            is_historical_backtest=res["is_historical_backtest"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get('/history/{market_id}/{commodity}')
async def get_forecast_history(market_id: str, commodity: str):
    """
    Recent predictions for market and commodity from deployed test predictions.
    Sanitizes NaN float values to None for strict JSON compliance.
    """
    if PREDS_PATH.exists():
        df = pd.read_csv(PREDS_PATH)
        match = df[(df['market'].str.lower() == market_id.lower()) & (df['commodity'].str.lower() == commodity.lower())]
        if not match.empty:
            records = match.tail(30).replace({np.nan: None}).to_dict(orient='records')
            return {"market_id": market_id, "commodity": commodity, "history": records}
    return {"market_id": market_id, "commodity": commodity, "history": []}

@router.get('/accuracy/{model_name}')
async def get_model_accuracy(model_name: str):
    """
    Model accuracy and metrics from master leaderboard.
    """
    if LEADERBOARD_PATH.exists():
        df = pd.read_csv(LEADERBOARD_PATH)
        match = df[df['Model'].str.lower().str.contains(model_name.lower())]
        if not match.empty:
            return {"model_name": model_name, "metrics": match.to_dict(orient='records')}
    raise HTTPException(status_code=404, detail=f"Metrics for model '{model_name}' not found in leaderboard")
