from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date

class ForecastRequest(BaseModel):
    state: str = Field(..., description="State name")
    district: str = Field(..., description="District name")
    market: str = Field(..., description="Market name")
    crop: str = Field(..., description="Crop name")
    variety: Optional[str] = Field(None, description="Crop variety")
    horizon: int = Field(..., description="Forecast horizon in days (1, 7, 30)")
    confidence_level: float = Field(0.95, description="Confidence level for uncertainty interval")

    @validator('horizon')
    def validate_horizon(cls, v):
        if v not in [1, 7, 30]:
            raise ValueError('Horizon must be 1, 7, or 30')
        return v

class PricePrediction(BaseModel):
    date: str
    predicted_price: float
    lower_bound: float
    upper_bound: float

class ForecastResponse(BaseModel):
    state: str
    district: str
    market: str
    crop: str
    horizon: int
    model: str
    model_version: str
    predictions: List[PricePrediction]
    confidence_level: float
    rmse: float
    mae: float

class MarketInfo(BaseModel):
    market_id: str
    market_name: str
    district: str
    state: str
    latitude: float
    longitude: float

class PriceHistoryRequest(BaseModel):
    market_id: str
    commodity: str
    start_date: str
    end_date: str

class PriceRecord(BaseModel):
    market_id: str
    commodity: str
    price_date: str
    min_price: float
    max_price: float
    modal_price: float
    arrival_quantity: Optional[float] = None

class ModelMetrics(BaseModel):
    model_name: str
    crop: str
    horizon: int
    mae: float
    rmse: float
    mape: float
    smape: float
    wape: float
    r2: float

class AlertRecord(BaseModel):
    market_id: str
    commodity: str
    alert_date: str
    alert_type: str
    severity: str
    description: str

class DataQualityReport(BaseModel):
    table_name: str
    total_records: int
    missing_pct: float
    duplicate_pct: float
    quality_score: float
