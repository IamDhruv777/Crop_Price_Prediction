"""
Generates synthetic data for testing the pipeline when data.gov.in API keys are unavailable.
Generates 2 years of daily data for Maharashtra (Nashik, Pune, Solapur) for Onion.
"""
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

# Paths
DATA_DIR = Path("D:/DataScience/CP/crop-price-prediction/data/raw")
MANDI_DIR = DATA_DIR / "mandi"
WEATHER_DIR = DATA_DIR / "weather"
ARRIVALS_DIR = DATA_DIR / "arrivals"
GEO_DIR = DATA_DIR / "geography"

for d in [MANDI_DIR, WEATHER_DIR, ARRIVALS_DIR, GEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Market Master
markets = [
    {"market_id": "MH_NASHIK_LASALGAON", "market_name": "Lasalgaon", "district": "Nashik", "state": "Maharashtra", "latitude": 20.1167, "longitude": 74.0167, "region_id": "MH"},
    {"market_id": "MH_PUNE_PUNE", "market_name": "Pune", "district": "Pune", "state": "Maharashtra", "latitude": 18.5204, "longitude": 73.8567, "region_id": "MH"},
    {"market_id": "MH_SOLAPUR_SOLAPUR", "market_name": "Solapur", "district": "Solapur", "state": "Maharashtra", "latitude": 17.6862, "longitude": 75.9079, "region_id": "MH"}
]
pd.DataFrame(markets).to_csv(GEO_DIR / "market_master.csv", index=False)

# 2. Time series generation
start_date = date(2022, 1, 1)
end_date = date(2023, 12, 31)
days = (end_date - start_date).days
dates = [start_date + timedelta(days=i) for i in range(days + 1)]

mandi_records = []
weather_records = []
arrival_records = []

np.random.seed(42)

for m in markets:
    base_price = 1500 if m["market_name"] == "Lasalgaon" else 1600
    
    # Generate time series with seasonality and noise
    t = np.arange(len(dates))
    seasonality = -500 * np.sin(2 * np.pi * t / 365)
    trend = t * 0.5
    noise = np.random.normal(0, 200, len(dates))
    
    prices = np.clip(base_price + seasonality + trend + noise, 300, 5000)
    
    for i, d in enumerate(dates):
        # Mandi Price
        mandi_records.append({
            "state": m["state"],
            "district": m["district"],
            "market": m["market_name"],
            "commodity": "Onion",
            "variety": "Other",
            "grade": "FAQ",
            "arrival_date": d.strftime("%d/%m/%Y"),
            "min_price": max(100, int(prices[i] - 300)),
            "max_price": int(prices[i] + 300),
            "modal_price": int(prices[i])
        })
        
        # Arrivals
        arrival_records.append({
            "market_id": m["market_id"],
            "commodity": "Onion",
            "arrival_date": d.strftime("%Y-%m-%d"),
            "arrival_quantity": max(10, int(1000 + np.random.normal(0, 200))),
            "unit": "Quintal",
            "data_source": "synthetic",
            "is_imputed": False
        })
        
        # Weather
        weather_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "latitude": m["latitude"],
            "longitude": m["longitude"],
            "temperature": 25.0 + 5*np.sin(2*np.pi*(t[i]-150)/365) + np.random.normal(0,2),
            "temperature_max": 30.0 + 5*np.sin(2*np.pi*(t[i]-150)/365) + np.random.normal(0,2),
            "temperature_min": 20.0 + 5*np.sin(2*np.pi*(t[i]-150)/365) + np.random.normal(0,2),
            "rainfall": max(0, np.random.normal(-5, 10)) if 150 < (t[i]%365) < 270 else 0,
            "humidity": 60 + np.random.normal(0, 10),
            "wind_speed": 3.0 + np.random.normal(0, 1)
        })

# Save JSON/CSV
pd.DataFrame(mandi_records).to_json(MANDI_DIR / "mandi_raw_Maharashtra_synthetic.json", orient="records")
pd.DataFrame(weather_records).to_csv(WEATHER_DIR / "weather_synthetic.csv", index=False)
pd.DataFrame(arrival_records).to_csv(ARRIVALS_DIR / "arrivals_synthetic.csv", index=False)

print("Synthetic data generated successfully!")
