# Krishi Bhav Saathi (कृषि भाव साथी) — Region-Wise AI Crop Price Prediction

A production-grade, spatial-temporal agricultural commodity price forecasting platform built using **Spatial Graph Neural Networks (GraphSAGE)**, **Deep Learning (PyTorch TCN/LSTM/GRU/Transformer)**, **Gradient Boosted Trees (XGBoost/RandomForest)**, and **Statistical Baselines**, integrated with an interactive **3D Streamlit dashboard** and **FastAPI REST backend**.

---

## Key Features

1. **Dual-Audience User Interface**:
   - **Local Farmer / Rural Vendor Mode**: Ultra-simplified, low-cognitive-load interface designed for rural users. Provides direct holding/selling advice (`🟢 माल रोकें (Hold)` / `🔴 आज बेचें (Sell)` / `🔵 स्थिर (Stable)`), transparent pricing comparisons, and relatable cultural context.
   - **Agricultural Researcher Mode**: Comprehensive analytics workbench featuring multi-model accuracy benchmarks, SHAP feature importance, NASA agroclimatology analysis, and architectural ablation studies.
2. **Interactive 3D Landing Page**:
   - Relatable 3D animated agricultural balance scale (*Tarazu*) built in Three.js with bilingual support (**हिन्दी / English**).
3. **Hierarchical Unified India Map**:
   - Multi-level choropleth map drilldown: **Nationwide $\rightarrow$ State $\rightarrow$ District $\rightarrow$ APMC Mandi**.
4. **Multi-Horizon Forecasting with 95% Conformal Confidence Intervals**:
   - Calibrated price forecasts across **1-Day**, **7-Day**, and **30-Day** horizons with mathematically guaranteed coverage intervals ($P_{\text{lower}} \le P_{\text{pred}} \le P_{\text{upper}}$).
5. **Phase 2 Benchmark Hub Arbitrage & Supply Shock Signals**:
   - Leading price signals from national benchmark hubs (Raichur, Lalitpur, Mumbai, Merta City, Indore, Burdwan, Kota, Dahod) capturing spatial price contagion and arrival supply-shock dynamics.
6. **Dedicated Per-Crop Champions**:
   - Specialized champion models trained per commodity to capture distinct perishability, MSP floors, and harvest cycles.

---

## Multi-Model Architecture & Benchmark Leaderboard

Evaluated on over 4.2 million real AGMARKNET transaction records and NASA POWER daily agroclimatology:

| Architecture Tier | Model Architecture | 7-Day RMSE (₹/Qtl) | MAE (₹/Qtl) | MAPE (%) | $R^2$ Score | Directional Accuracy | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Meta-Learner** | **Spatial-Temporal Super Ensemble (Champion)** | **672.36** | **403.04** | **8.00%** | **0.8823** | **57.35%** | **0.5955** |
| Statistical Baseline | Moving Average Baseline (7-Day) | 691.90 | 417.22 | 8.72% | 0.8753 | 61.26% | 0.6491 |
| Spatial Graph Neural Net | Spatial Graph (GraphSAGE + XGBoost) | 797.24 | 468.73 | 9.38% | 0.8345 | 57.67% | 0.6274 |
| Deep Learning | Temporal CNN (TCN - SmoothL1) | 821.86 | 545.20 | 10.75% | 0.8241 | 52.11% | 0.6721 |
| Classical ML | Random Forest Regressor | 838.08 | 580.85 | 12.15% | 0.8171 | 51.86% | 0.6378 |
| Statistical Baseline | Naive Persistence Baseline | 869.58 | 410.12 | 8.21% | 0.8031 | 50.20% | 0.6685 |
| Classical ML | Tuned XGBoost Regressor | 899.43 | 587.78 | 12.01% | 0.7893 | 51.82% | 0.6391 |
| Deep Learning | PyTorch GRU (SmoothL1) | 904.58 | 589.10 | 11.61% | 0.7869 | 52.26% | 0.6675 |
| Deep Learning | PyTorch LSTM (SmoothL1) | 917.63 | 592.30 | 11.86% | 0.7807 | 51.12% | 0.6556 |
| Deep Learning | Time-Series Transformer (SmoothL1) | 936.59 | 620.40 | 12.51% | 0.7716 | 50.12% | 0.6474 |

*Note: The test set average price is ₹5,316/Qtl (dominated by high-value pulses). An MAE of ₹403/Qtl corresponds to an average error of only ~₹4.0/kg, yielding a 92%–95% overall trust score.*

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/IamDhruv777/Crop_Price_Prediction.git
cd Crop_Price_Prediction

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Services

#### Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

#### Launch FastAPI Backend Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Explore interactive Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Running Automated Tests

Run the complete 10-test test suite covering inference bounds, zero-leakage causal lags, spatial distance, and API endpoints:
```bash
pytest tests/
```

---

## End-to-End Pipeline Execution

Run the master pipeline script to trigger data preprocessing, feature engineering, spatial graph generation, and model training:
```bash
# Standard pipeline run:
python scripts/run_pipeline.py

# Full benchmark execution across all architectures:
python scripts/run_pipeline.py --benchmark
```

---

## Docker Deployment

Deploy the full stack (PostGIS + FastAPI + Streamlit) via Docker Compose:
```bash
docker-compose up --build -d
```
- **Streamlit Frontend**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Backend**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **PostgreSQL Database**: `localhost:5432`

---

## Project Structure
```text
crop-price-prediction/
├── api/                     # FastAPI backend application & route handlers
│   ├── routes/              # Forecast, price, market, and analytics endpoints
│   ├── schemas.py           # Pydantic v2 data models
│   └── main.py              # Application factory & middleware
├── config/                  # Configuration files (crops, regions, models)
├── dashboard/               # Streamlit interactive visual application
│   ├── components/          # 3D Tarazu landing component & unified zoom map
│   ├── pages/               # Multi-page views (Farmer & Researcher)
│   ├── i18n.py              # Hindi & English internationalization engine
│   └── app.py               # Main landing & portal controller
├── data/                    # Cleaned datasets, spatial topologies, leaderboards
├── models/                  # Saved model registries and scalers (gitignored)
├── scripts/                 # Data pipelines, feature engineering, and training
├── src/                     # Core system package
│   ├── database/            # SQLAlchemy models & connection pool
│   ├── explainability/      # Tree-SHAP interpretability engine
│   ├── features/            # Causal lags, rolling stats, weather, arbitrage
│   ├── inference/           # Production predictor with conformal bounds
│   ├── ingestion/           # AGMARKNET & NASA POWER collectors
│   ├── models/              # PyTorch GNN, TCN, LSTM, Transformer, XGBoost
│   ├── preprocessing/       # Outlier detection, cleaning, and normalization
│   ├── spatial/             # Haversine distance, k-NN graph, GraphSAGE
│   ├── training/            # Chronological walk-forward validation
│   └── utils/               # Centralized configuration & logging
└── tests/                   # Pytest test suite (inference, features, spatial, API)
```
