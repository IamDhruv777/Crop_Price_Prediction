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
6. **Multi-Commodity Evaluation**:
   - Multi-horizon evaluation across 11 major Indian agricultural commodities (Arhar, Black Gram, Cowpea, Green Gram, Kabuli Chana, Lentil, Maize, Rice, Soyabean, Wheat, White Peas).

---

## Multi-Model Architecture & Benchmark Leaderboard

Evaluated on out-of-sample holdout test predictions from AGMARKNET mandi spot prices:

| Horizon | Architecture Tier | Model Architecture | RMSE (₹/Qtl) | MAE (₹/Qtl) | MAPE (%) | $R^2$ Score | Enterprise Trust (100−MAPE) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Day** | Machine Learning | Ridge Autoregressive | 284.40 | 103.65 | 3.11% | 0.9842 | 96.89% |
| **1-Day** | Machine Learning | XGBoost + Spatial KNN | 285.03 | 95.67 | 2.85% | 0.9842 | 97.15% |
| **1-Day** | Machine Learning | Tuned XGBoost Regressor | 285.36 | 96.18 | 2.86% | 0.9841 | 97.14% |
| **1-Day** | Statistical Baseline | Naive Persistence Baseline | 296.10 | 93.95 | 2.81% | 0.9829 | 97.19% |
| **7-Day** | Deep Learning | Temporal CNN (TCN) | 328.17 | 140.23 | 4.00% | 0.9784 | 96.00% |
| **7-Day** | Deep Learning | PyTorch GRU | 329.38 | 141.79 | 4.02% | 0.9783 | 95.98% |
| **7-Day** | Deep Learning | Time-Series Transformer | 330.37 | 141.31 | 4.02% | 0.9781 | 95.98% |
| **7-Day** | Deep Learning | PyTorch LSTM | 330.79 | 142.20 | 4.05% | 0.9781 | 95.95% |
| **7-Day** | Ensemble Meta-Learner | Super Ensemble (Stacking) | 332.22 | 147.98 | 4.26% | 0.9779 | 95.74% |
| **7-Day** | Machine Learning | Tuned XGBoost Regressor | 336.08 | 148.92 | 4.20% | 0.9774 | 95.80% |
| **7-Day** | Statistical Baseline | Naive Persistence Baseline | 363.38 | 141.90 | 4.08% | 0.9735 | 95.92% |
| **30-Day** | Statistical Baseline | Naive Persistence Baseline | 487.92 | 229.82 | 6.48% | 0.9512 | 93.52% |
| **30-Day** | Ensemble Meta-Learner | Super Ensemble (Stacking) | 500.86 | 260.80 | 7.37% | 0.9485 | 92.63% |
| **30-Day** | Machine Learning | Tuned XGBoost Regressor | 542.58 | 283.34 | 7.58% | 0.9396 | 92.42% |

*Note: For 30-Day forecasts, the Naive Persistence Baseline strictly outperforms all ML and deep learning models (Persistence RMSE ₹487.92 vs Model ₹500.86; MAE ₹229.82 vs ₹260.80). Consequently, Naive Persistence is the production-deployed model for 30-Day horizons.*

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
