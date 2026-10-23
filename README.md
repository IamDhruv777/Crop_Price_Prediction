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
   - Calibrated price forecasts across **1-Day**, **7-Day**, and **30-Day** horizons with empirically calibrated conformal prediction intervals ($P_{\text{lower}} \le P_{\text{pred}} \le P_{\text{upper}}$).
   - The farmer dashboard includes a **specific-date planner** for any day up to 30 days after the latest historical trading day. Only days 1, 7, and 30 are direct model outputs; all other dates are clearly labelled as straight-line interpolation between those anchors, not as separately validated daily forecasts.
5. **Phase 2 Benchmark Hub Arbitrage & Supply Shock Signals**:
   - Leading price signals from national benchmark hubs (Raichur, Lalitpur, Mumbai, Merta City, Indore, Burdwan, Kota, Dahod) capturing spatial price contagion and arrival supply-shock dynamics.
6. **Multi-Commodity Evaluation**:
   - Multi-horizon evaluation across 11 major Indian agricultural commodities (Arhar, Black Gram, Cowpea, Green Gram, Kabuli Chana, Lentil, Maize, Rice, Soyabean, Wheat, White Peas).

---

## Multi-Model Architecture & Benchmark Leaderboard

### 1. Canonical Production Deployed Pipeline (Complete-Feature Test Support)
Evaluated on complete feature support from holdout test partition (`data/metrics_deployed.json`):

| Horizon | Deployed Model Architecture | Governing Configuration | Out-of-Sample RMSE (₹/Qtl) | MAE (₹/Qtl) | $R^2$ Score | Commercial Trust (100−MAPE) | Mondrian 95% Coverage |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Day** | **Tuned XGBoost Regressor** | Anchor: $P_t$, Clip: $[-200, 200]$ | **₹283.38** | **₹95.40** | **0.9844** | **97.16%** | **96.14%** |
| **7-Day** | **Tuned XGBoost Regressor** | Anchor: $0.6P_t + 0.4\text{MA}_7$, Clip: $[-400, 400]$ | **₹332.69** | **₹147.03** | **0.9779** | **95.84%** | **95.60%** |
| **30-Day** | **Naive Persistence Baseline** | Anchor: $P_t$, Flat Trajectory | **₹486.53** | **₹228.68** | **0.9516** | **93.54%** | **94.95%** |

*Note on 30-Day Horizon: At 30 days ahead, the Naive Persistence Baseline strictly outperforms all trained classical ML and deep learning models on out-of-sample holdout test data (Persistence RMSE ₹486.53 vs ML ₹542.58; MAE ₹228.68 vs ₹283.34). Consequently, Naive Persistence is registered and served as the canonical deployed baseline for 30-Day horizons.*

### 2. Master Model Architecture Leaderboard (Offline Benchmark Exploration)
Consolidated from full architectural evaluation across classical ML, deep learning sequence models, and statistical baselines (`data/master_model_leaderboard.csv`):

| Horizon | Architecture Tier | Model Architecture | Evaluation Scope | RMSE (₹/Qtl) | MAE (₹/Qtl) | $R^2$ Score | Enterprise Trust (100−MAPE) | APMC Tol (±10%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Day** | Machine Learning | Tuned XGBoost Regressor | Offline Benchmark Only | 283.38 | 95.40 | 0.9844 | 97.16% | 94.66% |
| **1-Day** | Machine Learning | Ridge Autoregressive | Offline Benchmark Only | 284.40 | 103.65 | 0.9842 | 96.89% | 94.88% |
| **1-Day** | Machine Learning | XGBoost + Spatial KNN | Offline Benchmark Only | 285.03 | 95.67 | 0.9842 | 97.15% | 94.65% |
| **1-Day** | Machine Learning | LightGBM Regressor | Offline Benchmark Only | 285.49 | 96.30 | 0.9841 | 97.15% | 94.65% |
| **1-Day** | Ensemble Meta-Learner | Super Ensemble (Stacking) | Offline Benchmark Only | 286.16 | 101.11 | 0.9840 | 97.04% | 94.66% |
| **1-Day** | Machine Learning | Random Forest Regressor | Offline Benchmark Only | 292.02 | 96.87 | 0.9834 | 97.12% | 94.34% |
| **1-Day** | Statistical Baseline | Naive Persistence Baseline | Offline Benchmark Only | 296.10 | 93.95 | 0.9829 | 97.19% | 94.41% |
| **1-Day** | Statistical Baseline | Moving Average (7-day) | Offline Benchmark Only | 321.34 | 130.03 | 0.9799 | 96.25% | 92.01% |
| **7-Day** | Deep Learning | Temporal CNN (TCN) | Offline Benchmark Only | 328.17 | 140.23 | 0.9784 | 96.00% | 91.22% |
| **7-Day** | Deep Learning | PyTorch GRU | Offline Benchmark Only | 329.38 | 141.79 | 0.9783 | 95.98% | 91.14% |
| **7-Day** | Deep Learning | Time-Series Transformer | Offline Benchmark Only | 330.37 | 141.31 | 0.9781 | 95.98% | 90.99% |
| **7-Day** | Deep Learning | PyTorch LSTM | Offline Benchmark Only | 330.79 | 142.20 | 0.9781 | 95.95% | 91.05% |
| **7-Day** | Ensemble Meta-Learner | Super Ensemble (Stacking) | Offline Benchmark Only | 332.22 | 147.98 | 0.9779 | 95.74% | 90.78% |
| **7-Day** | Machine Learning | Tuned XGBoost Regressor | Offline Benchmark Only | 332.69 | 147.03 | 0.9779 | 95.84% | 90.57% |
| **7-Day** | Machine Learning | Random Forest Regressor | Offline Benchmark Only | 334.60 | 146.07 | 0.9776 | 95.84% | 90.48% |
| **7-Day** | Machine Learning | XGBoost + Spatial KNN | Offline Benchmark Only | 338.09 | 150.00 | 0.9771 | 95.78% | 90.37% |
| **7-Day** | Machine Learning | LightGBM Regressor | Offline Benchmark Only | 339.26 | 151.38 | 0.9769 | 95.75% | 90.19% |
| **7-Day** | Machine Learning | Ridge Autoregressive | Offline Benchmark Only | 339.44 | 165.07 | 0.9769 | 95.07% | 89.14% |
| **7-Day** | Statistical Baseline | Naive Persistence Baseline | Offline Benchmark Only | 363.38 | 141.90 | 0.9735 | 95.92% | 90.71% |
| **7-Day** | Statistical Baseline | Moving Average (7-day) | Offline Benchmark Only | 380.13 | 166.96 | 0.9711 | 95.24% | 88.20% |
| **30-Day** | Statistical Baseline | Naive Persistence Baseline | Offline Benchmark Only | 486.53 | 228.68 | 0.9516 | 93.54% | 80.12% |
| **30-Day** | Ensemble Meta-Learner | Super Ensemble (Stacking) | Offline Benchmark Only | 500.86 | 260.80 | 0.9485 | 92.63% | 78.36% |
| **30-Day** | Machine Learning | Ridge Autoregressive | Offline Benchmark Only | 522.66 | 298.51 | 0.9440 | 91.20% | 69.98% |
| **30-Day** | Machine Learning | LightGBM Regressor | Offline Benchmark Only | 536.52 | 280.08 | 0.9410 | 92.50% | 76.47% |
| **30-Day** | Machine Learning | Tuned XGBoost Regressor | Offline Benchmark Only | 542.58 | 283.34 | 0.9396 | 92.42% | 76.05% |
| **30-Day** | Machine Learning | Random Forest Regressor | Offline Benchmark Only | 545.97 | 283.81 | 0.9389 | 92.36% | 75.94% |
| **30-Day** | Machine Learning | XGBoost + Spatial KNN | Offline Benchmark Only | 560.37 | 294.47 | 0.9356 | 92.16% | 74.58% |
| **30-Day** | Statistical Baseline | Moving Average (30-day) | Offline Benchmark Only | 646.45 | 323.52 | 0.9143 | 91.14% | 71.71% |

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

The dashboard opens on a bilingual landing page, then asks you to sign up or log in before showing the personalised farmer / vendor / trader decision screen. Accounts are stored locally in `data/.app_users.json` (password hashes only; the file is git-ignored).

Demo accounts:

| Role | Mobile | Password |
|:---|:---|:---|
| Farmer | `9999999999` | `kisan1234` |
| Shopkeeper / Vendor | `8888888888` | `vyapari1234` |

The dashboard is a historical backtest demo over Agmarknet data (Oct 2016 – Jun 2017). It is not a live market feed.

#### Launch FastAPI Backend Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Explore interactive Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Running Automated Tests

Run the comprehensive 70-test automated test suite covering inference bounds, deployment parity, zero-leakage causal lags, spatial distance, API endpoints, dashboard authentication, the date planner, and the high-zoom map:
```bash
pytest tests/
```

### Verify the dashboard's claims yourself

Run the independent verifier from the repository root:
```bash
python scripts/verify_dashboard_claims.py
```

It recomputes the deployed RMSE, MAE, $R^2$, sample counts, and conformal coverage directly from `data/deployed_test_predictions.csv`; compares them with `data/metrics_deployed.json`; prints SHA-256 hashes of the model/data artifacts; checks the 1/7/30-day date-planner anchors; and confirms the high-zoom map guards are present. It also prints what the checks do **not** prove: live-price accuracy, an individual farmer's guaranteed 95% band, or genuine daily models for days 2-6 and 8-29.

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
