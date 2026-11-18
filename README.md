# Krishi Bhav Saathi (कृषि भाव साथी)
### *Region-Wise AI Crop Price Prediction, Spatial Arbitrage & Conformal Decision Engine*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-GraphSAGE-3C2179.svg)](https://pyg.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Dual--Head-eb5424.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20REST-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Ready-2496ED.svg)](https://www.docker.com/)
[![Tests Passing](https://img.shields.io/badge/Pytest-76%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Conformal Coverage](https://img.shields.io/badge/Conformal%20Coverage-96.46%25-success.svg)](data/mondrian_conformal_quantiles.json)
[![R2 Score](https://img.shields.io/badge/Champion%20R%C2%B2-0.9784-blueviolet.svg)](data/master_model_leaderboard.csv)

---

## 📑 Table of Contents
1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [System Architecture & 8-Stage Pipeline](#2-system-architecture--8-stage-pipeline)
3. [Technology Stack & Frameworks](#3-technology-stack--frameworks)
4. [Datasets, Geography & Ingestion](#4-datasets-geography--ingestion)
5. [Feature Engineering (48 Signals)](#5-feature-engineering-48-signals)
6. [Spatial Graph Modeling & Market Network](#6-spatial-graph-modeling--market-network)
7. [AI/ML Forecasting Engine & Model Tournament](#7-aiml-forecasting-engine--model-tournament)
8. [Mathematical Formulations & Objective Metrics](#8-mathematical-formulations--objective-metrics)
9. [Uncertainty Quantification & Conformal Guarantees](#9-uncertainty-quantification--conformal-guarantees)
10. [Empirical Results & Comprehensive Benchmark Leaderboard](#10-empirical-results--comprehensive-benchmark-leaderboard)
11. [Interactive Web Dashboard & UX Architecture](#11-interactive-web-dashboard--ux-architecture)
12. [FastAPI Backend & REST API Specification](#12-fastapi-backend--rest-api-specification)
13. [Quickstart, Installation & Docker Deployment](#13-quickstart-installation--docker-deployment)
14. [Automated Testing & Verification Suite](#14-automated-testing--verification-suite)
15. [Repository Directory Tree](#15-repository-directory-tree)
16. [Contributing, Citation & License](#16-contributing-citation--license)

---

## 1. Project Overview & Motivation

Agriculture sustains over **58% of India's rural households**, yet Indian farmers routinely suffer catastrophic financial losses during harvest cycles due to **extreme wholesale price volatility**, **asymmetric market information**, and **exploitation by regional middlemen**. In typical Agricultural Produce Market Committee (APMC) mandis, wholesale prices for major staples fluctuate wildly within a single 7-day period due to supply gluts, transportation bottlenecks, and unseasonal weather events.

Traditional forecasting methods either rely on naive national averages or standalone statistical regressions that fail to model:
1. **Spatial Price Contagion:** Price shifts in major trading hubs (e.g., Indore, Raichur, Kota) trigger lagged price cascades in surrounding regional mandis.
2. **Asymmetric Risk & Conformal Safety:** A single-point price guess (e.g., "₹2,200") provides zero guarantee to a smallholder farmer who faces bankruptcy if the real market price crashes below production cost.
3. **Cognitive Overload:** Complex econometrics confuse rural farmers who simply need to know: *"Should I sell my trolley today, or hold for next week?"*

**Krishi Bhav Saathi (कृषि भाव साथी)** is a full-stack, enterprise-grade AI forecasting and spatial arbitrage decision platform. It integrates **10-Year Daily AGMARKNET wholesale transactions**, **NASA POWER agroclimatological data**, and a **2,515-mandi spatial graph network** to deliver:
* **High-Precision Multi-Horizon Forecasts:** 1-Day, 7-Day, and 30-Day horizons with **96.0% price accuracy** ($R^2 = 0.978$).
* **Distribution-Free 95% Conformal Safety Bands:** Calibrated intervals empirically containing **96.46%** of all future market prices.
* **Dual-Head Directional Classification:** High-precision trend signals (**Hold** vs **Sell**) with conformal uncertainty gating and deadband noise filtering ($\pm 2.0\%$).
* **Dual-Audience Bilingual Web Dashboard:** Designed for rural farmers in plain Hindi/English with actionable **Trolley Arbitrage Net Profit Calculators** (deducting diesel expenses) alongside a research-grade econometric laboratory.

---

## 2. System Architecture & 8-Stage Pipeline

The platform is designed around a strict, leak-free 8-stage end-to-end data and modeling pipeline:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           KRISHI BHAV SAATHI — END-TO-END SYSTEM PIPELINE                                      │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
  [1. Data Collection]        [2. Cleaning & Preprocessing]     [3. Feature Engineering]     [4. Spatial Graph Modeling]
  • 10-Yr AGMARKNET Records    • Outlier Winsorization           • 48 Spatial-Temporal Signals • 2,515 Mandi Nodes
  • NASA POWER Daily Weather   • Zero Data Leakage Timeline      • Autoregressive Price Lags   • Haversine Distance Edges
  • APMC GPS Coordinates       • Missing Resampling              • Arrival & Weather Stress    • GraphSAGE / Spatial KNN
            │                               │                               │                            │
            ▼                               ▼                               ▼                            ▼
  [5. Forecasting Engine]     [6. Training & Validation]        [7. Prediction & Inference]  [8. UI & Decision Engine]
  • Statistical Baselines      • Walk-Forward Purge Splits       • Multi-Horizon (1D/7D/30D)  • Bilingual Web Dashboard
  • Machine Learning (XGB/RF)  • Optuna Bayesian Tuning          • 95% Conformal Coverage     • 3D Tarazu WebGL Three.js
  • Deep Learning (TCN/GRU)    • Evaluation on Holdout           • Dual-Head Trend Classifier • Live Nov/Dec 2026 Planner
```

### Detailed Pipeline Breakdown
* **Stage 1: Multi-Modal Data Ingestion:** Automated ingestion of daily mandi arrivals, modal wholesale prices, and meteorological signals across 2,515 APMC mandis.
* **Stage 2: Strict Chronological Preprocessing:** Zero data leakage guarantee. No random shuffling; strict chronological partitioning ensuring the model only uses historical signals to predict future days.
* **Stage 3: Feature Engineering:** Extraction of 48 domain-engineered features capturing autoregressive price momentum, arrival supply shocks, cyclical seasonality, and agroclimatic stress.
* **Stage 4: Region-Wise Spatial Modeling:** Topological graph construction connecting nearby mandis via geographical distance to capture cross-mandi price contagion and trade flows.
* **Stage 5: Multi-Model Tournament:** Cross-architectural competition spanning classical baselines, gradient-boosted decision trees, deep sequential networks, and stacking ensembles.
* **Stage 6: Chronological Walk-Forward Validation:** Strict time-series validation with 30-day purge embargos preventing temporal lookahead bias.
* **Stage 7: Production Inference & Conformal Gating:** Serving multi-horizon price points wrapped in Mondrian conformal intervals and dual-head directional classifications with false-positive deadband filtering.
* **Stage 8: Decision Dashboard & REST API:** Multi-page Streamlit dashboard featuring simple farmer and advanced researcher modes, an interactive India map, and high-concurrency FastAPI microservices.

---

## 3. Technology Stack & Frameworks

| Domain | Technologies & Libraries | Purpose |
| :--- | :--- | :--- |
| **Core AI & Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` | Tabular gradient boosted regressors, Stacking ensembles, Dual-head directional classifiers |
| **Deep Learning Sequence Models** | `PyTorch`, `torchvision`, `torchaudio` | Temporal Convolutional Networks (TCN), PyTorch GRU, LSTM, and Multi-Head Transformers |
| **Spatial Graph Neural Networks** | `torch-geometric` (PyG), `dgl`, `networkx`, `scipy.spatial` | GraphSAGE message passing, $k$-NN spatial graph construction, Haversine distance decay |
| **Statistical & Econometric** | `statsmodels`, `pmdarima` | Auto-ARIMA, SARIMAX, and rolling baseline benchmarks |
| **Hyperparameter Optimization** | `optuna` | Bayesian Optimization via Tree-structured Parzen Estimator (TPE) |
| **Model Explainability (XAI)** | `shap` | Tree-SHAP feature importance (Beeswarm, Waterfall, Dependence plots) |
| **Backend & Microservices** | `FastAPI`, `uvicorn`, `pydantic v2` | High-throughput asynchronous REST API for forecasts and spatial arbitrage |
| **Frontend & Visualization** | `Streamlit`, `Plotly Express`, `Plotly Graph Objects` | Interactive bilingual web dashboard, candlestick charts, econometric distributions |
| **3D Graphics & WebGL** | `Three.js` | Custom 3D animated agricultural balance scale (*Tarazu*) for rural cultural trust |
| **GIS & Geospatial Maps** | `Leaflet.js`, `TopoJSON-client`, `geopandas`, `shapely`, `folium` | Hierarchical zoom-responsive interactive map (States $\to$ Districts $\to$ APMC Mandis) |
| **Database & Persistence** | `PostgreSQL 15`, `PostGIS 3.3`, `SQLAlchemy`, `psycopg2-binary` | Spatial relational database for mandi geo-coordinates and historical records |
| **DevOps & Verification** | `Docker`, `docker-compose`, `pytest`, `python-dotenv` | Containerized microservices, 76 automated tests, independent verification scripts |

---

## 4. Datasets, Geography & Ingestion

### 1. 10-Year AGMARKNET Historical Master Dataset (2008–2017)
Public government APIs (such as `data.gov.in`) suffer from strict throttling, request quotas, and high server timeout rates when querying multi-year bulk data across thousands of mandis. To ensure complete scientific reproducibility, we utilized the verified **10-Year Open-Source AGMARKNET Master Dataset** curated by researcher **Ian Covert** ([GitHub: `iancovert/Agmarknet`](https://github.com/iancovert/Agmarknet)).
* **Coverage:** 10 continuous years (2008–2017) of daily wholesale transactions.
* **Volume:** Over 3.2 million rows (~400 MB) across all 36 Indian States and Union Territories.
* **11 Major Commodities:**
  1. **Wheat (गेहूं)**
  2. **Rice (चावल / धान)**
  3. **Maize (मक्का)**
  4. **Soyabean (सोयाबीन)**
  5. **Arhar / Tur-Red Gram (अरहर / तुअर)**
  6. **Green Gram / Moong (मूंग)**
  7. **Black Gram / Urd Beans (उड़द)**
  8. **Lentil / Masur (मसूर)**
  9. **Cowpea / Lobia (लोबिया)**
  10. **Kabuli Chana (काबुली चना)**
  11. **White Peas (सफेद मटर)**
* **Recorded Attributes:** `date`, `state`, `district`, `market`, `commodity`, `variety`, `arrivals_tonnes`, `min_price`, `max_price`, and `modal_price`.

### 2. NASA POWER Agroclimatology API
Daily meteorological signals fetched at the district centroid level:
* **Precipitation (`PRECTOTCORR`):** Daily rainfall (mm/day).
* **Maximum & Minimum Temperature (`T2M_MAX`, `T2M_MIN`):** Thermal stress signals (°C).
* **Relative Humidity (`RH2M`):** Atmospheric moisture content (%).
* **Solar Radiation (`ALLSKY_SFC_SW_DWN`):** Sunlight intensity (MJ/m²/day).

### 3. APMC Market Master & Geospatial Topography
* **2,515 Verified APMC Mandis** mapped with exact GPS coordinates (Latitude & Longitude).
* **723 Agricultural Districts** with geometric polygon centroids.
* Centroid-calibrated GeoJSON and TopoJSON layers for nationwide zoom drilldown.

---

## 5. Feature Engineering (48 Signals)

The model ingests **48 domain-engineered features** designed to capture every dimension of price formation:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          48 DOMAIN-ENGINEERED MODEL FEATURES                           │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Feature Category         │ Engineered Signals & Mathematical Description               │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Autoregressive Lags   │ modal_price_lag_1, lag_2, lag_3, lag_7, lag_14, lag_30      │
│ 2. Moving Statistics     │ rolling_mean_7, rolling_mean_14, rolling_mean_30,           │
│                          │ rolling_std_7, rolling_std_30, rolling_skew_30              │
│ 3. Price Momentum        │ price_momentum_7d, price_momentum_30d, bollinger_high_30,  │
│                          │ bollinger_low_30, bollinger_pct_b, price_acceleration       │
│ 4. Arrival Dynamics      │ arrivals_tonnes, arrival_lag_1, arrival_lag_7,              │
│                          │ arrival_rolling_mean_7, arrival_shock_ratio (arrivals/mean) │
│ 5. Temporal & Cyclical   │ day_of_week, day_of_year, month, year,                      │
│                          │ sin_day_of_year, cos_day_of_year, sin_month, cos_month      │
│ 6. Agroclimatic Weather  │ rainfall_mm, rainfall_lag_1, rainfall_rolling_sum_7,        │
│                          │ temp_max, temp_min, temp_range, humidity_avg, weather_shock │
│ 7. Spatial Network Lags  │ neighbor_price_mean_k5, neighbor_price_min_k5,             │
│                          │ neighbor_price_max_k5, spatial_price_spread_k5,             │
│                          │ regional_hub_price_lag_1, hub_spread_ratio                  │
│ 8. Market Characteristics│ mandi_encoded, district_encoded, state_encoded, crop_encoded│
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 6. Spatial Graph Modeling & Market Network

Mandis do not operate in silos. If prices surge in a major trading hub (e.g., Indore for Soyabean), surrounding regional mandis experience upward price contagion within 24 to 72 hours.

### Spatial Network Formulation
* **Nodes ($V$):** 2,515 APMC Mandis across India.
* **Edges ($E$):** Pairwise connections between mandis weighted by inverse geographic distance.
* **Haversine Distance Metric:**
  $$d(u, v) = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_u)\cos(\phi_v)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
* **Exponential Distance-Decay Adjacency Matrix:**
  $$W_{uv} = \begin{cases} \exp\left(-\frac{d(u,v)^2}{2\sigma^2}\right) & \text{if } v \in \mathcal{N}_k(u) \\ 0 & \text{otherwise} \end{cases}$$
  *(where $\mathcal{N}_k(u)$ represents the $k$-nearest geographical neighbor mandis, $k=5$)*.

### Spatial Feature Aggregation (GraphSAGE / Spatial KNN)
For each target mandi $u$ on day $t$, the spatial feature vector $\mathbf{h}_{\mathcal{N}(u), t}$ is generated by aggregating price and arrival states from neighboring markets:
$$\mathbf{h}_{\mathcal{N}(u), t} = \sum_{v \in \mathcal{N}(u)} \frac{W_{uv}}{\sum_{w \in \mathcal{N}(u)} W_{uw}} \mathbf{x}_{v, t}$$
This injects real-time spatial arbitrage pressure into both tree-based models and graph neural networks.

---

## 7. AI/ML Forecasting Engine & Model Tournament

To avoid model bias, the system trains and benchmarks **12 distinct model architectures** across 4 major paradigm tiers:

```
                         ┌──────────────────────────────────────────────────────────┐
                         │               MODEL ARCHITECTURE TIERS                   │
                         └──────────────────────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┬─────────────────────────┐
         ▼                         ▼                         ▼                         ▼
   [Statistical]           [Classical ML]            [Deep Learning]          [Meta-Ensembles]
   • Naive Persistence     • Tuned XGBoost           • Temporal CNN (TCN)     • Stacking Ensemble
   • 7-Day Moving Avg      • LightGBM Regressor      • PyTorch GRU            • Dual-Head XGBoost
   • 30-Day Moving Avg     • Random Forest           • PyTorch LSTM             Directional
   • SARIMAX Baseline      • Ridge Autoregressive    • Time-Series Transformer  Classifiers
```

### Deep Sequential Model Details
* **Temporal Convolutional Network (TCN):** Utilizes dilated causal 1D convolutions with residual blocks. Dilations ($d = [1, 2, 4, 8, 16]$) expand the receptive field to 60 historical days without recurrent bottlenecks or vanishing gradients.
* **PyTorch GRU & LSTM:** Bidirectional recurrent networks with temporal self-attention heads capturing long-term crop seasonality.
* **Time-Series Transformer:** Multi-head self-attention ($h=4$, $d_{\text{model}}=64$) with sinusoidal positional encodings mapping temporal interactions across 30-day lookback sequences.
* **Dual-Head Directional Classifiers (`XGBClassifier`):** Specialized binary classification heads trained explicitly on directional return targets ($y_{\text{dir}} = \mathbb{I}(P_{t+h} > P_t)$) with log-loss optimization to resolve sign-flip instabilities.

---

## 8. Mathematical Formulations & Objective Metrics

### Regression Error Formulations
1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
3. **Mean Absolute Percentage Error (MAPE):**
   $$\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$
4. **Commercial Price Accuracy:**
   $$\text{Accuracy}_{\text{Price}} = 100\% - \text{MAPE}$$
5. **Coefficient of Determination ($R^2$ Score):**
   $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

### Directional & Classification Metrics
1. **Raw Directional Accuracy:**
   $$\text{DA} = \frac{1}{n} \sum_{i=1}^n \mathbb{I}\left( \operatorname{sgn}(\hat{y}_{i, t+h} - y_{i, t}) == \operatorname{sgn}(y_{i, t+h} - y_{i, t}) \right)$$
2. **Precision, Recall & F1 Score:**
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
3. **Intersection over Union (IoU) for Interval Forecasting:**
   $$\text{IoU} = \frac{\mu\left([\hat{L}_i, \hat{U}_i] \cap [L_i, U_i]\right)}{\mu\left([\hat{L}_i, \hat{U}_i] \cup [L_i, U_i]\right)}$$
   *(Measuring the overlap between predicted conformal bounds and realized price volatility intervals)*.

---

## 9. Uncertainty Quantification & Conformal Guarantees

Point forecasts without uncertainty bounds are dangerous for farmers. Krishi Bhav Saathi implements **Inductive Split Conformal Prediction (Mondrian Per-Crop Calibration)** to guarantee distribution-free coverage.

```
       Lower Bound                       Point Forecast                     Upper Bound
   [ L_i = ŷ_i - Δ_conf ] <---------------- [ ŷ_i ] ----------------> [ U_i = ŷ_i + Δ_conf ]
   ├───────────────────────────────────────────────────────────────────────────────────────┤
   │                  95% CALIBRATED CONFORMAL CONFIDENCE ENVELOPE                         │
   │               Empirical Coverage on Unseen Holdout: 96.46% (Meets Guarantee)          │
   └───────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Conformal Quantile Formulation
Given a calibration set $\mathcal{D}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^m$, we compute non-conformity residuals:
$$s_i = |y_i - \hat{\mu}(x_i)|$$
For a targeted coverage guarantee $1 - \alpha = 0.95$, the conformal quantile $\hat{q}$ is derived via:
$$\hat{q} = \text{Quantile}\left( \frac{\lceil (m+1)(1-\alpha) \rceil}{m}, \; \{s_i\}_{i=1}^m \right)$$
The resulting predictive interval for any new test sample $x_{n+1}$ satisfies:
$$\mathbb{P}\left( y_{n+1} \in \left[ \hat{\mu}(x_{n+1}) - \hat{q}, \; \hat{\mu}(x_{n+1}) + \hat{q} \right] \right) \ge 1 - \alpha$$

### 2. Conformal Uncertainty Gating & Deadband Noise Suppression
* **Uncertainty Gating:** If the model's conformal width $\Delta_{\text{conf}}$ exceeds the volatility threshold $\tau_{\text{safe}}$, the system flags high uncertainty and advises the farmer to seek spot market quotes.
* **Deadband Noise Suppression ($\pm 2.0\%$):** Micro-price noise between $-2.0\%$ and $+2.0\%$ is suppressed into **"🔵 Stable Equilibrium (स्थिर)"**, preventing false trading alarms and raising actionable directional precision to **$>85\%$**.

---

## 10. Empirical Results & Comprehensive Benchmark Leaderboard

### Table 1: Primary 7-Day Horizon Full Architecture Leaderboard
Evaluated on the out-of-sample holdout test partition (`data/master_model_leaderboard.csv`):

| Rank | Model Architecture | Family | $R^2$ Score | RMSE (₹/qtl) | MAE (₹/qtl) | MAPE (%) | Price Accuracy | Directional Acc | F1 Score | 95% Conformal Cov |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Temporal CNN (TCN)** | Deep Learning | **0.9784** | **₹328.17** | **₹140.23** | **4.00%** | **96.00%** | 59.92% | 0.6582 | — |
| **2** | **PyTorch GRU** | Deep Learning | 0.9783 | ₹329.38 | ₹141.79 | 4.02% | 95.98% | 58.67% | 0.6405 | — |
| **3** | **Time-Series Transformer** | Deep Learning | 0.9781 | ₹330.37 | ₹141.31 | 4.02% | 95.98% | 60.65% | 0.6751 | — |
| **4** | **PyTorch LSTM** | Deep Learning | 0.9781 | ₹330.79 | ₹142.20 | 4.05% | 95.95% | 59.65% | 0.6587 | — |
| **5** | **Super Ensemble (Stacking)** | Ensemble Meta | 0.9779 | ₹332.22 | ₹147.98 | 4.26% | 95.74% | 49.23%*| 0.4450*| **96.46%** |
| **6** | **Tuned XGBoost Regressor** | Machine Learning | 0.9779 | ₹332.69 | ₹147.03 | 4.16% | 95.84% | 60.01% | 0.6759 | — |
| **7** | **Random Forest Regressor** | Machine Learning | 0.9776 | ₹334.60 | ₹146.07 | 4.16% | 95.84% | 61.67% | 0.6968 | — |
| **8** | **XGBoost + Spatial KNN** | Spatial ML | 0.9771 | ₹338.09 | ₹150.00 | 4.22% | 95.78% | 61.47% | 0.6949 | — |
| **9** | **LightGBM Regressor** | Machine Learning | 0.9769 | ₹339.26 | ₹151.38 | 4.25% | 95.75% | 59.60% | 0.6695 | — |
| **10** | **Ridge Autoregressive** | Linear ML | 0.9769 | ₹339.44 | ₹165.07 | 4.93% | 95.07% | 61.81% | 0.7354 | — |
| **11** | **Naive Persistence Baseline** | Statistical | 0.9735 | ₹363.38 | ₹141.90 | 4.08% | 95.92% | 59.05% | 0.7425 | — |
| **12** | **Moving Average (7-Day)** | Statistical | 0.9711 | ₹380.13 | ₹166.96 | 4.76% | 95.24% | 62.35% | 0.6824 | — |

*\*Note on Stacking Directional Score: Resolved in production via our dedicated Dual-Head Directional Classifier (`XGBClassifier`) which operates above the $\pm 2.0\%$ deadband, achieving $82\%–88\%$ actionable directional precision.*

---

### Table 2: Multi-Horizon Progression (1-Day vs 7-Day vs 30-Day)
Shows how predictive variance and errors scale across short, medium, and monthly planning horizons:

| Forecast Horizon | Deployed Model | $R^2$ Score | RMSE (₹/qtl) | MAE (₹/qtl) | MAPE (%) | Price Accuracy | Conformal 95% Cov | APMC Tolerance (±10%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Day Ahead** | **Tuned XGBoost** | **0.9844** | **₹283.38** | **₹95.40** | **2.84%** | **97.16%** | **96.18%** | **94.66%** |
| **7-Day Ahead** | **Tuned XGBoost / TCN**| **0.9779** | **₹332.69** | **₹147.03** | **4.16%** | **95.84%** | **95.90%** | **90.57%** |
| **30-Day Ahead** | **Naive Persistence** | **0.9516** | **₹486.53** | **₹228.68** | **6.46%** | **93.54%** | **96.03%** | **80.12%** |

*Methodological Finding on 30-Day Horizon:* At 30 days ahead, Naive Persistence strictly outperforms all trained classical ML and deep learning models on out-of-sample holdout test data (Persistence RMSE ₹486.53 vs ML ₹542.58). In keeping with strict scientific integrity, Naive Persistence is registered and deployed for 30-day horizons.

---

### Table 3: Per-Crop Performance Breakdown (All 11 Commodities)
Audited out-of-sample metrics extracted from `data/metrics_deployed_by_crop.csv`:

| Commodity | Test Samples (1D / 7D / 30D) | 1D RMSE | 1D Accuracy | 7D RMSE | 7D Accuracy | 30D RMSE | 30D Accuracy | Conformal Coverage (1D / 7D / 30D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wheat (गेहूं)** | 78,165 / 84,808 / 71,535 | ₹121.45 | 97.76% | ₹118.13 | 96.88% | ₹174.39 | 94.27% | 94.9% / 93.3% / 88.4% |
| **Rice (चावल / धान)** | 38,788 / 41,454 / 36,747 | ₹190.38 | 98.35% | ₹204.77 | 97.63% | ₹291.73 | 96.36% | 95.9% / 95.6% / 95.9% |
| **Maize (मक्का)** | 38,891 / 43,820 / 37,846 | ₹158.83 | 97.31% | ₹150.59 | 96.63% | ₹188.14 | 95.13% | 95.5% / 95.2% / 96.4% |
| **Soyabean (सोयाबीन)** | 34,392 / 38,308 / 33,628 | ₹134.49 | 97.76% | ₹147.75 | 97.16% | ₹198.47 | 96.32% | 96.9% / 96.7% / 98.0% |
| **Arhar (Tur-Red Gram)** | 27,825 / 30,996 / 26,175 | ₹362.32 | 96.21% | ₹491.22 | 93.26% | ₹733.78 | 90.03% | 96.8% / 96.7% / 98.5% |
| **Black Gram (Urd Beans)**| 29,670 / 32,905 / 28,510 | ₹496.08 | 95.46% | ₹597.31 | 92.97% | ₹873.29 | 89.38% | 98.7% / 98.9% / 99.0% |
| **Green Gram (Moong)** | 22,037 / 24,765 / 21,036 | ₹339.67 | 96.01% | ₹413.58 | 94.34% | ₹587.35 | 91.81% | 96.8% / 96.9% / 97.5% |
| **Lentil (Masur)** | 15,601 / 17,121 / 14,442 | ₹281.55 | 97.32% | ₹346.68 | 95.16% | ₹570.80 | 91.38% | 96.1% / 95.8% / 96.4% |
| **Cowpea (Lobia)** | 1,848 / 2,128 / 1,805 | ₹464.68 | 95.21% | ₹547.49 | 93.29% | ₹847.94 | 89.61% | 95.9% / 95.8% / 95.5% |
| **Kabuli Chana** | 2,851 / 3,157 / 2,646 | ₹1178.72 | 92.17% | ₹1186.15 | 90.48% | ₹1728.37 | 85.42% | 92.6% / 91.1% / 91.7% |
| **White Peas** | 822 / 868 / 739 | ₹356.40 | 94.80% | ₹356.14 | 94.15% | ₹455.84 | 91.57% | 97.7% / 98.8% / 99.0% |

---

## 11. Interactive Web Dashboard & UX Architecture

The platform features an interactive Streamlit web dashboard structured into distinct user views:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          DUAL-AUDIENCE UX ARCHITECTURE                                 │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ 🌾 SIMPLE FARMER VIEW (सरल किसान दृश्य)   │ 🔬 ADVANCED RESEARCHER VIEW (उन्नत शोध दृश्य)│
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • 1-Click Action Advice (Hold vs Sell)   │ • Econometric Dispersion & Volatility       │
│ • Trolley Arbitrage Profit Calculator    │ • 12-Month Longitudinal Boxplots & Outliers │
│   (Net Gain after ₹2,500 Diesel Deduct)  │ • Multi-Horizon Rolling Baselines (7D/14D)  │
│ • Optimal Selling Month Calendar         │ • Full Model Benchmark Leaderboard          │
│ • Top 3 Recommended APMC Podium Cards    │ • Spatial Spread & Regional Premia Analysis │
│ • 10-Year Annual Crop Growth Trendline   │ • SHAP Tree-Explainer Feature Attribution   │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

### Key UI Capabilities
1. **Interactive 3D Mandi Tarazu (Weighing Scale):** Built with WebGL and Three.js on the landing portal, symbolizing honest weights and transparent price discovery.
2. **Hierarchical Unified India Map:** Multi-level zoom transitions from **State Level (Zoom <6)** $\to$ **District Centroids (Zoom 6 to 8.8)** $\to$ **APMC Mandi Pins (Zoom $\ge$ 8.8)**.
3. **Specific Date Price Planner:**
   * **`📅 Live 30-Day Planning (Today: Nov/Dec 2026)`**: Anchors to current calendar dates. Allows selecting any day (e.g. 23 Nov, 30 Nov, or Dec) with safe bounds.
   * **`🏛️ Historical Backtest Audit (2017 Archive)`**: Instant toggle to view historical ground truth validation.
4. **Bilingual Localization Engine (`i18n.py`):** Seamless toggle between **हिन्दी (Hindi)** and **English** across all pages, metrics, and farmer alerts.

---

## 12. FastAPI Backend & REST API Specification

The platform includes an asynchronous FastAPI service (`api/main.py`) running on `Uvicorn`:

### Interactive Swagger Docs
Open `http://localhost:8000/docs` to test endpoints via the interactive OpenAPI UI.

### Core Endpoints

#### 1. Multi-Horizon Crop Price Forecast
* **`POST /api/v1/forecast`**
* **Request Body:**
  ```json
  {
    "crop": "Wheat",
    "market": "Indore",
    "state": "Madhya Pradesh",
    "district": "Indore",
    "horizon": "7d",
    "confidence_level": 0.95
  }
  ```
* **Response (200 OK):**
  ```json
  {
    "crop": "Wheat",
    "market": "Indore",
    "horizon": "7d",
    "predicted_price": 2340.50,
    "lower_bound": 2210.00,
    "upper_bound": 2470.00,
    "conformal_delta": 130.00,
    "directional_signal": "HOLD",
    "drift_pct": 3.85,
    "confidence_level": 0.95
  }
  ```

#### 2. Spatial Arbitrage & Net Margin Calculator
* **`POST /api/v1/arbitrage`**
* **Request Body:**
  ```json
  {
    "crop": "Soyabean",
    "source_market": "Dewas",
    "target_market": "Indore",
    "trolley_quintals": 50.0,
    "diesel_cost_inr": 2500.0
  }
  ```
* **Response (200 OK):**
  ```json
  {
    "source_market": "Dewas",
    "target_market": "Indore",
    "source_price_per_qtl": 3600.00,
    "target_price_per_qtl": 3720.00,
    "price_spread_per_qtl": 120.00,
    "gross_arbitrage_gain": 6000.00,
    "diesel_transport_cost": 2500.00,
    "net_profit_inr": 3500.00,
    "recommendation": "TRANSPORT_PROFITABLE"
  }
  ```

#### 3. Health & Readiness Probe
* **`GET /api/v1/health`**
* Returns system status, loaded model registry version, and spatial graph connectivity.

---

## 13. Quickstart, Installation & Docker Deployment

### 1. Prerequisites
* **Python:** Version 3.10, 3.11, 3.12, or 3.14
* **Git:** Installed and on PATH
* **Docker & Docker Compose:** *(Optional, for containerized run)*

### 2. Local Setup via Python venv
```bash
# 1. Clone repository
git clone https://github.com/IamDhruv777/Crop_Price_Prediction.git
cd Crop_Price_Prediction

# 2. Create and activate virtual environment
python -m venv venv
# Windows (CMD / PowerShell):
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt
```

### 3. Launching Services

#### Option A: One-Click Windows Launcher
Double-click `run.bat` in the repository root, or run:
```cmd
run.bat
```
*(Starts both FastAPI on port 8000 and Streamlit on port 8501 automatically).*

#### Option B: Manual Command Line Launch
* **Terminal 1 — Launch FastAPI Server:**
  ```bash
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
* **Terminal 2 — Launch Streamlit Dashboard:**
  ```bash
  streamlit run dashboard/app.py --server.port 8501
  ```

Open [http://localhost:8501](http://localhost:8501) in your browser.

#### Demo Credentials:
| Persona / Role | Mobile Number | Password | Target User View |
| :--- | :--- | :--- | :--- |
| **Smallholder Farmer** | `9999999999` | `kisan1234` | Simple Farmer View (सरल किसान दृश्य) |
| **Village Trader / Vendor** | `8888888888` | `vyapari1234` | Arbitrage & Mandi Comparisons |
| **Agricultural Researcher** | `7777777777` | `research123` | Advanced Researcher Analytics Workbench |

---

### 4. Docker Compose Multi-Container Deployment
Deploy the full microservices stack (PostGIS Database + FastAPI + Streamlit):
```bash
docker-compose up --build -d
```
* **Streamlit UI:** `http://localhost:8501`
* **FastAPI Docs:** `http://localhost:8000/docs`
* **PostgreSQL / PostGIS:** `localhost:5432`

---

## 14. Automated Testing & Verification Suite

The repository includes an automated test suite (**76 unit, integration, and audit tests**) ensuring strict deployment parity, zero data leakage, and conformal calibration:

```bash
# Run complete test suite
pytest tests/ -v
```

### Independent Claims Verification Script
Verify all published metrics, artifact hashes, and deployment claims directly against the holdout dataset:
```bash
python scripts/verify_dashboard_claims.py
```
This script independently calculates RMSE, MAE, $R^2$, sample counts, and conformal coverage directly from `data/deployed_test_predictions.csv` and verifies SHA-256 hashes against `data/metrics_deployed.json`.

---

## 15. Repository Directory Tree

```text
Crop_Price_Prediction/
├── api/                           # FastAPI backend microservice
│   ├── routes/                    # API route controllers (forecasts, arbitrage, prices)
│   ├── main.py                    # ASGI application factory & CORS configuration
│   └── schemas.py                 # Pydantic data validation contracts
├── config/                        # Global system configuration (YAML)
├── dashboard/                     # Multi-page interactive Streamlit dashboard
│   ├── components/                # 3D Tarazu Three.js & unified zoom map components
│   ├── pages/                     # Application pages (Overview, Forecasting, Map, Trends, AI)
│   ├── app.py                     # Main dashboard entrypoint & authentication router
│   ├── auth.py                    # Role-based user authentication & session management
│   └── i18n.py                    # Bilingual (Hindi / English) internationalization engine
├── data/                          # Datasets, geospatial boundaries, calibration quantiles
│   ├── raw/                       # Market Master coordinates & weather caches
│   ├── deployed_test_predictions.csv # Test partition holdout predictions & bounds
│   ├── master_model_leaderboard.csv  # 12-model architectural evaluation benchmark
│   └── mondrian_conformal_quantiles.json # Calibrated per-crop conformal deltas
├── models/                        # Serialized weights, scalers, and deployment registries
│   └── trained/                   # Model registry configs & directional heads
├── reports/                       # Forensic audit scorecards and model cards
│   ├── FINAL_SCORECARD.md         # Audited metrics scorecard
│   └── MODEL_CARD.md              # Detailed architecture specifications
├── scripts/                       # Pipelines, training suites, and verification tools
│   ├── calibrate_mondrian_conformal.py # Conformal quantile calibration
│   ├── download_bulk_data.py      # AGMARKNET bulk archive downloader
│   ├── run_pipeline.py            # Master pipeline runner
│   ├── train_complete_deep_learning_suite.py # TCN, GRU, LSTM, Transformer trainer
│   └── verify_dashboard_claims.py # Independent claims verifier
├── src/                           # Core reusable Python package
│   ├── database/                  # SQLAlchemy ORM models & connection pool
│   ├── explainability/            # SHAP feature importance analysis
│   ├── features/                  # Lag creation, rolling statistics, weather features
│   ├── inference/                 # Predictor engine with conformal bands & planning paths
│   ├── models/                    # PyTorch models, XGBoost, and Stacking ensembles
│   ├── preprocessing/             # Outlier handling & chronological splitting
│   └── spatial/                   # Haversine distance, k-NN graph, GraphSAGE
├── tests/                         # Pytest test suite (76 automated tests)
├── .gitignore                     # Git exclusions
├── docker-compose.yml             # Multi-container Docker deployment specification
├── Dockerfile                     # Microservices Docker build instructions
├── requirements.txt               # Pinned Python package dependencies
├── run.bat                        # Windows one-click launcher
└── README.md                      # Comprehensive project documentation
```

---

## 16. Contributing, Citation & License

### Contributing
Contributions, bug reports, and pull requests are welcome!
1. Fork the Project repository.
2. Create a Feature Branch (`git checkout -b feature/SpatialContagionModel`).
3. Commit your Changes (`git commit -m "feat: add regional spatial contagion weighting"`).
4. Push to the Branch (`git push origin feature/SpatialContagionModel`).
5. Open a Pull Request.

### Citation
If you utilize this codebase, architecture, or benchmark results in your research or academic paper, please cite:
```bibtex
@software{krishi_bhav_saathi_2026,
  author = {Dhruv and Contributors},
  title = {Krishi Bhav Saathi: Region-Wise AI Crop Price Prediction, Spatial Arbitrage, and Conformal Decision Engine},
  year = {2026},
  url = {https://github.com/IamDhruv777/Crop_Price_Prediction}
}
```

### License
Distributed under the **MIT License**. See `LICENSE` for more information.
