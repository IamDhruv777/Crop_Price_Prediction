# Crop Price Prediction using Advanced AI/ML Techniques

This project implements a comprehensive crop price prediction system using advanced machine learning, deep learning, and spatial-temporal modeling techniques. It forecasts agricultural commodity prices by leveraging historical price data, market arrivals, weather patterns, and geospatial information.

## Architecture & Phases

The project is structured into modular phases ensuring robust data handling and model evaluation:

1. **Phase 1: Data Architecture & ETL Framework**
   - Ingestion of historical price data, weather, and geospatial features.
   - Storage using PostgreSQL with PostGIS for spatial queries.
   - Rigorous data cleaning, imputation, and feature engineering.

2. **Phase 2: Baseline Models & Machine Learning**
   - Implementation of statistical baselines (ARIMA/SARIMA) and traditional ML (XGBoost, Random Forests).
   - Establishing baseline metrics for future comparisons.

3. **Phase 3: Advanced Spatial-Temporal Deep Learning**
   - Utilization of PyTorch, PyTorch Geometric, and DGL for Graph Neural Networks (GNNs).
   - Integration of Long Short-Term Memory (LSTM) and Transformers for temporal dynamics.

4. **Phase 4: Comprehensive Evaluation (Ablation Study)**
   - Strict adherence to chronological walk-forward validation.
   - No data leakage between training, validation, and test sets.
   - Extensive ablation studies to understand the impact of various features (e.g., weather vs. no weather, spatial vs. non-spatial).

5. **Phase 5: API & Dashboard Deployment**
   - **FastAPI** backend for model inference and data retrieval.
   - **Streamlit** dashboard for interactive visualization of forecasts and market trends.

## Setup Instructions

### Using Docker (Recommended)

1. Ensure Docker and Docker Compose are installed.
2. Create a `.env` file in the root directory (optional, defaults provided):
   ```env
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=crop_prices
   ```
3. Build and start the services:
   ```bash
   docker-compose up --build -d
   ```
4. Access the applications:
   - **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
   - **Database**: `localhost:5432`

### Local Development

1. Install system dependencies (e.g., GDAL for PostGIS/GeoPandas).
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Setup PostgreSQL database with PostGIS extension and configure `.env`.
4. Run the API:
   ```bash
   uvicorn api.main:app --reload
   ```
5. Run the Dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## Running the Pipeline

Execute the end-to-end pipeline (ETL, training, evaluation) via the main script:

```bash
python scripts/run_pipeline.py
```

## Modeling Methodology & Evaluation

- **Strict Chronological Validation**: Models are evaluated using walk-forward validation (e.g., train on 2010-2018, validate on 2019, test on 2020) reflecting real-world forecasting scenarios.
- **Ablation Studies**: We systematically remove model components (like spatial embeddings or specific weather features) to quantify their contribution to predictive accuracy.
- **Explainability**: SHAP values are integrated to interpret feature importance across our machine learning models.
