# MODEL CARD: CROP PRICE PREDICTION SYSTEM
**Model Identifier:** `crop-price-prediction-v2.1`  
**Model Date:** October 2026  
**Model Type:** Autoregressive Gradient Boosted Trees, Temporal Sequence Deep Learning, and Conformal Uncertainty Engine  
**License:** MIT  

---

## 1. MODEL OVERVIEW & INTENDED USE

### 1.1 Intended Use
- **Primary Use:** Providing smallholder farmers, FPOs (Farmer Producer Organizations), and agricultural procurement agencies with multi-horizon price guidance (1-Day, 7-Day, and 30-Day) across major APMC (Agricultural Produce Market Committee) mandis in India.
- **Decision Support:** Assisting farmers in determining optimal selling timing:
  - 🟢 **Hold (माल रोकें):** When forecast exceeds $+2.0\%$ above current modal price.
  - 🔴 **Sell (आज बेचें):** When forecast drops below $-2.0\%$ of current modal price.
  - 🔵 **Stable (स्थिर):** When forecast is within $[-2.0\%, +2.0\%]$.
- **Uncertainty Quantification:** Providing 95% Conformal Prediction Intervals ($[y_{\text{lower}}, y_{\text{upper}}]$) that guarantee that $\ge 95\%$ of true trading outcomes fall within the bounds under exchangeability.

### 1.2 Out-of-Scope & Prohibited Uses
- **High-Frequency Speculative Arbitrage:** Not designed for intra-day commodity trading.
- **Unverified Crop-Market Pairs:** Models must not be used on commodities or regions outside the verified training corpus without localized calibration.
- **Direct Input Imputation:** The API prohibits zero-filling missing features. If sufficient historical observations are absent, the model returns an explicit `insufficient_data` error.

---

## 2. TRAINING & EVALUATION DATA

### 2.1 Dataset Provenance
- **Source:** Real wholesale mandi trading records from AGMARKNET (Government of India Open Data Portal, `data.gov.in`).
- **Time Range:** January 2014 – June 2017.
- **Total Records:** 1,509,693 unique daily observations deduplicated from 1,740,759 raw mandi entries (intra-day varieties aggregated to 1 row per `market`, `commodity`, `date`).
- **Commodity Coverage (All 11 Core Crops):**
  1. Rice
  2. Wheat
  3. Maize
  4. Soyabean
  5. Arhar (Tur / Red Gram)
  6. Black Gram (Urd Beans)
  7. Green Gram (Moong)
  8. Lentil (Masur)
  9. Cowpea (Lobia / Karamani)
  10. White Peas
  11. Kabuli Chana (Chickpeas - White)

### 2.2 Leak-Free Partitioning & Embargoes
To prevent target leakage and data snooping, strict chronological partitioning with 30-day purge embargoes was enforced:
- **Train Set:** January 1, 2014 – November 30, 2015 ($n=658,037$)
- **Embargo Buffer 1:** December 1, 2015 – December 31, 2015 (31 days purged)
- **Validation Set:** January 1, 2016 – August 31, 2016 ($n=263,222$)
- **Embargo Buffer 2:** September 1, 2016 – September 30, 2016 (30 days purged)
- **Holdout Test Set:** October 1, 2016 – June 17, 2017 ($n=326,848$)

---

## 3. PERFORMANCE METRICS & BENCHMARK VALIDATION

All metrics are evaluated out-of-sample on the test partition ($n=326,848$):

| Horizon | Primary Architecture | Model R² | Naive Persistence R² | Model RMSE | Persistence RMSE | 100−MAPE | $\pm 10\%$ Tol | 95% Coverage |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day** | Tuned XGBoost / Ridge | **0.9841** | 0.9829 | **₹285.36** | ₹296.10 | 97.14% | 94.62% | 96.16% |
| **7-Day** | Temporal CNN (TCN) | **0.9784** | 0.9735 | **₹328.17** | ₹363.38 | 96.00% | 91.22% | 96.46% |
| **30-Day** | Naive Persistence Baseline ($P_t$) | 0.9485 | **0.9512** | ₹500.86 | **₹487.92** | 92.63% | **80.02%** | 96.64% |

### Macro Equal-Crop Evaluation (Unweighted by Crop Volume)
- **1-Day Macro R²:** 0.8358 (vs Persistence: 0.8206) | Macro Coverage: 92.28%
- **7-Day Macro R²:** 0.8331 (vs Persistence: 0.7715) | Macro Coverage: 92.75%
- **30-Day Macro R²:** 0.7039 (vs Persistence: 0.6469) | Macro Coverage: 93.33%

---

## 4. KNOWN LIMITATIONS & SCIENTIFIC DISCLOSURES

1. **Autoregressive Persistence at Long Horizons:**  
   At 30-Day horizons, machine learning models exhibit error compounding. **Naive Persistence ($P_t$) outperforms all ML models on RMSE, MAE, R², and Tolerance Band accuracy**. Consequently, Naive Persistence must be deployed for 30-day forecasts until fundamental crop sowing/acreage indicators are integrated.
2. **Meteorological Feature Caveat:**  
   The weather features (`temperature_mean`, `rainfall_mm`, `relative_humidity`, `wind_speed_ms`) in the pipeline were generated using thermodynamic climatology simulations with seasonal monsoon priors, rather than real-time live calls to the NASA POWER API.
3. **Thin-Market Uncertainty:**  
   Thinly traded commodities with high volatility (e.g., Kabuli Chana) have wider confidence bands. For Kabuli Chana, conformal coverage on pooled bounds is ~66%–68%, requiring crop-stratified (Mondrian) conformal intervals.
4. **Zero-Filling Prohibition:**  
   The model requires complete 48-feature vectors from the inference feature store. If an unseen market or missing trading history is requested, the system returns `HTTP 422 (insufficient_data)`.

---

## 5. ETHICAL & FAIR DEPLOYMENT CONSIDERATIONS

- **Price Manipulation Resistance:** Mandi prices are subject to local merchant cartelization in certain APMC yards. By incorporating cross-market spatial KNN features ($t-1$ neighbor prices from surrounding mandis within 200 km), the system reduces vulnerability to localized price suppression.
- **Fairness to Smallholder Farmers:** The system provides transparent, symmetric prediction intervals so farmers can evaluate both upside potential and downside downside risk before incurring transport and storage costs.
