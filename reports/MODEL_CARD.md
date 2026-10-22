# MODEL CARD: CROP PRICE PREDICTION SYSTEM
**Model Identifier:** `crop-price-prediction-v2.1`  
**Model Date:** October 2026  
**Model Type:** Residual Gradient Boosted Trees with Autoregressive Anchors and Mondrian Conformal Uncertainty  
**License:** MIT  

---

## 1. MODEL OVERVIEW & INTENDED USE

### 1.1 Intended Use
- **Primary Use:** Providing smallholder farmers, FPOs (Farmer Producer Organizations), and agricultural procurement agencies with multi-horizon price guidance (1-Day, 7-Day, and 30-Day) across major APMC (Agricultural Produce Market Committee) mandis in India.
- **Decision Support:** Assisting farmers in determining optimal selling timing:
  - 🟢 **Hold:** When forecast exceeds $+2.0%$ above current modal price.
  - 🔴 **Sell:** When forecast drops below $-2.0%$ of current modal price.
  - 🔵 **Stable:** When forecast is within $[-2.0%, +2.0%]$.
- **Uncertainty Quantification:** Providing 95% Conformal Prediction Intervals ($[y_{\text{lower}}, y_{\text{upper}}]$) targeting empirical $\ge 95%$ coverage under exchangeability.

### 1.2 Out-of-Scope & Prohibited Uses
- **High-Frequency Speculative Arbitrage:** Not designed for intra-day commodity trading.
- **Unverified Crop-Market Pairs:** Models must not be used on commodities or regions outside the verified training corpus without localized calibration.
- **Direct Input Imputation:** The API prohibits zero-filling missing features. If sufficient historical observations are absent, the model returns an explicit HTTP 422 (`insufficient_data`) error.

---

## 2. PRODUCTION SERVING SPECIFICATION & DEPLOYED METRICS

All metrics evaluated out-of-sample on complete-feature test observations via `scripts/evaluate_deployed.py`:

| Horizon | Deployed Model | Anchor | Clip Bounds | Test Samples ($n$) | RMSE (₹/qtl) | MAE (₹/qtl) | R² Score | Rel. Acc (100−MAPE) | Mondrian 95% Coverage |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day** | Tuned XGBoost | $P_t$ | $[-200, 200]$ | 290,890 | **₹283.38** | **₹95.40** | **0.9844** | 97.16% | **96.18%** |
| **7-Day** | Tuned XGBoost | $0.6P_t + 0.4\text{MA}_7$ | $[-400, 400]$ | 320,330 | **₹332.69** | **₹147.03** | **0.9779** | 95.84% | **95.90%** |
| **30-Day** | Naive Persistence | $P_t$ | None | 275,109 | **₹486.53** | **₹228.68** | **0.9516** | 93.54% | **96.03%** |

---

## 3. EQUAL-CROP MACRO PERFORMANCE

- **1-Day Macro R²:** 0.8357 | Macro RMSE: ₹371.32 | Macro Coverage: 96.18%
- **7-Day Macro R²:** 0.8334 | Macro RMSE: ₹414.53 | Macro Coverage: 95.90%
- **30-Day Macro R²:** 0.6475 | Macro RMSE: ₹604.55 | Macro Coverage: 96.03%
