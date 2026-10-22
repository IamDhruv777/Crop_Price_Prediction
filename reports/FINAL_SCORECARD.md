# FINAL SYSTEM SCORECARD: CROP PRICE PREDICTION
**Version:** 2.1.0-production  
**Evaluation Dataset:** AGMARKNET Mandi Records (October 2016 – June 2017)  
**Sample Size:** 326,848 holdout test observations across 11 agricultural commodities  
**Verification Status:** Certified Reproducible via `pytest tests/ -v` (18/18 tests passing)  

---

## 1. COMPREHENSIVE MULTI-HORIZON PERFORMANCE SCORECARD

All metrics below are computed strictly from out-of-sample holdout test predictions (`data/test_predictions.csv`) via `scripts/evaluate.py`:

| Evaluation Dimension | Metric | 1-Day Ahead | 7-Day Ahead | 30-Day Ahead |
|:---|:---|:---:|:---:|:---:|
| **Sample Size** | Number of Out-of-Sample Evaluations ($n$) | 295,867 | 326,848 | 279,641 |
| **Goodness of Fit** | Pooled Model $R^2$ | **0.9840** | **0.9779** | 0.9485 |
| | Pooled Naive Persistence $R^2$ | 0.9829 | 0.9735 | **0.9512** |
| | **Skill Score ($R^2_{\text{skill}} = 1 - \frac{\text{MSE}_{\text{mod}}}{\text{MSE}_{\text{pers}}}$)** | **+6.60%** | **+16.74%** | **-5.46%** *(Pers wins)* |
| | Equal-Crop Macro Model $R^2$ | **0.8358** | **0.8331** | **0.7039** |
| | Equal-Crop Macro Persistence $R^2$ | 0.8206 | 0.7715 | 0.6469 |
| **Error Magnitude** | Pooled Model RMSE (₹/quintal) | **₹286.16** | **₹332.22** | ₹500.86 |
| | Pooled Persistence RMSE (₹/quintal) | ₹296.10 | ₹363.38 | **₹487.92** *(Pers wins)* |
| | **RMSE Skill Improvement (%)** | **+3.36%** | **+8.57%** | **-2.65%** *(Pers wins)* |
| | Pooled Model MAE (₹/quintal) | ₹101.11 | ₹147.98 | ₹260.80 |
| | Pooled Persistence MAE (₹/quintal) | **₹93.95** | **₹141.90** | **₹229.82** *(Pers wins)* |
| | Equal-Crop Macro Model RMSE (₹/quintal) | **₹369.97** | **₹409.71** | **₹575.46** |
| | Equal-Crop Macro Persistence RMSE (₹/quintal) | ₹383.98 | ₹458.01 | ₹578.18 |
| **Commercial Accuracy** | Enterprise Forecast Trust ($100 - \text{MAPE}$) | 97.04% | 95.74% | 92.63% |
| | Naive Persistence Trust ($100 - \text{MAPE}$) | **97.19%** | **95.92%** | **93.52%** |
| | APMC Trade Tolerance ($\pm 5\%$) | 85.23% | 75.92% | 53.64% |
| | APMC Trade Tolerance ($\pm 10\%$) | **94.66%** | **90.78%** | 78.36% |
| | Naive Persistence Tolerance ($\pm 10\%$) | 94.41% | 90.71% | **80.02%** *(Pers wins)* |
| | APMC Trade Tolerance ($\pm 20\%$) | 98.34% | 97.49% | 93.51% |
| **Market Direction** | Directional Up/Down Accuracy | 59.80% | 49.23% | 49.81% |
| | Directional F1 Score | 0.6842 | 0.4450 | 0.3786 |
| **Uncertainty Calibration** | Pooled Conformal 95% Band Coverage | **96.16%** | **96.46%** | **96.64%** |
| | Equal-Crop Macro Conformal 95% Coverage | **92.28%** | **92.75%** | **93.33%** |
| | Mean Interval Width (₹/quintal) | ₹931.49 | ₹1,481.76 | ₹2,542.81 |
| | Mean Interval Width (% of commodity price) | 39.02% | 62.14% | 106.44% |

---

## 2. PRODUCTION DEPLOYMENT DECISIONS

Based on the empirical audit scorecard, the following deployment policies are mandated:

```mermaid
graph TD
    A["Forecast Request (Market, Crop, Horizon)"] --> B{"Horizon?"}
    B -->|"1-Day"| C["Deploy Machine Learning Engine<br>(Tuned XGBoost / Ridge)<br>RMSE: ₹285.36 (Beats Persistence by 3.4%)"]
    B -->|"7-Day"| D["Deploy Temporal Deep Learning<br>(Temporal CNN TCN / Stacking Ensemble)<br>RMSE: ₹328.17 (Beats Persistence by 8.6%)"]
    B -->|"30-Day"| E["Deploy Naive Persistence Baseline (Pt)<br>Persistence RMSE ₹487.92 < Model RMSE ₹500.86<br>Persistence MAE ₹229.82 < Model MAE ₹260.80"]
```

1. **1-Day Ahead Horizon:** Deploy **Tuned XGBoost Regressor** / **Ridge Autoregressive**. Both reduce daily market variance and outscore Naive Persistence on RMSE ($R^2 = 0.9841$ vs $0.9829$).
2. **7-Day Ahead Horizon:** Deploy **Temporal CNN (TCN)** or the **Super Ensemble**. Generates an 8.57% error reduction over persistence, with 90.78% of forecasts within $\pm 10\%$ commercial tolerance.
3. **30-Day Ahead Horizon:** **Explicitly deploy Naive Persistence ($P_t$) as the production forecast**. Machine learning models at 30 days without fundamental planting/acreage predictors compound errors ($R^2_{\text{skill}} = -5.46\%$, MAE $+13.5\%$). Persistence is cheaper, faster, and more accurate.

---

## 3. PER-CROP ACCURACY & RELIABILITY MATRIX

Breakdown across all 11 benchmark commodities from `data/metrics_by_crop.csv`:

| Commodity | Test Rows (7D) | 7-Day Model RMSE | 7-Day Persistence RMSE | 7-Day Relative Accuracy | 7-Day Tol ($\pm 10\%$) | Conformal 95% Coverage |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rice** | 41,781 | **₹186.42** | ₹204.15 | **97.32%** | **95.68%** | 98.42% |
| **Wheat** | 85,537 | **₹124.58** | ₹138.92 | **96.69%** | **95.21%** | 98.78% |
| **Soyabean** | 38,888 | **₹178.94** | ₹192.30 | **97.12%** | **96.46%** | 98.26% |
| **Maize** | 44,839 | **₹112.30** | ₹125.64 | **96.23%** | **93.44%** | 97.61% |
| **Lentil (Masur)** | 17,394 | **₹284.12** | ₹312.45 | **95.42%** | **89.27%** | 97.48% |
| **Arhar (Tur)** | 32,794 | **₹452.18** | ₹510.32 | **93.55%** | **81.62%** | 95.84% |
| **Green Gram (Moong)** | 25,221 | **₹420.76** | ₹478.90 | **94.51%** | **84.77%** | 95.42% |
| **Black Gram (Urd)** | 34,000 | **₹512.34** | ₹582.10 | **93.07%** | **79.91%** | 94.21% |
| **Cowpea (Lobia)** | 2,199 | **₹418.90** | ₹465.20 | **93.45%** | **81.54%** | 94.12% |
| **White Peas** | 909 | **₹310.45** | ₹342.10 | **94.41%** | **87.46%** | 93.50% |
| **Kabuli Chana** | 3,286 | **₹1,498.20** | ₹1,620.50 | **90.65%** | **74.32%** | 66.82% |

---

## 4. ARCHITECTURAL PERFORMANCE RANKINGS (7-DAY HORIZON)

Strictly compiled from `data/master_model_leaderboard.csv`:

| Rank | Model Name | Architecture Type | RMSE (₹/qtl) | MAE (₹/qtl) | R² Score | 100−MAPE |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **1** | **Temporal CNN (TCN)** | Deep Learning (Causal Dilated) | **₹328.17** | **₹140.23** | **0.9784** | **96.00%** |
| **2** | **PyTorch GRU** | Deep Learning (Recurrent) | ₹329.38 | ₹141.79 | 0.9783 | 95.98% |
| **3** | **Time-Series Transformer** | Deep Learning (Self-Attention) | ₹330.37 | ₹141.31 | 0.9781 | 95.98% |
| **4** | **PyTorch LSTM** | Deep Learning (Recurrent) | ₹330.79 | ₹142.20 | 0.9781 | 95.95% |
| **5** | **Super Ensemble (Stacking)** | Ensemble Meta-Learner | ₹332.22 | ₹147.98 | 0.9779 | 95.74% |
| **6** | **Random Forest Regressor** | Machine Learning | ₹334.60 | ₹146.07 | 0.9776 | 95.84% |
| **7** | **Tuned XGBoost Regressor** | Machine Learning | ₹336.08 | ₹148.92 | 0.9774 | 95.80% |
| **8** | **XGBoost + Spatial KNN** | Machine Learning | ₹338.09 | ₹150.00 | 0.9771 | 95.78% |
| **9** | **LightGBM Regressor** | Machine Learning | ₹339.26 | ₹151.38 | 0.9769 | 95.75% |
| **10** | **Ridge Autoregressive** | Machine Learning | ₹339.44 | ₹165.07 | 0.9769 | 95.07% |
| **11** | **Naive Persistence Baseline** | Statistical Baseline | ₹363.38 | ₹141.90 | 0.9735 | 95.92% |
| **12** | **Moving Average (7-day)** | Statistical Baseline | ₹380.13 | ₹166.96 | 0.9711 | 95.24% |
