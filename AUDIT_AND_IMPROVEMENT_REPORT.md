# FORENSIC AUDIT & VERIFIED PERFORMANCE IMPROVEMENT REPORT
**Repository:** `crop-price-prediction`  
**Git Branch:** `fix/audit-and-real-improvements`  
**Audit Status:** METHODICALLY REPAIRED & VERIFIED AGAINST PERSISTENCE  
**Holdout Test Partition:** 326,848 samples across all 11 commodities (October 2016 – June 2017)  
**Historical Backtest Data:** Features reflect historical trade observations (2014–2017), not live real-time feeds.

---

## 1. EXECUTIVE SUMMARY & FORENSIC AUDIT FINDINGS

A forensic audit of the `crop-price-prediction` repository revealed serious integrity and methodological defects that invalidated previously advertised metrics while obscuring genuine machine learning capabilities. All defects have been diagnosed, quarantined, and replaced with mathematically sound, leak-free implementations.

### 1.1 Defects Identified in Baseline Codebase
1. **Severe Dataset Truncation (2 crops evaluated vs. 11 advertised):**  
   `scripts/build_unified_features.py` sliced the 4.2M-row master with `df.iloc[-300000:]`. Because the master CSV was grouped by commodity, the evaluation set contained only **Green Gram** (36,943 rows) and **Cowpea** (2,930 rows). The other 9 commodities were completely excluded.
2. **Horizon Leakage and Row-Shift Collisions:**  
   Targets were generated via row shifts (`shift(-1)`, `shift(-7)`, `shift(-30)`) rather than true calendar dates, resulting in median calendar intervals of 1, 11, and 47 days. Intra-day varieties caused 2.02% of 1-Day targets to point to the exact same calendar day ($t=t$).
3. **Partition Overlap (Lack of Purge Embargo):**  
   No embargo buffer existed between chronological train, validation, and test cutoffs.
4. **Hardcoded Leaderboard Overwrites:**  
   `scripts/consolidate_master_leaderboard.py` contained hardcoded Python dictionaries and literal overwrites forcing R² to 0.9520, RMSE to 442.15.
5. **Inference Anchor Mismatch:**  
   Training trained models on anchors ($P_t$ for 1D, $0.6P_t + 0.4\text{MA}_7$ for 7D, $0.5P_t + 0.5\text{MA}_{30}$ for 30D), but inference added deltas solely to $P_t$, causing prediction shifts of up to ₹8,600+ on 7D and ₹10,500+ on 30D.
6. **Zero-Filling of Missing Features:**  
   Inference code filled missing lag features with zero, causing silent pipeline skew.

---

## 2. AUDIT REMEDIATION ACTIONS PERFORMED

| Action Item | Forensic Remedy Implemented | Verification Artifact |
|---|---|---|
| **Synthetic Generator Quarantined** | Moved `populate_all_crops_test_predictions.py` to `scripts/archive/` and revoked execution. | `scripts/archive/` |
| **Leaderboard Rewritten** | Rewrote `scripts/consolidate_master_leaderboard.py` to strictly read runtime metrics from `full_architecture_benchmark_metrics.csv` and `deep_learning_benchmark_metrics.csv`. Deleted all literal dictionaries, hardcoded training times, and fallback constants. | `scripts/consolidate_master_leaderboard.py` |
| **Single Source of Truth Evaluator** | Created immutable `scripts/evaluate.py` to calculate R², RMSE, MAE, MAPE, Relative Accuracy ($100-\text{MAPE}$), Tolerance Accuracies ($\pm 5\%, \pm 10\%, \pm 20\%$), Directional Accuracy/F1, Skill Scores vs. Persistence, and Mondrian 95% Conformal Coverage dynamically. | `scripts/evaluate.py` |
| **Dataset Reconstruction** | Rebuilt `scripts/build_unified_features.py`: Loaded 1.7M AGMARKNET records across ALL 11 CROPS (2014–2017). Deduplicated intra-day varieties to 1 row per `(market, commodity, date)`: 1,509,693 clean observations. | `data/processed/unified_splits/` |
| **Calendar-Ahead Horizons** | Implemented vectorized `pd.merge_asof` with strict future calendar constraints ($t+1\text{ to }t+3$ for 1-Day, $t+5\text{ to }t+9$ for 7-Day, $t+26\text{ to }t+34$ for 30-Day). Zero same-day targets verified across all 326,848 test rows. | `tests/test_audit_integrity.py` |
| **Strict 30-Day Purge Embargos** | Enforced 31-day embargo (Dec 2015) between Train and Val; enforced 30-day embargo (Sep 2016) between Val and Test. Zero cross-partition target leakage. | `tests/test_audit_integrity.py` |
| **Inference Anchor Parity** | Refactored `src/inference/predictor.py` to apply the exact training anchors: $P_t$ for 1-Day, $0.6P_t + 0.4\text{MA}_7$ for 7-Day, and flat persistence for 30-Day. | `src/inference/predictor.py` |
| **30-Day Persistence Deployment** | Implemented explicit deployment of Naive Persistence Baseline ($P_t$) for 30-day horizons in `src/inference/predictor.py`, reporting verified persistence benchmark metrics (RMSE ₹487.92, MAE ₹229.82). | `src/inference/predictor.py` |
| **Mondrian Conformal Calibration** | Created `scripts/calibrate_mondrian_conformal.py` calibrating crop-specific 95% quantiles on the validation partition. Reduced Wheat interval width by 79% while achieving $\ge 95\%$ macro coverage. | `scripts/calibrate_mondrian_conformal.py` |
| **Supported Feature Store** | Built `data/processed/inference_feature_store.csv` containing 5,218 pairs, explicitly tagging 4,554 complete pairs (87.3%) and rejecting 664 incomplete pairs with HTTP 422 (`insufficient_data`). | `data/processed/inference_feature_store.csv` |
| **Dynamic Report Generation** | Built `scripts/generate_audit_reports.py` formatting all report tables directly from evaluated artifacts with zero manually typed numbers. | `scripts/generate_audit_reports.py` |

---

## 3. HONEST MODEL VS. NAIVE PERSISTENCE EVALUATION

A central finding of this audit is that **commodity price forecasting must always be benchmarked against Naive Persistence ($P_t$)**. Because commodity prices exhibit high autocorrelation, naive persistence achieves very high $R^2$ scores simply by predicting yesterday's price. The true test of machine learning value is the **Skill Score** ($R^2_{\text{skill}} = 1 - \text{MSE}_{\text{model}} / \text{MSE}_{\text{pers}}$).

The table below reports runtime evaluation on the out-of-sample holdout test set ($n=326,848$ observations across all 11 crops):

### 3.1 Model vs. Naive Persistence (Pooled Overall)

| Horizon | Test Samples ($n$) | Metric | Model (Ensemble) | Naive Persistence ($P_t$) | Skill Score / Net Improvement | Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **1-Day Ahead** | 295,867 | **R² Score** | **0.9840** | 0.9829 | **+6.60%** | Model Wins on RMSE/R² |
| | | **RMSE (₹/qtl)** | **₹286.16** | ₹296.10 | **+3.36%** | Model Wins |
| | | **MAE (₹/qtl)** | ₹101.11 | **₹93.95** | **-7.62%** | Persistence Wins |
| | | **Relative Acc (100−MAPE)** | 97.04% | **97.19%** | **-0.15%** | Persistence Wins |
| | | **Tolerance ($\pm 10\%$)** | **94.66%** | 94.41% | **+0.25%** | Model Wins |
| | | **Macro Mondrian 95% Cov** | **96.07%** | — | Mean Width: 33.57% of price | Disclosed below 95% |
| **7-Day Ahead** | 326,848 | **R² Score** | **0.9779** | 0.9735 | **+16.42%** | Model Wins on RMSE/R² |
| | | **RMSE (₹/qtl)** | **₹332.22** | ₹363.38 | **+8.58%** | Model Wins |
| | | **MAE (₹/qtl)** | ₹147.98 | **₹141.90** | **-4.28%** | Persistence Wins |
| | | **Relative Acc (100−MAPE)** | 95.74% | **95.92%** | **-0.18%** | Persistence Wins |
| | | **Tolerance ($\pm 10\%$)** | **90.78%** | 90.71% | **+0.07%** | Tie |
| | | **Macro Mondrian 95% Cov** | **95.90%** | — | Mean Width: 43.13% of price | Disclosed below 95% |
| **30-Day Ahead** | 279,641 | **R² Score** | 0.9485 | **0.9512** | **-5.37%** | **PERSISTENCE WINS** |
| | | **RMSE (₹/qtl)** | ₹500.86 | **₹487.92** | **-2.65%** | **PERSISTENCE WINS** |
| | | **MAE (₹/qtl)** | ₹260.80 | **₹229.82** | **-13.48%** | **PERSISTENCE WINS** |
| | | **Relative Acc (100−MAPE)** | 92.63% | **93.52%** | **-0.89%** | **PERSISTENCE WINS** |
| | | **Tolerance ($\pm 10\%$)** | 78.36% | **80.02%** | **-1.66%** | **PERSISTENCE WINS** |
| | | **Macro Mondrian 95% Cov** | **96.02%** | — | Mean Width: 69.39% of price | Disclosed below 95% |

> [!IMPORTANT]
> **Scientific Finding on 30-Day Horizons:**  
> At 30-Day horizons, **Naive Persistence strictly outperforms the ML Stacking Ensemble across every pooled metric** (lower RMSE by ₹12.94, lower MAE by ₹30.98, higher R² by 0.0027). Consequently, the production inference API strictly deploys Naive Persistence ($P_t$) as the 30-day forecast.

---

### 3.2 Equal-Crop Macro Averages (Unweighted by Crop Volume)

| Horizon | Macro Model R² | Macro Pers R² | Macro Model RMSE | Macro Pers RMSE | Macro Model MAE | Macro Pers MAE | Macro Mondrian 95% Cov | Macro Mondrian Width |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day** | **0.8358** | 0.8206 | **₹369.97** | ₹383.98 | ₹168.56 | **₹159.24** | **96.07%** | 33.57% |
| **7-Day** | **0.8331** | 0.7715 | **₹409.71** | ₹458.01 | ₹232.80 | **₹232.45** | **95.90%** | 43.13% |
| **30-Day** | **0.7039** | 0.6469 | **₹577.20** | ₹600.98 | ₹388.42 | **₹371.24** | **96.02%** | 69.39% |

---

## 4. PER-CROP BREAKDOWN ACROSS ALL 11 CROPS

Generated dynamically from `data/metrics_by_crop.csv`:

| Commodity | Test Samples (1D / 7D / 30D) | 1D Model RMSE | 1D RelAcc | 7D Model RMSE | 7D RelAcc | 30D Pers RMSE | 30D RelAcc | Mondrian Cov (1D / 7D / 30D) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arhar (Tur-Red Gram)** | 29,232 / 32,794 / 27,512 | ₹362.13 | 95.98% | ₹480.50 | 93.55% | ₹724.94 | 90.15% | 96.9% / 96.8% / 98.4% |
| **Black Gram (Urd Beans)** | 30,423 / 34,000 / 29,239 | ₹507.63 | 95.18% | ₹597.62 | 93.07% | ₹876.01 | 89.34% | 98.6% / 98.9% / 99.1% |
| **Cowpea (Lobia-Karamani)** | 1,898 / 2,199 / 1,848 | ₹461.96 | 95.01% | ₹539.37 | 93.45% | ₹841.68 | 89.64% | 95.9% / 95.8% / 95.6% |
| **Green Gram (Moong)** | 22,359 / 25,221 / 21,321 | ₹337.97 | 95.91% | ₹408.99 | 94.51% | ₹586.57 | 91.83% | 96.4% / 96.8% / 97.5% |
| **Kabuli Chana (Chickpeas-White)** | 2,967 / 3,286 / 2,750 | ₹1160.36 | 92.15% | ₹1166.35 | 90.65% | ₹1706.14 | 85.49% | 92.0% / 91.0% / 91.7% |
| **Lentil (Masur)** | 15,829 / 17,394 / 14,677 | ₹284.74 | 97.10% | ₹337.83 | 95.42% | ₹568.44 | 91.41% | 96.4% / 96.0% / 96.4% |
| **Maize** | 39,694 / 44,839 / 38,609 | ₹161.55 | 97.18% | ₹153.00 | 96.23% | ₹191.63 | 95.05% | 95.6% / 95.5% / 96.3% |
| **Rice** | 39,027 / 41,781 / 37,001 | ₹190.28 | 98.27% | ₹208.06 | 97.32% | ₹292.73 | 96.35% | 95.6% / 95.6% / 95.8% |
| **Soyabean** | 34,796 / 38,888 / 34,071 | ₹135.47 | 97.69% | ₹149.17 | 97.12% | ₹199.89 | 96.30% | 96.8% / 96.7% / 98.0% |
| **Wheat** | 78,787 / 85,537 / 71,849 | ₹119.60 | 97.76% | ₹119.06 | 96.69% | ₹174.24 | 94.27% | 95.1% / 93.2% / 88.4% |
| **White Peas** | 855 / 909 / 764 | ₹348.00 | 94.96% | ₹346.87 | 94.41% | ₹448.54 | 91.79% | 97.4% / 98.6% / 99.1% |

### Conformal Coverage Disclosures (<95% Threshold)
While the equal-crop macro average coverage is $\ge 95\%$ across all horizons, specific commodities fall below 95%:
- **1-Day:** Kabuli Chana (91.98%)
- **7-Day:** Kabuli Chana (91.05%), Wheat (93.24%)
- **30-Day:** Kabuli Chana (91.67%), Wheat (88.41%)

All other 9 commodities achieve $\ge 95\%$ coverage.

---

## 5. ARCHITECTURAL LEADERBOARDS BY HORIZON

Generated strictly from `data/master_model_leaderboard.csv`:

### 5.1 1-Day Horizon Leaderboard
| Rank | Model Architecture | Family | RMSE (₹/qtl) | MAE (₹/qtl) | R² Score | Relative Accuracy | Tolerance (±10%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Ridge Autoregressive** | Machine Learning | ₹284.40 | ₹103.65 | 0.9842 | 96.89% | 94.88% |
| **2** | **XGBoost + Spatial KNN Features** | Machine Learning | ₹285.03 | ₹95.67 | 0.9842 | 97.15% | 94.65% |
| **3** | **Tuned XGBoost Regressor** | Machine Learning | ₹285.36 | ₹96.18 | 0.9841 | 97.14% | 94.62% |
| **4** | **LightGBM Regressor** | Machine Learning | ₹285.49 | ₹96.30 | 0.9841 | 97.15% | 94.65% |
| **5** | **Super Ensemble (Stacking)** | Ensemble Meta-Learner | ₹286.16 | ₹101.11 | 0.9840 | 97.04% | 94.66% |
| **6** | **Random Forest Regressor** | Machine Learning | ₹292.02 | ₹96.87 | 0.9834 | 97.12% | 94.34% |
| **7** | **Naive Persistence Baseline** | Statistical Baseline | ₹296.10 | ₹93.95 | 0.9829 | 97.19% | 94.41% |
| **8** | **Moving Average (7-day)** | Statistical Baseline | ₹321.34 | ₹130.03 | 0.9799 | 96.25% | 92.01% |

### 5.2 7-Day Horizon Leaderboard
| Rank | Model Architecture | Family | RMSE (₹/qtl) | MAE (₹/qtl) | R² Score | Relative Accuracy | Tolerance (±10%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Temporal CNN (TCN)** | Deep Learning | ₹328.17 | ₹140.23 | 0.9784 | 96.00% | 91.22% |
| **2** | **PyTorch GRU** | Deep Learning | ₹329.38 | ₹141.79 | 0.9783 | 95.98% | 91.14% |
| **3** | **Time-Series Transformer** | Deep Learning | ₹330.37 | ₹141.31 | 0.9781 | 95.98% | 90.99% |
| **4** | **PyTorch LSTM** | Deep Learning | ₹330.79 | ₹142.20 | 0.9781 | 95.95% | 91.05% |
| **5** | **Super Ensemble (Stacking)** | Ensemble Meta-Learner | ₹332.22 | ₹147.98 | 0.9779 | 95.74% | 90.78% |
| **6** | **Random Forest Regressor** | Machine Learning | ₹334.60 | ₹146.07 | 0.9776 | 95.84% | 90.48% |
| **7** | **Tuned XGBoost Regressor** | Machine Learning | ₹336.08 | ₹148.92 | 0.9774 | 95.80% | 90.38% |
| **8** | **XGBoost + Spatial KNN Features** | Machine Learning | ₹338.09 | ₹150.00 | 0.9771 | 95.78% | 90.37% |
| **9** | **LightGBM Regressor** | Machine Learning | ₹339.26 | ₹151.38 | 0.9769 | 95.75% | 90.19% |
| **10** | **Ridge Autoregressive** | Machine Learning | ₹339.44 | ₹165.07 | 0.9769 | 95.07% | 89.14% |
| **11** | **Naive Persistence Baseline** | Statistical Baseline | ₹363.38 | ₹141.90 | 0.9735 | 95.92% | 90.71% |
| **12** | **Moving Average (7-day)** | Statistical Baseline | ₹380.13 | ₹166.96 | 0.9711 | 95.24% | 88.20% |

### 5.3 30-Day Horizon Leaderboard
| Rank | Model Architecture | Family | RMSE (₹/qtl) | MAE (₹/qtl) | R² Score | Relative Accuracy | Tolerance (±10%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Naive Persistence Baseline** | Statistical Baseline | ₹487.92 | ₹229.82 | 0.9512 | 93.52% | 80.02% |
| **2** | **Super Ensemble (Stacking)** | Ensemble Meta-Learner | ₹500.86 | ₹260.80 | 0.9485 | 92.63% | 78.36% |
| **3** | **Ridge Autoregressive** | Machine Learning | ₹522.66 | ₹298.51 | 0.9440 | 91.20% | 69.98% |
| **4** | **LightGBM Regressor** | Machine Learning | ₹536.52 | ₹280.08 | 0.9410 | 92.50% | 76.47% |
| **5** | **Tuned XGBoost Regressor** | Machine Learning | ₹542.58 | ₹283.34 | 0.9396 | 92.42% | 76.05% |
| **6** | **Random Forest Regressor** | Machine Learning | ₹545.97 | ₹283.81 | 0.9389 | 92.36% | 75.94% |
| **7** | **XGBoost + Spatial KNN Features** | Machine Learning | ₹560.37 | ₹294.47 | 0.9356 | 92.16% | 74.58% |
| **8** | **Moving Average (30-day)** | Statistical Baseline | ₹646.45 | ₹323.52 | 0.9143 | 91.14% | 71.71% |

---

## 6. REPRODUCTION & VERIFICATION COMMANDS

```bash
# 1. Run all 18 automated forensic audit tests
python -m pytest tests/ -v

# 2. Calibrate Mondrian conformal quantiles strictly on validation set
python scripts/calibrate_mondrian_conformal.py

# 3. Build the 5,218-pair inference feature store (4,554 complete pairs)
python scripts/build_inference_feature_store.py

# 4. Evaluate out-of-sample predictions against Naive Persistence
python scripts/evaluate.py --predictions data/test_predictions.csv --output data/metrics.json

# 5. Consolidate master leaderboard
python scripts/consolidate_master_leaderboard.py

# 6. Regenerate reports from evaluated artifacts
python scripts/generate_audit_reports.py
```
