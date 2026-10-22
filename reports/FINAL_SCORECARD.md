# FINAL SYSTEM SCORECARD: CROP PRICE PREDICTION (DEPLOYED INFERENCE PIPELINE)
**Version:** 2.1.0-production  
**Evaluation Dataset:** AGMARKNET Mandi Records (October 2016 – June 2017)  
**Sample Support:** Complete-feature test rows evaluated strictly through production inference pipelines (290,890 1D, 320,330 7D, 275,109 30D)  
**Verification Status:** Verified via complete test suite (26 core regression tests + 12 audit tests passing)  
**Historical Backtest Data:** Features reflect historical trade observations (2014–2017), not live real-time feeds.

---

## 1. PRODUCTION DEPLOYED PIPELINE PERFORMANCE

All metrics below are computed strictly from out-of-sample holdout test predictions (`data/deployed_test_predictions.csv`) via `scripts/evaluate_deployed.py`:

| Evaluation Dimension | Metric | 1-Day Ahead (Tuned XGBoost, Clipped ±200) | 7-Day Ahead (Tuned XGBoost, Clipped ±400) | 30-Day Ahead (Naive Persistence) |
|:---|:---|:---:|:---:|:---:|
| **Sample Size** | Complete-Feature Test Samples ($n$) | 290,890 | 320,330 | 275,109 |
| **Goodness of Fit** | Pooled Model $R^2$ | **0.9844** | **0.9779** | **0.9516** |
| | Equal-Crop Macro Model $R^2$ | **0.8357** | **0.8334** | **0.6475** |
| **Error Magnitude** | Pooled Model RMSE (₹/quintal) | **₹283.38** | **₹332.69** | **₹486.53** |
| | Pooled Model MAE (₹/quintal) | **₹95.40** | **₹147.03** | **₹228.68** |
| | Equal-Crop Macro Model RMSE (₹/quintal) | **₹371.32** | **₹414.53** | **₹604.55** |
| | Equal-Crop Macro Model MAE (₹/quintal) | **₹162.54** | **₹237.33** | **₹374.00** |
| **Commercial Accuracy** | Enterprise Forecast Trust ($100 - \text{MAPE}$) | **97.16%** | **95.84%** | **93.54%** |
| | APMC Trade Tolerance ($\pm 5%$) | **85.69%** | **75.66%** | **59.75%** |
| | APMC Trade Tolerance ($\pm 10%$) | **94.66%** | **90.57%** | **80.12%** |
| | APMC Trade Tolerance ($\pm 20%$) | **98.32%** | **97.45%** | **94.49%** |
| **Market Direction** | Directional Up/Down Accuracy | 48.91% | 53.08% | 63.53% |
| | Directional F1 Score | 0.4783 | 0.5322 | 0.0000 |
| **Uncertainty Calibration** | Pooled Conformal 95% Band Coverage | **96.14%** | **95.60%** | **94.95%** |
| | Equal-Crop Macro Mondrian 95% Coverage | **96.18%** | **95.90%** | **96.03%** |
| | Mean Interval Width (% of commodity price) | 34.56% | 43.58% | 69.30% |

---

## 2. PRODUCTION DEPLOYMENT POLICY

Based on empirical audit verification and benchmark convergence, the following deployment policies are active in `src/inference/predictor.py` and registered in `models/trained/deployment_registry.json`:

1. **1-Day Ahead Horizon:** Serves **Tuned XGBoost Regressor** (`anchor = modal_price`, delta clipped to $[-200, 200]$). Out-of-sample RMSE: **₹283.38**, MAE: **₹95.40**, $R^2$: **0.9844**.
2. **7-Day Ahead Horizon:** Serves **Tuned XGBoost Regressor** (`anchor = 0.6*modal_price + 0.4*price_rolling_mean_7`, delta clipped to $[-400, 400]$). Out-of-sample RMSE: **₹332.69**, MAE: **₹147.03**, $R^2$: **0.9779**.
3. **30-Day Ahead Horizon:** Strictly serves **Naive Persistence Baseline ($P_t$)**. Machine learning models at 30 days underperform persistence across all metrics on the test partition. Flat persistence is served in the API: RMSE **₹486.53**, MAE **₹228.68**, $R^2$: **0.9516**.

---

## 3. PER-CROP ACCURACY & RELIABILITY MATRIX (7-DAY AHEAD)

### 3.1 Production Deployed Complete-Feature Evaluation
Generated strictly from `data/metrics_deployed_by_crop.csv`:

| Commodity | Test Samples ($n$) | Deployed RMSE | MAE | Rel. Acc (100−MAPE) | Tol (±10%) | Mondrian 95% Cov | Mean Interval Width | Width (% Price) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arhar (Tur-Red Gram)** | 30,996 | ₹491.22 | ₹300.84 | 93.26% | 80.55% | **96.69%** | ₹2360.92 | 52.9% |
| **Black Gram (Urd Beans)** | 32,905 | ₹597.31 | ₹372.06 | 92.97% | 79.65% | **98.93%** | ₹4070.68 | 72.6% |
| **Cowpea (Lobia-Karamani)** | 2,128 | ₹547.49 | ₹315.16 | 93.29% | 81.30% | **95.77%** | ₹2650.24 | 59.8% |
| **Green Gram (Moong)** | 24,765 | ₹413.58 | ₹253.50 | 94.34% | 84.20% | **96.90%** | ₹1967.40 | 42.4% |
| **Kabuli Chana (Chickpeas-White)** | 3,157 | ₹1186.15 | ₹745.29 | 90.48% | 74.09% | **91.10%** | ₹3765.02 | 42.0% |
| **Lentil (Masur)** | 17,121 | ₹346.68 | ₹222.17 | 95.16% | 88.17% | **95.83%** | ₹1361.22 | 28.6% |
| **Maize** | 43,820 | ₹150.59 | ₹48.35 | 96.63% | 93.42% | **95.25%** | ₹325.62 | 23.7% |
| **Rice** | 41,454 | ₹204.77 | ₹61.14 | 97.63% | 95.65% | **95.59%** | ₹541.66 | 21.7% |
| **Soyabean** | 38,308 | ₹147.75 | ₹76.88 | 97.16% | 96.41% | **96.70%** | ₹561.64 | 20.5% |
| **Wheat** | 84,808 | ₹118.13 | ₹55.30 | 96.88% | 95.10% | **93.29%** | ₹310.20 | 18.0% |
| **White Peas** | 868 | ₹356.14 | ₹159.98 | 94.15% | 86.75% | **98.85%** | ₹2919.26 | 97.2% |

### 3.2 Full Holdout Benchmark Reference Matrix
Generated strictly from `data/metrics_by_crop.csv`:

| Commodity | Benchmark Samples ($n$) | Benchmark RMSE | Benchmark Persistence RMSE | Rel. Acc (100−MAPE) | Tol (±10%) | Pooled Cov 95% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arhar (Tur-Red Gram)** | 32,794 | ₹480.50 | ₹488.05 | 93.55% | 81.62% | 91.28% |
| **Black Gram (Urd Beans)** | 34,000 | ₹597.62 | ₹644.82 | 93.07% | 79.91% | 88.01% |
| **Cowpea (Lobia-Karamani)** | 2,199 | ₹539.37 | ₹615.09 | 93.45% | 81.54% | 88.77% |
| **Green Gram (Moong)** | 25,221 | ₹408.99 | ₹448.48 | 94.51% | 84.77% | 93.94% |
| **Kabuli Chana (Chickpeas-White)** | 3,286 | ₹1166.35 | ₹1327.12 | 90.65% | 74.32% | 68.32% |
| **Lentil (Masur)** | 17,394 | ₹337.83 | ₹358.69 | 95.42% | 89.27% | 96.67% |
| **Maize** | 44,839 | ₹153.00 | ₹186.51 | 96.23% | 93.44% | 99.65% |
| **Rice** | 41,781 | ₹208.06 | ₹261.51 | 97.32% | 95.68% | 98.62% |
| **Soyabean** | 38,888 | ₹149.17 | ₹168.44 | 97.12% | 96.46% | 99.48% |
| **Wheat** | 85,537 | ₹119.06 | ₹137.31 | 96.69% | 95.21% | 99.71% |
| **White Peas** | 909 | ₹346.87 | ₹402.10 | 94.41% | 87.46% | 95.82% |

### Conformal Coverage Disclosures (<95% Threshold)
Mondrian calibration on validation complete support achieves macro average coverage $\ge 95%$ across all horizons (96.18% 1D, 95.90% 7D, 96.03% 30D). Specific commodities with high volatility fall below 95%:
- **1-Day:** Kabuli Chana (Chickpeas-White) (92.60%), Wheat (94.94%)
- **7-Day:** Kabuli Chana (Chickpeas-White) (91.10%), Wheat (93.29%)
- **30-Day:** Kabuli Chana (Chickpeas-White) (91.72%), Wheat (88.36%)

All other commodities achieve $\ge 95%$ empirical coverage.

---

## 4. ARCHITECTURAL PERFORMANCE RANKINGS (7-DAY HORIZON)

Generated strictly from `data/master_model_leaderboard.csv`:

| Rank | Model Name | Architecture Type | RMSE (₹/qtl) | MAE (₹/qtl) | R² Score | 100−MAPE | Tol (±10%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Temporal CNN (TCN)** | Deep Learning | ₹328.17 | ₹140.23 | 0.9784 | 96.00% | 91.22% |
| **2** | **PyTorch GRU** | Deep Learning | ₹329.38 | ₹141.79 | 0.9783 | 95.98% | 91.14% |
| **3** | **Time-Series Transformer** | Deep Learning | ₹330.37 | ₹141.31 | 0.9781 | 95.98% | 90.99% |
| **4** | **PyTorch LSTM** | Deep Learning | ₹330.79 | ₹142.20 | 0.9781 | 95.95% | 91.05% |
| **5** | **Super Ensemble (Stacking)** | Ensemble Meta-Learner | ₹332.22 | ₹147.98 | 0.9779 | 95.74% | 90.78% |
| **6** | **Tuned XGBoost Regressor** | Machine Learning | ₹332.69 | ₹147.03 | 0.9779 | 95.84% | 90.57% |
| **7** | **Random Forest Regressor** | Machine Learning | ₹334.60 | ₹146.07 | 0.9776 | 95.84% | 90.48% |
| **8** | **XGBoost + Spatial KNN Features** | Machine Learning | ₹338.09 | ₹150.00 | 0.9771 | 95.78% | 90.37% |
| **9** | **LightGBM Regressor** | Machine Learning | ₹339.26 | ₹151.38 | 0.9769 | 95.75% | 90.19% |
| **10** | **Ridge Autoregressive** | Machine Learning | ₹339.44 | ₹165.07 | 0.9769 | 95.07% | 89.14% |
| **11** | **Naive Persistence Baseline** | Statistical Baseline | ₹363.38 | ₹141.90 | 0.9735 | 95.92% | 90.71% |
| **12** | **Moving Average (7-day)** | Statistical Baseline | ₹380.13 | ₹166.96 | 0.9711 | 95.24% | 88.20% |

---

## 5. FEATURE STORE COVERAGE & DATA CONTRACT

- **Total Unique Market-Crop Pairs:** 5,218
- **Fully Supported Complete Pairs (100% 48 features non-null):** 4,554 (87.3%)
- **Unsupported Incomplete Pairs (<30d trading history):** 664 (12.7%) -> strictly returns HTTP 422 (`insufficient_data`)
- **Zero-Fill Prohibition:** Production inference strictly enforces a zero-fill prohibition, returning an explicit error for incomplete rows and maintaining complete-feature parity between offline benchmarks and online serving.

---

## 6. CRYPTOGRAPHIC ARTIFACT INTEGRITY MANIFEST

Generated dynamically from `data/report_manifest.json`:

| Artifact File | Relative Path | Size (Bytes) | SHA-256 Checksum |
|:---|:---|:---:|:---|
| `metrics_deployed.json` | `data/metrics_deployed.json` | 15,955 | `00098456ad8f55f24db21389cd20f3807e342fc0c3f62694bfc1090b983a7956` |
| `metrics_deployed_by_crop.csv` | `data/metrics_deployed_by_crop.csv` | 2,927 | `f43213065df417a9baa184900602945c5246c8df159a316811579bfa36201c94` |
| `deployed_test_predictions.csv` | `data/deployed_test_predictions.csv` | 63,552,161 | `225f28194e0aebecd7f08ecb62db8b9267fa4421a856ffbb1ad165dfb9f7ebc3` |
| `mondrian_conformal_quantiles.json` | `data/mondrian_conformal_quantiles.json` | 3,039 | `8cbd372153342fb440bab700c37298f85145c7252c012f8359a012039a92be6b` |
| `master_model_leaderboard.csv` | `data/master_model_leaderboard.csv` | 3,732 | `f06f2697d4ca6d93c042d9233fc2715f14658c3697f57cd682d54f6af12d8156` |
| `deployment_registry.json` | `models/trained/deployment_registry.json` | 1,802 | `11b4ef33212888a01a9bfab761a6548ce336438b70c9f327a1b1106547966c3c` |
