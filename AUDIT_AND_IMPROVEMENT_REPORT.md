# FORENSIC AUDIT & VERIFIED PERFORMANCE IMPROVEMENT REPORT
**Repository:** `crop-price-prediction`  
**Git Branch:** `fix/audit-and-real-improvements`  
**Audit Status:** DEPLOYED INFERENCE PIPELINE VERIFIED AND REPRODUCIBLE  
**Holdout Test Partition:** Evaluated on complete-feature test samples (290,890 1D, 320,330 7D, 275,109 30D)  
**Historical Backtest Data:** Features reflect historical trade observations (2014–2017), not live real-time feeds.

---

## 1. EXECUTIVE SUMMARY & FORENSIC AUDIT FINDINGS

A forensic audit of the `crop-price-prediction` repository revealed serious integrity and methodological defects that invalidated previously advertised metrics while obscuring genuine machine learning capabilities. All defects have been diagnosed, quarantined, and replaced with mathematically sound, leak-free implementations matching production serving.

### 1.1 Deployed Architecture Summary
1. **1-Day Ahead:** Tuned XGBoost Regressor with residual clipping ($\pm 200$), anchored on modal price. Out-of-sample RMSE: **₹283.38**, MAE: **₹95.40**, $R^2$: **0.9844**, 95% Coverage: **96.14%**.
2. **7-Day Ahead:** Tuned XGBoost Regressor with residual clipping ($\pm 400$), anchored on $0.6P_t + 0.4\text{MA}_7$. Out-of-sample RMSE: **₹332.69**, MAE: **₹147.03**, $R^2$: **0.9779**, 95% Coverage: **95.60%**.
3. **30-Day Ahead:** Naive Persistence Baseline ($P_t$). Machine learning models at 30 days underperform persistence across all metrics. Out-of-sample RMSE: **₹486.53**, MAE: **₹228.68**, $R^2$: **0.9516**, 95% Coverage: **94.95%**.

---

## 2. PRODUCTION SERVING PARITY TABLE

| Horizon | Deployed Model | Anchor Equation | Delta Clip | Test RMSE | Test MAE | Test R² | Rel. Acc (100−MAPE) | Tol (±10%) | Mondrian 95% Cov |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day Ahead** | Tuned XGBoost | $P_t$ | $[-200, 200]$ | **₹283.38** | **₹95.40** | **0.9844** | 97.16% | 94.66% | **96.18%** |
| **7-Day Ahead** | Tuned XGBoost | $0.6P_t + 0.4\text{MA}_7$ | $[-400, 400]$ | **₹332.69** | **₹147.03** | **0.9779** | 95.84% | 90.57% | **95.90%** |
| **30-Day Ahead** | Naive Persistence | $P_t$ | None | **₹486.53** | **₹228.68** | **0.9516** | 93.54% | 80.12% | **96.03%** |

---

## 3. PER-CROP BREAKDOWN ACROSS ALL 11 CROPS

Generated dynamically from `data/metrics_deployed_by_crop.csv`:

| Commodity | Test Samples (1D / 7D / 30D) | 1D Deployed RMSE | 1D RelAcc | 7D Deployed RMSE | 7D RelAcc | 30D Pers RMSE | 30D RelAcc | Mondrian Cov (1D / 7D / 30D) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arhar (Tur-Red Gram)** | 27,825 / 30,996 / 26,175 | ₹362.32 | 96.21% | ₹491.22 | 93.26% | ₹733.78 | 90.03% | 96.8% / 96.7% / 98.5% |
| **Black Gram (Urd Beans)** | 29,670 / 32,905 / 28,510 | ₹496.08 | 95.46% | ₹597.31 | 92.97% | ₹873.29 | 89.38% | 98.7% / 98.9% / 99.0% |
| **Cowpea (Lobia-Karamani)** | 1,848 / 2,128 / 1,805 | ₹464.68 | 95.21% | ₹547.49 | 93.29% | ₹847.94 | 89.61% | 95.9% / 95.8% / 95.5% |
| **Green Gram (Moong)** | 22,037 / 24,765 / 21,036 | ₹339.67 | 96.01% | ₹413.58 | 94.34% | ₹587.35 | 91.81% | 96.8% / 96.9% / 97.5% |
| **Kabuli Chana (Chickpeas-White)** | 2,851 / 3,157 / 2,646 | ₹1178.72 | 92.17% | ₹1186.15 | 90.48% | ₹1728.37 | 85.42% | 92.6% / 91.1% / 91.7% |
| **Lentil (Masur)** | 15,601 / 17,121 / 14,442 | ₹281.55 | 97.32% | ₹346.68 | 95.16% | ₹570.80 | 91.38% | 96.1% / 95.8% / 96.4% |
| **Maize** | 38,891 / 43,820 / 37,846 | ₹158.83 | 97.31% | ₹150.59 | 96.63% | ₹188.14 | 95.13% | 95.5% / 95.2% / 96.4% |
| **Rice** | 38,788 / 41,454 / 36,747 | ₹190.38 | 98.35% | ₹204.77 | 97.63% | ₹291.73 | 96.36% | 95.9% / 95.6% / 95.9% |
| **Soyabean** | 34,392 / 38,308 / 33,628 | ₹134.49 | 97.76% | ₹147.75 | 97.16% | ₹198.47 | 96.32% | 96.9% / 96.7% / 98.0% |
| **Wheat** | 78,165 / 84,808 / 71,535 | ₹121.45 | 97.76% | ₹118.13 | 96.88% | ₹174.39 | 94.27% | 94.9% / 93.3% / 88.4% |
| **White Peas** | 822 / 868 / 739 | ₹356.40 | 94.80% | ₹356.14 | 94.15% | ₹455.84 | 91.57% | 97.7% / 98.8% / 99.0% |

---

## 4. REPRODUCTION & VERIFICATION COMMANDS

```bash
# 1. Run full regression test suite (26 core regression + 12 deployment audit tests)
python -m pytest tests/ -v

# 2. Calibrate Mondrian conformal quantiles strictly on validation complete support
python scripts/calibrate_mondrian_conformal.py

# 3. Evaluate deployed inference pipeline on test set complete support
python scripts/evaluate_deployed.py

# 4. Regenerate reports and cryptographic manifest
python scripts/generate_audit_reports.py
```
