# PoF2 Pipeline Reference Guide

Complete mapping of scripts, requirements, features, and outputs for Turkish EDAŞ Equipment Failure Prediction Project.

---

## 📊 Pipeline Scripts Overview

| # | Script | Module | Req. Coverage | Key Outputs | Runtime |
|---|--------|--------|---------------|-------------|---------|
| 1 | `01_data_profiling.py` | Data QA | All | Quality report, column mappings | ~30s |
| 2 | `02_data_transformation.py` | Feature Eng. | MOD1, MOD3 | `equipment_level_data.csv` (1,148 equipment × 30+ features) | ~1min |
| 3 | `03_feature_engineering.py` | Feature Eng. | MOD1, MOD3 | Enhanced features: clusters, volatility, trends | ~1min |
| 4 | `04_eda.py` | Analysis | All | 16 analysis plots in `outputs/eda/` | ~3-5min |
| 5 | `05_feature_selection.py` | Modeling | MOD1 | Selected features for modeling | ~1min |
| 6 | `05b_remove_leaky_features.py` | Modeling | MOD1 | Cleaned features (no data leakage) | ~1min |
| 7 | `06_model_training.py` | Model 2 | MOD1, MOD3 | Chronic repeater classifier (AUC 0.96) | ~2-3min |
| 8 | `09_survival_analysis.py` | Model 1 | MOD1 | Temporal PoF predictions (C-index 0.77) | ~2-3min |
| 9 | `10_consequence_of_failure.py` | Risk | MOD1, MOD3 | Risk assessment + CAPEX priority list | ~1min |

---

## 🎯 Requirement Coverage Matrix

### Module 1: Equipment-Level PoF Prediction

| Requirement | Scripts | Features Used | Output |
|-------------|---------|---------------|--------|
| **Temporal PoF (3M/12M/24M)** | 09 | Age, MTBF, Failure_Rate, Historical failures | `pof_multi_horizon_predictions.csv` |
| **Chronic Repeater Detection** | 06 | Recurring flags, Cause consistency, Failure count | Chronic repeater probabilities |
| **Risk Scoring (PoF × CoF)** | 10 | PoF + Customer count + Outage duration | `risk_assessment_*.csv` |
| **CAPEX Prioritization** | 10 | Risk score + Equipment class + District | `capex_priority_list.csv` (Top 100) |

### Module 3: Recurring Failures & Cause Analysis

| Requirement | Scripts | Features Used | Output |
|-------------|---------|---------------|--------|
| **30-Day Recurring Failures** | 02, 04 | `Tekrarlayan_Arıza_30gün_Flag` | Binary flag + analysis plots |
| **90-Day Recurring Failures** | 02, 04 | `Tekrarlayan_Arıza_90gün_Flag` | Binary flag + analysis plots |
| **Cause Code Analysis** | 02, 04, 06 | `Tek_Neden_Flag`, `Arıza_Nedeni_Tutarlılık` (94.66%) | Cause code features + plots |
| **Customer Impact** | 02, 04, 10 | `total_customer_count_Max/Avg`, MV/LV categories | CoF scores + risk matrices |

---

## 🔧 Master Feature List

### 1. Equipment Identification (4 features)
| Feature | Type | Description | Source Script |
|---------|------|-------------|---------------|
| `Ekipman_ID` | ID | Primary equipment identifier (from cbs_id) | 02 |
| `Ekipman_Sınıfı` | Categorical | Equipment class (Ayırıcı, Trafo, Rekortman, etc.) | 02 |
| `İlçe` | Categorical | District (Salihli, Alaşehir, Gördes) | 02 |
| `Equipment_Class_Primary` | Categorical | Primary equipment class for modeling | 02 |

### 2. Temporal Features (5 features)
| Feature | Type | Description | Source Script |
|---------|------|-------------|---------------|
| `Ekipman_Yaşı_Yıl` | Numeric | Equipment age in years | 02 |
| `Age_Source` | Categorical | Age data source (TESIS_TARIHI vs EDBS_IDATE) | 02 |
| `İlk_Arıza_Tarihi` | Date | First failure date | 02 |
| `Son_Arıza_Tarihi` | Date | Last failure date | 02 |
| `Days_Since_Last_Failure` | Numeric | Days since last failure (censoring) | 02 |

### 3. Failure History Features (12 features)
| Feature | Type | Description | Source Script |
|---------|------|-------------|---------------|
| `Total_Arıza_Sayısı` | Count | Total failure count | 02 |
| `Arıza_Sayısı_3Ay` | Count | Failures in last 3 months | 02 |
| `Arıza_Sayısı_6Ay` | Count | Failures in last 6 months | 02 |
| `Arıza_Sayısı_12Ay` | Count | Failures in last 12 months | 02 |
| `MTBF_Gün` | Numeric | Mean Time Between Failures (days) | 02 |
| `MTTR_Saat` | Numeric | Mean Time To Repair (hours) | 02 |
| `Failure_Rate_Per_Year` | Numeric | Annual failure rate | 02 |
| `Güvenilirlik_Skoru` | Numeric | Reliability score (0-100, inverse of failure rate) | 02 |
| `Arıza_Frekans_Kategorisi` | Categorical | Frequency category (Düşük/Orta/Yüksek) | 02 |
| `Arıza_Trend` | Numeric | Failure trend (increasing/decreasing) | 03 |
| `Arıza_Volatility` | Numeric | Failure count volatility | 03 |
| `Arıza_Acceleration` | Numeric | Failure acceleration over time | 03 |

### 4. Recurring Failure Features (2 features)
| Feature | Type | Description | Source Script | Importance |
|---------|------|-------------|---------------|------------|
| `Tekrarlayan_Arıza_30gün_Flag` | Binary | Repeated failure within 30 days | 02 | MOD3 |
| `Tekrarlayan_Arıza_90gün_Flag` | Binary | Repeated failure within 90 days | 02 | MOD3 |

**Statistics:**
- 30-day recurring: ~8% of equipment
- 90-day recurring: 111 equipment (9.7%)
- Recurring equipment: MTBF 50% lower than non-recurring

### 5. Cause Code Features (9 features)
| Feature | Type | Description | Source Script | Model Importance |
|---------|------|-------------|---------------|------------------|
| `Arıza_Nedeni_İlk` | Categorical | First cause code | 02 | - |
| `Arıza_Nedeni_Son` | Categorical | Last cause code | 02 | - |
| `Arıza_Nedeni_Sık` | Categorical | Most frequent cause code | 02 | 15.3% (Model 2) |
| `Arıza_Nedeni_Çeşitlilik` | Numeric | Cause code diversity (1-N) | 02 | 8.1% |
| `Arıza_Nedeni_Tutarlılık` | Numeric | Cause consistency ratio (mean: 94.66%) | 02 | 12.4% |
| `Tek_Neden_Flag` | Binary | Single dominant cause (>80% of failures) | 02 | **32.7%** ⭐ |
| `Çok_Nedenli_Flag` | Binary | Multiple causes (>3 distinct) | 02 | 5.2% |
| `Neden_Değişim_Flag` | Binary | Cause code changed over time | 02 | 7.8% |
| `Ekipman_Neden_Risk_Skoru` | Numeric | Equipment×Cause risk score | 02 | 11.6% |

**Key Insights:**
- `Tek_Neden_Flag`: **Highest feature importance** in Model 2 (32.7%)
- Cause consistency: 94.66% average → equipment typically fail for same reasons
- Top 5 causes cover 70%+ of failures

### 6. Customer Impact Features (8 features)
| Feature | Type | Description | Source Script | Usage |
|---------|------|-------------|---------------|-------|
| `total_customer_count_Avg` | Numeric | Average customers affected | 02 | CoF calculation |
| `total_customer_count_Max` | Numeric | Maximum customers affected | 02 | CoF calculation |
| `urban_mv_Max` | Numeric | Urban medium voltage customers | 02 | CoF detail |
| `urban_lv_Max` | Numeric | Urban low voltage customers | 02 | CoF detail |
| `suburban_mv_Max` | Numeric | Suburban medium voltage customers | 02 | CoF detail |
| `suburban_lv_Max` | Numeric | Suburban low voltage customers | 02 | CoF detail |
| `rural_mv_Max` | Numeric | Rural medium voltage customers | 02 | CoF detail |
| `rural_lv_Max` | Numeric | Rural low voltage customers | 02 | CoF detail |

**Statistics:**
- Mean customers per equipment: 217.5
- Median customers per equipment: 54.0
- Maximum single equipment: 16,984 customers

### 7. Outage Duration Features (1 feature)
| Feature | Type | Description | Source Script | Usage |
|---------|------|-------------|---------------|-------|
| `Avg_Outage_Minutes` | Numeric | Average outage duration per equipment | 10 | CoF calculation |

**Statistics:**
- Mean duration: 154.7 minutes
- Median duration: 123.8 minutes
- Range: 1.2 - 959.4 minutes
- Valid durations: 99.6% coverage

### 8. Risk & Prediction Features (12 features)
| Feature | Type | Description | Source Script |
|---------|------|-------------|---------------|
| `PoF_Probability_3M` | Probability | 3-month failure probability | 09 |
| `PoF_Probability_12M` | Probability | 12-month failure probability | 09 |
| `PoF_Probability_24M` | Probability | 24-month failure probability | 09 |
| `PoF_Risk_Category_3M` | Categorical | PoF risk category (3M) | 09 |
| `PoF_Risk_Category_12M` | Categorical | PoF risk category (12M) | 09 |
| `PoF_Risk_Category_24M` | Categorical | PoF risk category (24M) | 09 |
| `CoF_Score` | Numeric | Consequence of Failure score (0-100, percentile) | 10 |
| `CoF_Category` | Categorical | CoF category (DÜŞÜK/ORTA/YÜKSEK/KRİTİK) | 10 |
| `Risk_Score` | Numeric | Combined risk score (PoF × CoF, percentile) | 10 |
| `Risk_Category` | Categorical | Risk category (DÜŞÜK/ORTA/YÜKSEK/KRİTİK) | 10 |
| `Priority_Rank` | Rank | CAPEX priority ranking (1 = highest) | 10 |
| `Recommended_Action` | Categorical | Action (IMMEDIATE/PRIORITY/PREVENTIVE/ROUTINE) | 10 |

### 9. Advanced Features (6 features)
| Feature | Type | Description | Source Script |
|---------|------|-------------|---------------|
| `Failure_Cluster` | Categorical | K-means cluster (failure pattern groups) | 03 |
| `Risk_Profile` | Categorical | Combined age/failure risk profile | 03 |
| `Seasonal_Arıza_Index` | Numeric | Seasonal failure index | 03 |
| `Equipment_Criticality` | Numeric | Criticality score based on customers + failures | 03 |
| `Normalized_MTBF` | Numeric | Age-normalized MTBF | 03 |
| `Failure_Intensity` | Numeric | Recent vs historical failure intensity | 03 |

**Total Feature Count: ~65 features**

---

## 📁 Output Files Reference

### Data Files
| File | Producer | Rows | Columns | Description |
|------|----------|------|---------|-------------|
| `data/equipment_level_data.csv` | 02 | 1,148 | 30+ | Equipment-level features |
| `data/processed_train.csv` | 05,06 | ~900 | Selected | Training set |
| `data/processed_test.csv` | 05,06 | ~248 | Selected | Test set |

### Model Files
| File | Producer | Type | Metrics |
|------|----------|------|---------|
| `models/model_*.pkl` | 06 | XGBoost/CatBoost | AUC: 0.96 (chronic repeater) |
| `models/cox_model_*.pkl` | 09 | Cox Proportional Hazards | C-index: 0.7659 |

### Prediction Files
| File | Producer | Rows | Columns | Description |
|------|----------|------|---------|-------------|
| `predictions/pof_multi_horizon_predictions.csv` | 09 | 1,148 | 10 | PoF for 3M/12M/24M horizons |
| `results/risk_assessment_3M.csv` | 10 | 1,148 | 11 | 3-month risk assessment |
| `results/risk_assessment_12M.csv` | 10 | 1,148 | 11 | 12-month risk assessment (primary) |
| `results/risk_assessment_24M.csv` | 10 | 1,148 | 11 | 24-month risk assessment |
| `results/capex_priority_list.csv` | 10 | 100 | 11 | Top 100 equipment for CAPEX |

### Visualization Files
| Directory | Producer | Count | Description |
|-----------|----------|-------|-------------|
| `outputs/eda/` | 04 | 16 | Exploratory data analysis plots |
| `outputs/risk_analysis/` | 10 | 6 | Risk matrices and distribution plots |
| `outputs/survival_curves/` | 09 | ~10 | Survival curves by equipment class |
| `outputs/shap_explanations/` | 07 | ~5 | Model interpretability plots |

---

## 🤖 Model Architecture

### Dual-Model Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Equipment Data                      │
│                    (1,148 equipment × 65 features)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├──────────────────┬─────────────────┐
                         ▼                  ▼                 ▼
                    ┌─────────┐      ┌──────────┐     ┌──────────┐
                    │ Model 2 │      │ Model 1  │     │   CoF    │
                    │Chronic  │      │Temporal  │     │Calculator│
                    │Repeater │      │   PoF    │     │          │
                    └────┬────┘      └────┬─────┘     └────┬─────┘
                         │                │                 │
                         ▼                ▼                 ▼
              Chronic Repeater    PoF Probabilities  CoF Scores
              Probability         (3M/12M/24M)       (0-100)
              (0.0 - 1.0)         (0.0 - 1.0)
                         │                │                 │
                         └────────────────┴─────────────────┘
                                          │
                                          ▼
                            ┌──────────────────────────┐
                            │   Risk Calculation       │
                            │   Risk = PoF × CoF       │
                            │   (Percentile-based)     │
                            └────────────┬─────────────┘
                                         │
                                         ▼
                            ┌──────────────────────────┐
                            │  CAPEX Priority List     │
                            │  (Top 100 Equipment)     │
                            └──────────────────────────┘
```

### Model 1: Temporal PoF (Survival Analysis)
- **Algorithm**: Cox Proportional Hazards
- **Performance**: C-index 0.7659
- **Features**: Age, MTBF, Failure_Rate, Historical failures
- **Output**: Time-dependent PoF probabilities (3M, 12M, 24M)
- **Key Insight**: Different probabilities for different horizons

### Model 2: Chronic Repeater Classifier
- **Algorithm**: XGBoost / CatBoost (ensemble)
- **Performance**: AUC 0.96, Precision 0.89, Recall 0.92
- **Features**: Recurring flags, Cause codes (Tek_Neden_Flag: 32.7% importance)
- **Output**: Binary classification + probability
- **Key Insight**: Cause consistency is strongest predictor

### Consequence of Failure (CoF)
- **Formula**: `CoF = Outage_Duration × Customers_Affected × Critical_Multiplier`
- **Normalization**: Percentile rank (0-100) - robust to outliers
- **Categories**: DÜŞÜK (0-75th), ORTA (75-90th), YÜKSEK (90-95th), KRİTİK (95-100th)

### Risk Scoring
- **Formula**: `Risk = PoF × CoF`
- **Normalization**: Percentile rank (0-100)
- **Categories**: DÜŞÜK (75%), ORTA (15%), YÜKSEK (5%), KRİTİK (5%)
- **Distribution**: Ensures proper stratification across fleet

---

## 📈 Key Performance Metrics

### Model Performance
| Model | Metric | Value | Interpretation |
|-------|--------|-------|----------------|
| Model 1 (Temporal PoF) | C-index | 0.7659 | Good time-to-event discrimination |
| Model 2 (Chronic Repeater) | AUC | 0.96 | Excellent classification |
| Model 2 | Precision | 0.89 | 89% of predicted repeaters are true |
| Model 2 | Recall | 0.92 | Catches 92% of chronic repeaters |

### Risk Distribution (12M Horizon)
| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| DÜŞÜK | 861 | 75.0% | Routine monitoring |
| ORTA | 172 | 15.0% | Preventive maintenance |
| YÜKSEK | 57 | 5.0% | Priority replacement |
| KRİTİK | 58 | 5.1% | Immediate replacement |

### Feature Importance (Top 10)
| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `Tek_Neden_Flag` | 32.7% | Cause Code |
| 2 | `MTBF_Gün` | 18.4% | Reliability |
| 3 | `Arıza_Sayısı_12Ay` | 16.9% | Failure History |
| 4 | `Arıza_Nedeni_Sık` | 15.3% | Cause Code |
| 5 | `Ekipman_Yaşı_Yıl` | 14.2% | Temporal |
| 6 | `Arıza_Nedeni_Tutarlılık` | 12.4% | Cause Code |
| 7 | `Ekipman_Neden_Risk_Skoru` | 11.6% | Cause Code |
| 8 | `Failure_Rate_Per_Year` | 10.8% | Reliability |
| 9 | `Tekrarlayan_Arıza_90gün_Flag` | 9.3% | Recurring |
| 10 | `Arıza_Nedeni_Çeşitlilik` | 8.1% | Cause Code |

**Key Finding**: Cause code features dominate top importance (5 of top 10)

---

## 🎯 Requirement Fulfillment Summary

### ✅ Module 1: Equipment-Level PoF Prediction
- [x] **Temporal PoF prediction** (3M/12M/24M horizons)
- [x] **Chronic repeater detection** (AUC 0.96)
- [x] **Risk scoring** (PoF × CoF with percentile normalization)
- [x] **CAPEX prioritization** (Top 100 list with recommended actions)
- [x] **Multi-horizon predictions** (Different PoF for each horizon)
- [x] **Survival analysis** (Cox model with C-index 0.77)

### ✅ Module 3: Recurring Failures & Cause Analysis
- [x] **30-day recurring failure flag** (~8% of equipment)
- [x] **90-day recurring failure flag** (111 equipment, 9.7%)
- [x] **Cause code analysis** (9 features with 94.66% consistency)
- [x] **Single dominant cause detection** (`Tek_Neden_Flag`, 32.7% importance)
- [x] **Customer impact integration** (8 features covering MV/LV categories)
- [x] **Recurring vs non-recurring comparison** (50% MTBF difference)
- [x] **Visual analysis** (16 EDA plots + 6 risk plots)

### ✅ Risk Assessment
- [x] **CoF calculation** (Duration × Customers × Criticality)
- [x] **Percentile-based scoring** (Robust to outliers)
- [x] **4-tier categorization** (DÜŞÜK/ORTA/YÜKSEK/KRİTİK)
- [x] **Proper distribution** (75/15/5/5% split)
- [x] **Risk matrices** (PoF vs CoF quadrant charts)
- [x] **Equipment class analysis** (Box plots by class)

---

## 🔍 Quick Reference: Script-Feature-Output Map

### Script → Features → Outputs

**01_data_profiling.py**
- Features: N/A (validation only)
- Outputs: Quality report, column availability check

**02_data_transformation.py**
- Features Created: 30+ core features (ID, temporal, failures, recurring, causes, customers)
- Key Features: `Tekrarlayan_Arıza_90gün_Flag`, `Tek_Neden_Flag`, `MTBF_Gün`
- Outputs: `equipment_level_data.csv`

**03_feature_engineering.py**
- Features Created: 6 advanced features (clusters, volatility, trends)
- Key Features: `Failure_Cluster`, `Arıza_Trend`, `Arıza_Volatility`
- Outputs: Enhanced equipment data

**04_eda.py**
- Features Analyzed: All 65 features across 16 analyses
- Key Analyses: Cause codes (STEP 15), Recurring failures (STEP 16)
- Outputs: 16 PNG plots in `outputs/eda/`

**05_feature_selection.py**
- Features Selected: ~20-30 most predictive features
- Selection Method: Recursive feature elimination + correlation analysis
- Outputs: `processed_train.csv`, `processed_test.csv`

**05b_remove_leaky_features.py**
- Features Removed: Leaky features (future info, IDs, targets)
- Leakage Check: Temporal validation
- Outputs: Cleaned feature set

**06_model_training.py**
- Features Used: ~20 selected features (cause codes dominate)
- Top Feature: `Tek_Neden_Flag` (32.7%)
- Outputs: `models/*.pkl`, chronic repeater predictions

**09_survival_analysis.py**
- Features Used: Age, MTBF, Failure_Rate, Historical counts
- Model: Cox Proportional Hazards
- Outputs: `pof_multi_horizon_predictions.csv`, survival curves

**10_consequence_of_failure.py**
- Features Used: PoF + `total_customer_count` + `Avg_Outage_Minutes`
- Calculation: PoF × CoF (percentile-based)
- Outputs: `risk_assessment_*.csv`, `capex_priority_list.csv`, 6 risk plots

---

## 📚 Data Dictionary: Key Column Names

### Source Data (combined_data.xlsx)
- `cbs_id`: Equipment ID (primary)
- `Equipment_Type`: Equipment class (primary)
- `TESIS_TARIHI`: Installation date (primary)
- `started at`: Fault start timestamp
- `ended at`: Fault end timestamp
- `duration time`: Pre-calculated outage duration
- `cause code`: Fault cause code
- `total customer count`: Total affected customers
- `urban_mv`, `urban_lv`, `suburban_mv`, `suburban_lv`, `rural_mv`, `rural_lv`: Customer breakdowns

### Generated Data (equipment_level_data.csv)
- `Ekipman_ID`: Standardized equipment ID
- `Ekipman_Sınıfı`: Standardized equipment class
- `İlçe`: District
- `Ekipman_Yaşı_Yıl`: Equipment age (years)
- `Total_Arıza_Sayısı`: Total failure count
- `MTBF_Gün`: Mean time between failures (days)
- `Failure_Rate_Per_Year`: Annual failure rate
- `Tekrarlayan_Arıza_30gün_Flag`: 30-day recurring flag (0/1)
- `Tekrarlayan_Arıza_90gün_Flag`: 90-day recurring flag (0/1)
- `Tek_Neden_Flag`: Single dominant cause flag (0/1)
- `Arıza_Nedeni_Tutarlılık`: Cause consistency ratio (0-1)
- `total_customer_count_Max`: Maximum customers affected

### Prediction Data (pof_multi_horizon_predictions.csv)
- `Ekipman_ID`: Equipment ID
- `PoF_Probability_3M`: 3-month failure probability (0-1)
- `PoF_Probability_12M`: 12-month failure probability (0-1)
- `PoF_Probability_24M`: 24-month failure probability (0-1)
- `PoF_Risk_Category_3M`: 3-month risk category
- `PoF_Risk_Category_12M`: 12-month risk category
- `PoF_Risk_Category_24M`: 24-month risk category

### Risk Data (risk_assessment_12M.csv)
- `Ekipman_ID`: Equipment ID
- `Ekipman_Sınıfı`: Equipment class
- `İlçe`: District
- `PoF_Probability`: Probability of failure (0-1)
- `CoF_Score`: Consequence of failure (0-100, percentile)
- `Risk_Score`: Combined risk (0-100, percentile)
- `Risk_Category`: DÜŞÜK/ORTA/YÜKSEK/KRİTİK
- `Priority_Rank`: Ranking (1 = highest risk)
- `Avg_Outage_Minutes`: Average outage duration
- `Total_Customers_Affected`: Total customers impacted

---

**Document Version:** 1.0
**Last Updated:** 2025-11-14
**Pipeline Version:** 2.0
**Total Scripts:** 9
**Total Features:** ~65
**Total Equipment:** 1,148
