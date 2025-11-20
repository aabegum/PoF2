# PoF2 Pipeline - Execution Order & Script Guide
**Turkish EDAŞ Equipment Failure Prediction Pipeline**

Last Updated: November 20, 2025

---

## 🎯 **Quick Start**

To run the entire pipeline:
```bash
python run_pipeline.py
```

To run a specific step:
```bash
python 01_data_profiling.py
python 02_data_transformation.py
# ... etc
```

---

## 📋 **REQUIRED EXECUTION ORDER (10 Steps)**

### **Data Preparation Phase (Steps 1-4)**

#### **STEP 1: Data Profiling**
**Script:** `01_data_profiling.py`
**Purpose:** Validate raw data quality and temporal coverage
**Input:** `data/combined_data.xlsx` (raw fault records)
**Output:** Console validation report (no files)
**Duration:** ~30 seconds
**Key Checks:**
- Equipment ID strategy
- Temporal coverage (100% timestamps)
- Data completeness
- Customer impact coverage

---

#### **STEP 2: Data Transformation**
**Script:** `02_data_transformation.py`
**Purpose:** Transform fault-level → equipment-level data
**Input:** `data/combined_data.xlsx` (1,210+ fault records)
**Output:** `data/equipment_level_data.csv` (789 equipment records)
**Duration:** ~1-2 minutes
**Key Features Created:**
- Equipment age (2 columns: days + years)
- MTBF calculations
- Temporal features (3M/6M/12M fault counts)
- Customer impact ratios
- ~70 base features

**Validation:** Checks 700-1000 rows, required columns present

---

#### **STEP 3: Feature Engineering**
**Script:** `03_feature_engineering.py`
**Purpose:** Create optimal 30-feature set (TIER 1-8)
**Input:** `data/equipment_level_data.csv`
**Output:** `data/features_engineered.csv` (30 features)
**Duration:** ~1 minute
**Key Features Created:**
- Age-to-expected-life ratios
- MTBF trend and variability
- Overdue factors
- Interaction features

**Validation:** Expects exactly 30 features

---

#### **STEP 4: Feature Selection**
**Script:** `05_feature_selection.py`
**Purpose:** Remove leaky/redundant features (VIF analysis)
**Input:** `data/features_engineered.csv` (30 features)
**Output:** `data/features_reduced.csv` (25-30 features)
**Duration:** ~1 minute
**Process:**
1. Remove data leakage features
2. Remove highly correlated features (r > 0.85)
3. VIF analysis (remove multicollinear features, VIF > 10)

**Validation:** Ensures 25-30 features, no leakage patterns

---

### **Model Training Phase (Steps 5-6)**

#### **STEP 5: Temporal PoF Model**
**Script:** `06_temporal_pof_model.py`
**Purpose:** Train XGBoost/CatBoost for 4 time horizons
**Input:** `data/features_reduced.csv`
**Output:**
- `predictions/predictions_3m.csv`
- `predictions/predictions_6m.csv`
- `predictions/predictions_12m.csv`
- `predictions/predictions_24m.csv`
- `models/xgboost_*.pkl`

**Duration:** ~2-3 minutes
**What It Predicts:** WHEN equipment will fail (3M/6M/12M/24M windows)
**Expected Performance:** AUC 0.75-0.85

**Validation:** Checks all 4 prediction files exist, required columns present

---

#### **STEP 6: Chronic Classifier**
**Script:** `06_chronic_classifier.py`
**Purpose:** Identify chronically failing equipment
**Input:** `data/features_reduced.csv`
**Output:** `predictions/chronic_repeaters.csv`
**Duration:** ~2-3 minutes
**What It Predicts:** WHICH equipment are failure-prone (90-day recurrence)
**Expected Performance:** AUC 0.85-0.92

**Validation:** Checks chronic_repeaters.csv exists

---

### **Model Analysis Phase (Steps 7-8)**

#### **STEP 7: Model Explainability**
**Script:** `07_explainability.py`
**Purpose:** SHAP analysis for feature importance
**Input:**
- `data/features_reduced.csv`
- `models/xgboost_*.pkl`

**Output:**
- `outputs/explainability/*.png` (SHAP plots)
- `results/feature_importance.csv`

**Duration:** ~1-2 minutes
**What It Does:** Explains which features drive predictions

**Validation:** No file validation (creates visualizations)

---

#### **STEP 8: Probability Calibration**
**Script:** `08_calibration.py`
**Purpose:** Calibrate model probabilities for accurate risk estimates
**Input:**
- `data/features_reduced.csv`
- `models/xgboost_*.pkl`

**Output:**
- `models/calibrated_*.pkl`
- `outputs/calibration/*.png` (calibration curves)

**Duration:** ~1 minute
**What It Does:** Ensures predicted probabilities match actual failure rates

**Validation:** No file validation (updates models in place)

---

### **Alternative Model Phase (Step 9)**

#### **STEP 9: Cox Survival Model**
**Script:** `09_cox_survival_model.py`
**Purpose:** Alternative time-to-event predictions using Cox PH
**Input:**
- `data/features_reduced.csv`
- `data/combined_data.xlsx` (for survival times)

**Output:**
- `predictions/pof_multi_horizon_predictions.csv`
- `outputs/survival/*.png` (Kaplan-Meier curves)

**Duration:** ~2-3 minutes
**What It Does:**
- Cox Proportional Hazards model
- Kaplan-Meier survival curves
- Multi-horizon predictions (3M/6M/12M/24M)

**Why Both Models?**
- **Step 5 (Temporal PoF):** Binary classification for each horizon
- **Step 9 (Cox Survival):** Time-to-event analysis (continuous time)
- **Use Case:** Cox provides hazard ratios, Temporal PoF provides simpler probabilities

**Validation:** Checks pof_multi_horizon_predictions.csv exists

---

### **Risk Assessment Phase (Step 10)**

#### **STEP 10: Risk Assessment**
**Script:** `10_consequence_of_failure.py`
**Purpose:** Calculate PoF × CoF = Risk, generate CAPEX priorities
**Input:**
- `predictions/pof_multi_horizon_predictions.csv` (from Step 9)
- `data/equipment_level_data.csv` (customer impact)
- `data/combined_data.xlsx` (outage durations)

**Output:**
- `results/risk_assessment_3M.csv`
- `results/risk_assessment_6M.csv`
- `results/risk_assessment_12M.csv`
- `results/risk_assessment_24M.csv`
- `results/capex_priority_list.csv` (Top 100)
- `outputs/risk_analysis/*.png`

**Duration:** ~1 minute
**Risk Formula:**
```
CoF = Outage_Duration × Customer_Count × Critical_Multiplier
Risk = PoF × CoF
Risk_Score = normalize(Risk, 0, 100)
```

**Risk Categories:**
- **DÜŞÜK (Low):** 0-40
- **ORTA (Medium):** 40-70
- **YÜKSEK (High):** 70-90
- **KRİTİK (Critical):** 90-100

**Validation:** Checks all 4 risk files + CAPEX list exist

---

## 🔄 **Pipeline Flow Diagram**

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                      │
└─────────────────────────────────────────────────────────┘
                            ↓
    [1] Data Profiling → Validate quality
                            ↓
    [2] Data Transformation → 1,210 faults → 789 equipment
                            ↓
    [3] Feature Engineering → Create 30 features
                            ↓
    [4] Feature Selection → Remove leakage + VIF (25-30)
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   MODEL TRAINING (DUAL)                  │
└─────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────┴───────────────────────┐
    ↓                                               ↓
[5] Temporal PoF Model                    [6] Chronic Classifier
    (XGBoost/CatBoost)                         (90-day recurrence)
    4 horizons: 3M/6M/12M/24M                  WHICH are failure-prone?
    WHEN will it fail?
    ↓                                               ↓
    └───────────────────────┬───────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   MODEL ANALYSIS                         │
└─────────────────────────────────────────────────────────┘
                            ↓
    [7] Model Explainability → SHAP analysis
                            ↓
    [8] Probability Calibration → Calibrate probabilities
                            ↓
┌─────────────────────────────────────────────────────────┐
│              ALTERNATIVE MODEL (OPTIONAL)                │
└─────────────────────────────────────────────────────────┘
                            ↓
    [9] Cox Survival Model → Time-to-event analysis
        (Alternative to Step 5)
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   RISK ASSESSMENT                        │
└─────────────────────────────────────────────────────────┘
                            ↓
    [10] Risk Assessment → PoF × CoF = Risk
         Generate CAPEX Priority List (Top 100)
```

---

## 🎯 **Model Strategy: Why 3 Models?**

### **1. Temporal PoF Model (Step 5) - PRIMARY**
**Purpose:** Binary classification for each time horizon
**Output:** 4 probability scores (3M, 6M, 12M, 24M)
**Use Case:**
- Simple probability: "70% chance of failure in 6 months"
- Easy to explain to stakeholders
- Direct risk calculation

**Algorithm:** XGBoost/CatBoost (gradient boosting)

---

### **2. Chronic Classifier (Step 6) - COMPLEMENTARY**
**Purpose:** Identify equipment with recurring failures
**Output:** Binary classification (chronic vs non-chronic)
**Use Case:**
- Replace vs Repair decision
- Chronic equipment → Immediate replacement
- Non-chronic → Targeted repair

**Algorithm:** XGBoost/CatBoost (gradient boosting)

---

### **3. Cox Survival Model (Step 9) - ALTERNATIVE**
**Purpose:** Time-to-event analysis (survival analysis)
**Output:** Hazard ratios + survival curves
**Use Case:**
- Continuous time-to-failure prediction
- Hazard ratios (how features affect failure rate)
- More sophisticated statistical analysis

**Algorithm:** Cox Proportional Hazards

**When to Use:**
- Research/academic analysis: Use Cox Survival
- Production/operations: Use Temporal PoF (simpler)
- Strategic planning: Use both and compare

---

## ⏱️ **Total Pipeline Runtime**

| Phase | Steps | Duration |
|-------|-------|----------|
| Data Preparation | 1-4 | ~3-5 min |
| Model Training | 5-6 | ~4-6 min |
| Model Analysis | 7-8 | ~2-3 min |
| Alternative Model | 9 | ~2-3 min |
| Risk Assessment | 10 | ~1 min |
| **TOTAL** | **1-10** | **~12-18 min** |

---

## 📁 **Key Output Files**

### **Must Have (Critical):**
```
data/
  ├── equipment_level_data.csv       [Step 2]
  ├── features_engineered.csv        [Step 3]
  └── features_reduced.csv           [Step 4]

predictions/
  ├── predictions_3m.csv             [Step 5]
  ├── predictions_6m.csv             [Step 5]
  ├── predictions_12m.csv            [Step 5]
  ├── predictions_24m.csv            [Step 5]
  ├── chronic_repeaters.csv          [Step 6]
  └── pof_multi_horizon_predictions.csv [Step 9]

results/
  ├── risk_assessment_3M.csv         [Step 10]
  ├── risk_assessment_6M.csv         [Step 10]
  ├── risk_assessment_12M.csv        [Step 10]
  ├── risk_assessment_24M.csv        [Step 10]
  └── capex_priority_list.csv        [Step 10] ⭐ KEY OUTPUT
```

### **Nice to Have (Analysis):**
```
models/
  ├── xgboost_3m.pkl
  ├── xgboost_6m.pkl
  ├── xgboost_12m.pkl
  └── xgboost_24m.pkl

outputs/
  ├── explainability/*.png
  ├── calibration/*.png
  ├── survival/*.png
  └── risk_analysis/*.png
```

---

## 🚫 **Common Mistakes**

### **1. Running Steps Out of Order**
❌ **Wrong:** Running Step 5 before Step 4
✅ **Correct:** Follow 1→2→3→4→5→6→7→8→9→10

**Why:** Each step depends on previous step's output files

---

### **2. Skipping Validation Steps**
❌ **Wrong:** Ignoring validation errors
✅ **Correct:** Fix validation errors before continuing

**Why:** Invalid data propagates errors through entire pipeline

---

### **3. Mixing Step 5 and Step 9 Outputs**
❌ **Wrong:** Using Step 5 predictions for Step 10
✅ **Correct:** Step 10 uses Step 9 (Cox Survival) outputs

**Why:** Step 10 expects multi-horizon format from Step 9

---

## 🔍 **Validation**

Every step with data outputs is automatically validated:

```bash
# Validate specific step
python pipeline_validation.py --step 2

# Validate all steps
python pipeline_validation.py --all

# Quick check (files only)
python pipeline_validation.py --step 2 --quick
```

---

## 📞 **Troubleshooting**

### **Pipeline Failed at Step X**
1. Check log file: `logs/run_TIMESTAMP/0X_script_name.log`
2. Look for error messages in STDERR section
3. Validate previous step output: `python pipeline_validation.py --step X-1`

### **Validation Failed**
1. Check error message for specific issue
2. Common issues:
   - Missing input file → Run previous step
   - Wrong number of features → Check feature engineering
   - Missing columns → Check data transformation

### **Model Performance Issues**
1. Check Step 7 (Explainability) for feature importance
2. Review Step 8 (Calibration) curves
3. Compare Step 5 vs Step 9 predictions

---

## 📚 **Related Documentation**

- `config.py` - All configuration parameters
- `pipeline_validation.py` - Validation framework
- `OPTION_B_PROGRESS.md` - Pipeline optimization progress
- `PIPELINE_RUNNER_FIX.md` - Recent fixes applied

---

**Last Updated:** November 20, 2025
**Pipeline Version:** v5.2 (Validation + Clear Naming)
**Author:** Data Analytics Team
