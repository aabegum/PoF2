# PoF2 Pipeline Usage Guide

**Last Updated**: 2025-11-25
**Pipeline Version**: v2.0 (Phase 1 Optimized)

This guide explains how to run the complete PoF2 equipment failure prediction pipeline.

---

## 🚀 Quick Start

### Recommended: Automated Pipeline Runner

```bash
python run_pipeline.py
```

This runs all 11 steps sequentially with automatic logging and validation.

---

## 📋 Pipeline Steps

The production pipeline consists of **11 steps** (10 main + 1 optional diagnostic):

| Step | Script | Purpose | Time |
|------|--------|---------|------|
| **1** | `01_data_profiling.py` | Validate data quality and temporal coverage | ~30s |
| **2** | `02_data_transformation.py` | Transform fault-level to equipment-level | ~1min |
| **3** | `03_feature_engineering.py` | Create optimal feature set (TIER 1-8) | ~2min |
| **4** | `04_feature_selection.py` | Leakage removal + VIF reduction | ~1min |
| **4b** | `05_equipment_id_audit.py` | Verify ID consolidation (optional) | ~30s |
| **5** | `06_temporal_pof_model.py` | Train XGBoost/CatBoost temporal models | ~2min |
| **6** | `07_chronic_classifier.py` | Train chronic repeater classifier | ~1min |
| **7** | `08_explainability.py` | SHAP feature importance analysis | ~2min |
| **8** | `09_calibration.py` | Calibrate probability predictions | ~3min |
| **9** | `10_survival_model.py` | Cox proportional hazards model | ~2min |
| **10** | `11_consequence_of_failure.py` | Calculate Risk = PoF × CoF | ~1min |

**Total Execution Time**: ~15-20 minutes

---

## 📂 Output Structure

After successful pipeline execution:

### Predictions
```
predictions/
├── predictions_3m.csv                    # 3-month temporal PoF predictions
├── predictions_6m.csv                    # 6-month temporal PoF predictions
├── predictions_12m.csv                   # 12-month temporal PoF predictions
├── chronic_repeaters.csv                 # Chronic equipment classification
├── pof_multi_horizon_predictions.csv     # Survival model multi-horizon predictions
└── capex_priority_list.csv               # Top 100 highest-risk equipment
```

### Models
```
models/
├── xgboost_3m.pkl                        # Temporal PoF model (3-month)
├── xgboost_6m.pkl                        # Temporal PoF model (6-month)
├── xgboost_12m.pkl                       # Temporal PoF model (12-month)
├── catboost_3m.pkl                       # CatBoost temporal model (3-month)
├── catboost_6m.pkl                       # CatBoost temporal model (6-month)
├── catboost_12m.pkl                      # CatBoost temporal model (12-month)
├── chronic_classifier.pkl                # Chronic repeater classifier
├── calibrated_isotonic_6m.pkl            # Isotonic calibrated model (6-month)
├── calibrated_isotonic_12m.pkl           # Isotonic calibrated model (12-month)
├── calibrated_sigmoid_6m.pkl             # Platt calibrated model (6-month)
├── calibrated_sigmoid_12m.pkl            # Platt calibrated model (12-month)
└── cox_survival_model.pkl                # Cox proportional hazards model
```

### Results (Analysis CSVs)
```
results/
├── model_performance_comparison.csv      # XGBoost vs CatBoost metrics
├── feature_importance_by_horizon.csv     # Feature importance per horizon
├── shap_feature_importance.csv           # SHAP-based feature importance
├── calibration_metrics.csv               # Calibration performance metrics
├── pof_category_aggregation.csv          # PoF statistics by equipment class
└── pof_outliers.csv                      # High-risk outlier equipment
```

### Visualizations
```
outputs/
├── explainability/
│   ├── shap_summary_3m.png               # SHAP summary (3-month)
│   ├── shap_summary_6m.png               # SHAP summary (6-month)
│   ├── shap_summary_12m.png              # SHAP summary (12-month)
│   ├── shap_dependence_*.png             # SHAP dependence plots
│   └── shap_waterfall_examples.png       # Individual prediction explanations
├── calibration/
│   ├── calibration_curves_6m.png         # Calibration curves (6-month)
│   ├── calibration_curves_12m.png        # Calibration curves (12-month)
│   └── reliability_diagrams.png          # Reliability diagrams
├── survival/
│   ├── kaplan_meier_by_class.png         # Kaplan-Meier curves by equipment class
│   ├── kaplan_meier_by_district.png      # Kaplan-Meier curves by district
│   └── survival_probabilities.png        # Survival probability trends
└── model_evaluation/
    ├── roc_curves.png                    # ROC curves (all horizons)
    ├── precision_recall_curves.png       # PR curves (all horizons)
    └── confusion_matrices.png            # Confusion matrices
```

### Logs
```
logs/
└── run_YYYYMMDD_HHMMSS/                  # Timestamped run directory
    ├── pipeline_master.log                # All outputs combined
    ├── pipeline_summary.txt               # Execution summary with timings
    ├── step_01_data_profiling.log
    ├── step_02_data_transformation.log
    ├── step_03_feature_engineering.log
    ├── step_04_feature_selection.log
    ├── step_05_temporal_pof_model.log
    ├── step_07_chronic_classifier.log
    ├── step_08_explainability.log
    ├── step_09_calibration.log
    ├── step_09_survival_model.log
    └── step_10_risk_assessment.log
```

---

## 🔍 Understanding the Results

### CAPEX Priority List (`capex_priority_list.csv`)

The most important output for decision-making. Contains top 100 highest-risk equipment with:

| Column | Description |
|--------|-------------|
| `Priority_Rank` | Priority ranking (1 = highest risk) |
| `Ekipman_ID` | Equipment identifier |
| `Ekipman_Sınıfı` | Equipment class (Ayırıcı, Trafo, etc.) |
| `İlçe` | District (Salihli, Alaşehir, Gördes) |
| `PoF_Probability_12M` | 12-month failure probability (0.0-1.0) |
| `CoF_Score` | Consequence of failure score (0-100) |
| `Risk_Score` | Combined risk score (0-100) |
| `Risk_Category` | DÜŞÜK / ORTA / YÜKSEK / KRİTİK |
| `Avg_Outage_Minutes` | Average outage duration |
| `Total_Customers_Affected` | Total customers impacted |

### Risk Categories (Percentile-Based)

Equipment is classified into 4 risk categories based on percentiles:

- **DÜŞÜK (Low)**: 0-75th percentile (~75% of equipment)
- **ORTA (Medium)**: 75-90th percentile (~15% of equipment)
- **YÜKSEK (High)**: 90-95th percentile (~5% of equipment)
- **KRİTİK (Critical)**: 95-100th percentile (~5% of equipment)

### Temporal Predictions (`predictions_*.csv`)

Individual horizon predictions from XGBoost models:

- **predictions_3m.csv**: 3-month failure predictions
- **predictions_6m.csv**: 6-month failure predictions
- **predictions_12m.csv**: 12-month failure predictions

Each contains:
- `Ekipman_ID`, `Equipment_Class`
- `PoF_Probability`: Failure probability
- `Risk_Score`: Normalized risk score (0-100)
- `Risk_Class`: Low / Medium / High / Critical

### Survival Model Predictions (`pof_multi_horizon_predictions.csv`)

Multi-horizon predictions from Cox proportional hazards model:

- `PoF_Probability_3M`: 3-month survival-based PoF
- `PoF_Probability_6M`: 6-month survival-based PoF
- `PoF_Probability_12M`: 12-month survival-based PoF
- `PoF_Probability_24M`: 24-month survival-based PoF
- `Risk_Category`: Overall risk classification

**Note**: The final risk assessment (Step 10) uses these survival-based predictions.

### Chronic Repeaters (`chronic_repeaters.csv`)

Equipment classified by recurrence pattern (90-day window):

- `Ekipman_ID`, `Equipment_Class_Primary`
- `Chronic_Probability`: Probability of being a chronic repeater (0.0-1.0)
- `Chronic_Prediction`: Binary classification (0 = Normal, 1 = Chronic)
- `Is_Chronic`: Final chronic flag

---

## 🔧 Manual Execution (Individual Steps)

If you need to run specific steps individually:

```bash
# Data preparation
python 01_data_profiling.py
python 02_data_transformation.py
python 03_feature_engineering.py
python 04_feature_selection.py

# Optional: ID audit
python 05_equipment_id_audit.py

# Model training
python 06_temporal_pof_model.py
python 07_chronic_classifier.py

# Model analysis
python 08_explainability.py
python 09_calibration.py

# Survival analysis
python 10_survival_model.py

# Risk assessment
python 11_consequence_of_failure.py
```

**Important**: Steps must be run in order - each step depends on outputs from previous steps.

---

## 📊 Optional Analysis Scripts

Not part of the main production pipeline but useful for research and diagnostics:

### Exploratory Data Analysis
```bash
python analysis/exploratory/04_eda.py
```

Generates 16 comprehensive exploratory analyses (~5 min runtime):
- Equipment class distributions
- Temporal failure trends
- District analysis
- Age and MTBF analysis
- Correlation matrices
- Recurring failure patterns

### Baseline Comparison
```bash
python analysis/diagnostics/06b_logistic_baseline.py
```

Trains logistic regression baseline for comparison (~2 min runtime).

### Archived Scripts

See `archived/README.md` for scripts not in production pipeline:
- **07_walkforward_validation.py**: Walk-forward validation with expanding window
- **08_class_imbalance_analysis.py**: Class imbalance analysis and recommendations
- **09_train_with_smote.py**: SMOTE oversampling experiments
- **logger.py**: Unused logging infrastructure module

These can be run for research/diagnostic purposes if needed.

---

## ⚙️ Configuration

Pipeline behavior is controlled by `config.py`. Key parameters:

### Data Configuration
```python
CUTOFF_DATE = '2024-06-25'           # Temporal cutoff for predictions
INPUT_FILE = DATA_DIR / 'raw_fault_data.xlsx'
```

### Model Parameters
```python
RANDOM_STATE = 42                     # Reproducibility
TEST_SIZE = 0.30                      # 30% test split
N_FOLDS = 3                           # Cross-validation folds
HORIZONS = {'3M': 90, '6M': 180, '12M': 365}  # Prediction windows
```

### Feature Selection
```python
VIF_THRESHOLD = 10                    # VIF threshold for multicollinearity
VIF_TARGET = 10                       # Target VIF after removal
CORRELATION_THRESHOLD = 0.85          # Correlation threshold
```

### Protected Features
30 optimal features are protected from removal - see `config.py` line 116-149.

---

## 🐛 Troubleshooting

### Common Issues

**Error: "File not found: data/raw_fault_data.xlsx"**
- Ensure input file exists in `data/` directory
- Check `config.py` INPUT_FILE path

**Error: "ModuleNotFoundError: No module named 'xgboost'"**
- Install dependencies: `pip install -r requirements.txt`

**Warning: "Model not found: models/xgboost_6m.pkl"**
- Run pipeline from Step 1 (or at least Step 5 for model training)
- Don't skip steps - they depend on each other

**Error: "Target column has 100% positive class (3M)"**
- Expected for 3M horizon (excluded from calibration)
- Pipeline automatically filters out 3M for calibration
- No action needed

### Validation Failures

The pipeline includes automatic validation at each step. If validation fails:

1. Check the log file for the failed step
2. Verify input data quality (Step 1 output)
3. Ensure sufficient equipment records (minimum 100)
4. Check for missing required columns

---

## 📖 Additional Documentation

- **PIPELINE_EXECUTION_ORDER.md**: Detailed pipeline flow with data transformations
- **DUAL_MODELING_ANALYSIS.md**: Analysis of temporal vs survival modeling approach
- **COLUMN_NAMING_STANDARD.md**: Turkish/English column naming conventions
- **PIPELINE_CLEANUP_SUMMARY.md**: Recent cleanup changes and improvements

---

## 🔄 Pipeline Workflow Summary

```
┌──────────────────────────────────────────────────────────────┐
│                   RAW FAULT DATA                             │
│              (data/raw_fault_data.xlsx)                      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  STEP 1-2: DATA PREPARATION   │
         │  Profile → Transform          │
         │  Output: equipment_level.csv  │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  STEP 3-4: FEATURE PIPELINE   │
         │  Engineer → Select            │
         │  Output: features_reduced.csv │
         └───────────────┬───────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
  ┌───────────────┐           ┌───────────────┐
  │ TEMPORAL MODEL│           │ SURVIVAL MODEL│
  │ Steps 5-8     │           │ Step 9        │
  │ XGBoost/SHAP  │           │ Cox PH        │
  └───────┬───────┘           └───────┬───────┘
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  STEP 10: RISK ASSESS │
            │  PoF × CoF = Risk     │
            │  CAPEX Priority List  │
            └───────────────────────┘
```

---

## 💡 Tips for Best Results

1. **Run full pipeline regularly** to keep predictions current
2. **Monitor calibration metrics** (Brier score, log loss) for model quality
3. **Use SHAP plots** (Step 7) to explain predictions to stakeholders
4. **Focus on CAPEX priority list** for actionable insights
5. **Compare temporal vs survival predictions** for validation
6. **Archive logs** for each run to track model performance over time

---

**Last Updated**: 2025-11-25
**For Questions**: See main README.md or PIPELINE_EXECUTION_ORDER.md
