# PoF2: Turkish EDAŞ Equipment Failure Prediction Pipeline

**Varlık Yönetimi Arıza İlişkisi (Asset Management Fault Relationship)**
Predictive maintenance and risk assessment for Turkish electricity distribution companies

---

## 🎯 Quick Start

```bash
# Run the complete pipeline (13 steps, ~12-18 minutes)
python run_pipeline.py
```

**Output**: CAPEX priority list + risk assessments for 3M/6M/12M horizons

---

## 📚 Documentation

### **Essential Guides**
- **[Pipeline Execution Order](PIPELINE_EXECUTION_ORDER.md)** - Complete technical reference (13 steps)
- **[Critical Issues](CRITICAL_ISSUES_BEFORE_PIPELINE_EXECUTION.md)** - Action required before production
- **[Phase 1 Summary](PHASE_1_COMPLETION_SUMMARY.md)** - What was fixed in Phase 1

### **Configuration**
- **[config.py](config.py)** - All pipeline parameters (dates, horizons, VIF thresholds)
- **[column_mapping.py](column_mapping.py)** - EN ↔ TR column mappings + leakage detection

### **Additional Documentation**
See [docs/](docs/) folder for detailed guides:
- Pipeline usage and troubleshooting
- Healthy equipment integration
- Data requirements
- Architecture analysis

---

## 📊 Pipeline Overview

### **13-Step Production Pipeline**

| Step | Script | Purpose | Duration |
|------|--------|---------|----------|
| **0** | `00_input_data_source_analysis.py` | Validate raw input data | ~30s |
| **1** | `01_data_profiling.py` | Profile data quality | ~1min |
| **2a** | `02a_healthy_equipment_loader.py` | Load healthy equipment (optional) | ~2min |
| **2** | `02_data_transformation.py` | Fault → equipment level | ~1-2min |
| **3** | `03_feature_engineering.py` | Create 30 optimal features | ~1min |
| **4** | `04_feature_selection.py` | Leakage removal + VIF | ~1min |
| **5** | `05_equipment_id_audit.py` | Verify ID consistency (optional) | ~30s |
| **6** | `06_temporal_pof_model.py` | Train PoF models (3M/6M/12M) | ~2-3min |
| **7** | `07_chronic_classifier.py` | Identify chronic failures | ~2-3min |
| **8** | `08_explainability.py` | SHAP feature importance | ~1-2min |
| **9** | `09_calibration.py` | Calibrate probabilities | ~1min |
| **10** | `10_survival_model.py` | Cox survival analysis | ~2-3min |
| **11** | `11_consequence_of_failure.py` | Risk = PoF × CoF | ~1min |

**Total Runtime**: ~12-18 minutes

---

## 📁 Key Output Files

### **Risk Assessment (Primary Deliverable)**
```
results/
├── capex_priority_list.csv        ⭐ Top 100 equipment for replacement
├── risk_assessment_3M.csv          3-month risk scores
├── risk_assessment_6M.csv          6-month risk scores
└── risk_assessment_12M.csv         12-month risk scores
```

### **Predictions**
```
predictions/
├── predictions_3m.csv              3-month temporal PoF
├── predictions_6m.csv              6-month temporal PoF
├── predictions_12m.csv             12-month temporal PoF
├── chronic_repeaters.csv           Chronic failure classification
└── pof_multi_horizon_predictions.csv  Cox survival predictions
```

### **Intermediate Data**
```
data/
├── equipment_level_data.csv        Equipment-level dataset (789 records)
├── features_engineered.csv         Engineered features (30 features)
└── features_reduced.csv            Final feature set (after VIF)
```

---

## ⚙️ Configuration

### **Key Parameters** (edit `config.py`)

```python
# Temporal Configuration
CUTOFF_DATE = '2024-06-25'     # Split historical/prediction
HORIZONS = {
    '3M': 90,    # 3 months
    '6M': 180,   # 6 months
    '12M': 365   # 12 months
}

# Feature Selection
VIF_THRESHOLD = 10.0             # Multicollinearity threshold
CORRELATION_THRESHOLD = 0.85     # Feature correlation limit

# Model Training
RANDOM_STATE = 42                # Reproducibility seed
TEST_SIZE = 0.30                 # Train/test split
```

---

## ✅ Validation

```bash
# Validate specific step output
python pipeline_validation.py --step 2

# Validate all steps
python pipeline_validation.py --all

# Quick file existence check
python pipeline_validation.py --step 10 --quick
```

---

## 🔬 Optional Analysis Tools

Not part of main pipeline but useful for research:

```bash
# Exploratory data analysis (16 analyses)
python analysis/exploratory/04_eda.py

# Logistic regression baseline
python analysis/diagnostics/06b_logistic_baseline.py

# Infant mortality analysis
python analysis/explore_infant_mortality.py
```

See `archived/README.md` for historical/experimental scripts.

---

## 📋 Requirements

### **Data**
- **Input**: `data/inputs/fault_merged_data.xlsx` (~1,210 fault records)
- **Optional**: `data/inputs/health_merged_data.xlsx` (healthy equipment)

### **Python Dependencies**
- pandas, numpy, scikit-learn
- xgboost, catboost
- lifelines (survival analysis)
- shap (explainability)
- matplotlib, seaborn

### **System**
- Python 3.8+
- 8GB RAM minimum
- ~5-10GB disk space for outputs

---

## 🚨 Before Production

**Read**: [CRITICAL_ISSUES_BEFORE_PIPELINE_EXECUTION.md](CRITICAL_ISSUES_BEFORE_PIPELINE_EXECUTION.md)

**Critical items**:
1. ✅ **Smart Feature Selection** - Architectural flaw needs resolution
2. ✅ **File Path Structure** - Config vs reality mismatch
3. ⚠️ **Turkish Localization** - Required for client delivery

---

## 📦 Phase 1 Enhancements (Completed)

✅ **Phase 1.1**: Equipment ID consistency (100% target-feature alignment)
✅ **Phase 1.2**: Leakage feature removal (chronic AUC: 1.0 → 0.75-0.88)
✅ **Phase 1.3**: Enhanced leakage detection (target indicator patterns)
✅ **Phase 1.4**: Mixed dataset training (789 → 5,567 equipment)
✅ **Phase 1.5**: Standardized imputation strategy (documented)

See [PHASE_1_COMPLETION_SUMMARY.md](PHASE_1_COMPLETION_SUMMARY.md) for details.

---

## 🏗️ Project Structure

```
PoF2/
├── 00-11_*.py              Main pipeline scripts (13 steps)
├── run_pipeline.py         Pipeline orchestrator
├── config.py               Configuration parameters
├── column_mapping.py       EN ↔ TR mappings
├── pipeline_validation.py  Data quality validation
├── smart_feature_selection.py  Feature selection engine
│
├── data/                   Input and intermediate data
│   ├── inputs/            Raw fault data
│   └── arsiv/             Archived data
│
├── predictions/           Model predictions (CSV)
├── results/               Risk assessments (CSV)
├── models/                Trained models (PKL)
├── outputs/               Visualizations (PNG)
├── logs/                  Pipeline execution logs
│
├── analysis/              Optional research tools
│   ├── exploratory/       EDA scripts
│   └── diagnostics/       Baseline comparisons
│
├── archived/              Non-production scripts
├── docs/                  Detailed documentation
└── utils/                 Utility functions
```

---

## 🆘 Troubleshooting

### **Pipeline fails at Step X**
1. Check log: `logs/run_TIMESTAMP/0X_script_name.log`
2. Validate previous step: `python pipeline_validation.py --step X-1`
3. See [PIPELINE_EXECUTION_ORDER.md](PIPELINE_EXECUTION_ORDER.md) troubleshooting section

### **Missing input files**
Run Step 0 to validate: `python 00_input_data_source_analysis.py`

### **Model performance issues**
- Check Step 8 (explainability) for feature importance
- Review Step 9 (calibration) curves
- Compare temporal PoF vs Cox survival predictions

---

## 📞 Support

- **Pipeline Documentation**: See [PIPELINE_EXECUTION_ORDER.md](PIPELINE_EXECUTION_ORDER.md)
- **Configuration Help**: See inline comments in `config.py`
- **Data Issues**: See [DATA_ANALYSIS_STRATEGY.md](DATA_ANALYSIS_STRATEGY.md)
- **Phase 1 Fixes**: See [PHASE_1_COMPLETION_SUMMARY.md](PHASE_1_COMPLETION_SUMMARY.md)

---

## 📜 License & Attribution

**Project**: PoF2 - Probability of Failure (Phase 2)
**Client**: Turkish EDAŞ (Electricity Distribution Companies)
**Domain**: Utility Asset Management, Predictive Maintenance
**Author**: Data Analytics Team
**Last Updated**: 2025-11-28

---

## 🎯 Next Steps

1. **Resolve Critical Issues** - See CRITICAL_ISSUES_BEFORE_PIPELINE_EXECUTION.md
2. **Run Pipeline** - `python run_pipeline.py`
3. **Validate Outputs** - `python pipeline_validation.py --all`
4. **Review Results** - Check `results/capex_priority_list.csv`
5. **Turkish Localization** - Add TR outputs before client delivery

**Status**: Production-ready after critical issues resolved ⚠️
