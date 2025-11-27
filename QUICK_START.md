# Quick Start Guide
**Version**: 4.0 with Phase 1 Enhancements
**Updated**: 2025-11-27

---

## 🚀 Run Everything (Recommended)

```bash
python run_pipeline.py
```

**What this does**:
1. ✅ Validates input data (Script 00 - NEW)
2. ✅ Profiles data quality
3. ✅ Transforms data with Phase 1 fixes
4. ✅ Engineers features
5. ✅ Selects features with leakage detection
6. ✅ Trains PoF model with mixed dataset
7. ✅ Trains classifier without leakage
8. ✅ Generates predictions and reports

**Time**: ~2-3 hours
**Output**: Logs in `logs/run_TIMESTAMP/`

---

## 📋 Just Validate Input Data

```bash
python 00_input_data_source_analysis.py
```

Run this FIRST to ensure your data is ready.

---

## 🔍 Step-by-Step (Manual)

```bash
# Step 0: Input validation
python 00_input_data_source_analysis.py

# Step 1: Data profiling
python 01_data_profiling.py

# Step 2a: Load healthy equipment (optional)
python 02a_healthy_equipment_loader.py

# Step 2: Transform data
python 02_data_transformation.py

# Step 3: Engineer features
python 03_feature_engineering.py

# Step 4: Select features
python 04_feature_selection.py

# Step 6: Train PoF model
python 06_temporal_pof_model.py

# Step 7: Train classifier
python 07_chronic_classifier.py
```

---

## 📂 Key Output Files

After running pipeline:

```
data/
├── features_engineered.csv       ← After feature engineering
├── features_reduced.csv          ← After feature selection
└── equipment_level_data.csv      ← After data transformation

predictions/
├── chronic_repeaters.csv         ← Classifier predictions
├── temporal_pof_predictions.csv  ← PoF model predictions
└── high_risk_equipment.csv       ← Ranked risk list

models/
├── temporal_pof_xgboost.pkl
├── temporal_pof_catboost.pkl
├── chronic_repeater_xgboost.pkl
└── chronic_repeater_catboost.pkl

logs/run_TIMESTAMP/
├── 00_input_data_source_analysis.log
├── 01_data_profiling.log
├── ... (one per step)
└── master_log.txt
```

---

## ✅ Phase 1 Validation Checklist

After pipeline runs, verify Phase 1 fixes:

### Phase 1.1: Equipment ID Consistency
- [ ] Step 5 (Equipment ID Audit) shows 100% match rate
- [ ] No warnings about missing Equipment IDs
- [ ] All 5,567 equipment have targets

### Phase 1.2: Leakage Removal
- [ ] Step 7 (Chronic Classifier) AUC is 0.75-0.88 (NOT 1.0)
- [ ] Feature importance shows realistic patterns
- [ ] Leakage features excluded from training

### Phase 1.3: Leakage Detection
- [ ] Step 4 (Feature Selection) detects leakage patterns
- [ ] Tekrarlayan_Arıza_* features removed
- [ ] AgeRatio_Recurrence_Interaction removed

### Phase 1.4: Mixed Dataset
- [ ] Step 6 (PoF Model) uses 5,567 equipment
- [ ] ~48% failed, ~52% healthy equipment
- [ ] Healthy equipment targets are all zeros

### Phase 1.5: Imputation Analysis
- [ ] Step 3 (Feature Engineering) prints missing value analysis
- [ ] Features with >50% missing identified
- [ ] Imputation strategy documented

---

## 🎯 Common Tasks

### Validate Data Only
```bash
python 00_input_data_source_analysis.py
```

### Train Only Models
```bash
python 04_feature_selection.py  # Get clean features
python 06_temporal_pof_model.py # PoF predictions
python 07_chronic_classifier.py # Chronic predictions
```

### Analyze Results
```bash
python analysis/exploratory/04_eda.py
```

### Check Logs
```bash
# Latest run
cat logs/run_latest/master_log.txt

# Specific step
cat logs/run_latest/02_data_transformation.log
```

---

## ⚠️ Troubleshooting

### Input File Not Found
- Check file exists: `data/combined_data_son.xlsx`
- Check path in `config.py`
- Run: `python 00_input_data_source_analysis.py`

### Pipeline Fails at Step 2
- Run Step 0 and 1 independently
- Check `logs/run_TIMESTAMP/02_data_transformation.log`
- Verify columns exist in input file

### Features Selected Too Aggressively
- Check Phase 1.3 leakage detection in Step 4 log
- Review which patterns removed which features
- Edit LEAKAGE_PATTERNS in `column_mapping.py` if needed

### Chronic Classifier AUC Still 1.0
- Verify Phase 1.2 changes applied
- Check leakage features excluded in Step 7
- Regenerate features_reduced.csv (run Step 4 again)

---

## 📊 Configuration

Edit `config.py` to customize:

```python
# Input file
INPUT_FILE = 'data/combined_data_son.xlsx'

# Cutoff date for temporal windows
CUTOFF_DATE = datetime(2024, 6, 25)

# Feature selection thresholds
VIF_THRESHOLD = 10.0
CORRELATION_THRESHOLD = 0.95

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3
```

---

## 📝 Next Steps After Pipeline

1. **Review Phase 1 Validation** ✅ (above checklist)
2. **Analyze Results** → Run `python analysis/exploratory/04_eda.py`
3. **Create Reports** → Check `reports/` folder
4. **Deploy Models** → Use pickled models in `models/` folder
5. **Monitor Predictions** → Track `predictions/` outputs

---

## 🆘 Need Help?

- **Data issues**: See `DATA_ANALYSIS_STRATEGY.md`
- **Phase 1 fixes**: See `PHASE_1_COMPLETION_SUMMARY.md`
- **Pipeline details**: See `PIPELINE_WORKFLOW_GUIDE.md`
- **Overall audit**: See `PHASE_1_AUDIT_REPORT.md`

---

**Status**: Ready to run!
**Command**: `python run_pipeline.py`
