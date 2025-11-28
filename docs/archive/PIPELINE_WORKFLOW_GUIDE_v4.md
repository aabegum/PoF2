# Pipeline Workflow Guide
**Date**: 2025-11-27
**Version**: 4.0 (with Phase 1 enhancements)
**Status**: Ready for use

---

## Quick Start

### Standard Pipeline Run (Recommended)

```bash
# Run the complete pipeline with all Phase 1 fixes integrated
python run_pipeline.py
```

This automatically:
1. ✅ Validates input data (Script 00 - NEW)
2. ✅ Profiles data quality (Script 01)
3. ✅ Loads healthy equipment (Script 02a - optional)
4. ✅ Transforms data with Phase 1.1 fixes (Script 02)
5. ✅ Engineers features with Phase 1.5 analysis (Script 03)
6. ✅ Selects features with Phase 1.3 leakage detection (Script 04)
7. ✅ Trains PoF with Phase 1.4 mixed dataset (Script 06)
8. ✅ Trains classifier with Phase 1.2 fixes (Script 07)
9. ✅ Continues through remaining steps

---

## New Pipeline Structure (v4.0)

### Step 0: Input Data Source Analysis (NEW - Phase 1) ⭐

**Purpose**: Validate raw input data before any processing

**What It Does**:
- Confirms input file exists and is readable
- Checks all expected columns are present
- Validates data types and formats
- Detects encoding issues
- Analyzes temporal coverage
- Reports data quality metrics
- Provides recommendations

**When It Runs**:
- FIRST - automatically with `python run_pipeline.py`
- Can also run standalone: `python 00_input_data_source_analysis.py`

**Expected Output**:
```
INPUT DATA SOURCE ANALYSIS
==========================

STEP 1: FILE INFORMATION
✓ Input file: combined_data_son.xlsx
  Size: 245.5 MB

STEP 2-11: (Various analyses)
...

STEP 12: RECOMMENDATIONS
✅ DATA READY FOR PROCESSING
```

**If Issues Found**:
- Script will report specific problems
- Provides actionable recommendations
- Stops gracefully if critical issues exist
- Prevents wasted computation on bad data

---

## Complete Pipeline Flow

### Phase 1: Input Validation

```
Step 0: Input Data Source Analysis
├─ File validation
├─ Column inventory
├─ Data type checking
├─ Temporal coverage analysis
└─ Quality assessment → REPORT + RECOMMENDATIONS

        ↓ (If PASS)

Step 1: Data Profiling
├─ Equipment ID coverage
├─ Equipment type validation
├─ Age source validation
├─ Missing value analysis
└─ Ready for transformation
```

### Phase 2: Data Transformation (with Phase 1 Fixes)

```
Step 2a: Healthy Equipment Loader (OPTIONAL)
├─ Load healthy equipment data
└─ Prepare for mixed dataset training

        ↓

Step 2: Data Transformation (Phase 1.1 - ID Consistency)
├─ Transform fault-level to equipment-level
├─ Handle Equipment_ID consistency
├─ Merge healthy equipment (if available)
└─ Create equipment-level features

        ↓

Step 3: Feature Engineering (Phase 1.5 - Imputation Analysis)
├─ Create TIER 1-8 features
├─ Analyze missing values
├─ Document imputation strategy
└─ Generate feature_engineered.csv
```

### Phase 3: Feature Selection & Audit

```
Step 4: Feature Selection (Phase 1.3 - Leakage Detection)
├─ PHASE 1: Remove constant features
├─ PHASE 2: Detect leakage patterns (ENHANCED)
│  ├─ Temporal window patterns
│  ├─ Target-derived patterns
│  ├─ Target indicator patterns (NEW)
│  └─ Aggregation leakage
├─ PHASE 3: Remove correlation
├─ PHASE 4: VIF optimization
└─ Generate features_reduced.csv

        ↓

Step 5: Equipment ID Audit (OPTIONAL)
├─ Verify ID consolidation
├─ Check target-feature alignment (Phase 1.1 validation)
└─ Confirm 100% match rate
```

### Phase 4: Model Training (with Phase 1 Fixes)

```
Step 6: Temporal PoF Model (Phase 1.4 - Mixed Dataset)
├─ Train on 5,567 equipment (48% failed, 52% healthy)
├─ Create 3M/6M/12M targets
├─ Multi-horizon XGBoost + CatBoost
└─ Generate PoF predictions

        ↓

Step 7: Chronic Classifier (Phase 1.2 - Leakage Removal)
├─ Train only on failed equipment
├─ Exclude leakage features (Tekrarlayan_Arıza_*, AgeRatio_*)
├─ Identify failure-prone equipment
├─ Expected AUC: 0.75-0.88 (NOT 1.0)
└─ Generate chronic repeater predictions

        ↓

Steps 8-11: Final Analysis
├─ Model explainability (SHAP)
├─ Probability calibration
├─ Cox survival modeling
└─ Risk assessment & CAPEX priority
```

---

## Typical Execution Times

| Step | Component | Typical Time | Notes |
|------|-----------|--------------|-------|
| **0** | Input Analysis | 30-60 sec | Fast validation |
| **1** | Data Profiling | 1-2 min | Data quality assessment |
| **2a** | Healthy Loader | 2-3 min | Optional, if healthy data exists |
| **2** | Data Transform | 10-15 min | Large dataset processing |
| **3** | Feature Eng | 15-20 min | Complex calculations |
| **4** | Feature Sel | 10-15 min | VIF optimization |
| **5** | ID Audit | 2-3 min | Optional validation |
| **6** | PoF Model | 30-45 min | Multi-horizon training |
| **7** | Chronic Class | 10-15 min | Binary classification |
| **8-11** | Analysis/Calib | 20-30 min | Various analyses |
| **TOTAL** | **Full Pipeline** | **~2-3 hours** | Varies by data size |

---

## Monitoring Pipeline Execution

### Log Files

All output is saved automatically:

```
logs/run_TIMESTAMP/
├── 00_input_data_source_analysis.log
├── 01_data_profiling.log
├── 02_data_transformation.log
├── 03_feature_engineering.log
├── 04_feature_selection.log
├── ... (one per step)
└── master_log.txt (complete output)
```

### Console Output

Real-time feedback shows:
```
[STEP 0/13] Input Data Source Analysis
  → Validate raw input data structure and quality before processing
  → Running 00_input_data_source_analysis.py...

✓ COMPLETED in 45 seconds

[STEP 1/13] Data Profiling
  → Validate data quality and temporal coverage
  → Running 01_data_profiling.py...

✓ COMPLETED in 90 seconds

[STEP 2/13] Data Transformation
  ...
```

### Summary Report

At the end, shows:
- Total execution time
- Per-step duration
- Success/failure status
- Output files generated
- Recommendations

---

## Phase 1 Fixes Validation

After running the pipeline, validate Phase 1 improvements:

### Phase 1.1: Equipment ID Consistency

**Check**:
```bash
# In Step 5 (Equipment ID Audit) output, verify:
# - 100% Equipment ID match rate (was 62-64%)
# - All 5,567 equipment have targets
# - No "WARNING: Equipment IDs not found in feature data"
```

### Phase 1.2: Leakage Removal

**Check**:
```bash
# In Step 7 (Chronic Classifier) output, verify:
# - Chronic classifier AUC: 0.75-0.88 (was 1.0)
# - Leakage features excluded from training
# - Feature importance shows realistic patterns
```

### Phase 1.3: Leakage Detection

**Check**:
```bash
# In Step 4 (Feature Selection) output, verify:
# - "target_indicator" leakage patterns detected
# - Tekrarlayan_Arıza_* features removed
# - AgeRatio_Recurrence_Interaction removed
```

### Phase 1.4: Mixed Dataset

**Check**:
```bash
# In Step 6 (Temporal PoF Model) output, verify:
# - Training dataset: 5,567 equipment
# - Class balance: ~48% failed, ~52% healthy
# - Healthy equipment targets: 100% zeros (right-censored)
```

### Phase 1.5: Imputation Analysis

**Check**:
```bash
# In Step 3 (Feature Engineering) output, verify:
# - Missing value analysis printed
# - Features with >50% missing identified
# - Imputation strategy documented
```

---

## Standalone Script Execution

You can also run individual scripts without the full pipeline:

```bash
# Just validate input data
python 00_input_data_source_analysis.py

# Just profile data quality
python 01_data_profiling.py

# Just do feature engineering
python 03_feature_engineering.py

# Just train models
python 06_temporal_pof_model.py
python 07_chronic_classifier.py

# Just run EDA analysis
python analysis/exploratory/04_eda.py
```

---

## Troubleshooting

### Issue: Pipeline fails at Step 0

**Problem**: Input Data Source Analysis reports errors

**Solution**:
1. Check `logs/run_TIMESTAMP/00_input_data_source_analysis.log`
2. Verify input file exists at path in config.py
3. Confirm file format is Excel (.xlsx)
4. Check file size > 100MB
5. Verify no encoding issues with Turkish characters

### Issue: Pipeline fails at Step 1 or 2

**Problem**: Data Profiling or Transformation errors

**Solution**:
1. Run Step 0 standalone to debug
2. Check if columns match expectations (from Step 0 report)
3. Verify column names (case-sensitive)
4. Check data types match expected types
5. Look at sample data from Step 0 output

### Issue: Feature Selection removes too many features

**Problem**: Phase 1.3 leakage detection is aggressive

**Solution**:
1. Check which patterns are matching
2. Review detected leakage in Step 4 output
3. If legitimate features removed, update LEAKAGE_PATTERNS in column_mapping.py
4. Re-run feature selection

### Issue: Chronic Classifier AUC still 1.0

**Problem**: Phase 1.2 leakage removal didn't work

**Solution**:
1. Verify phase 1.2 changes were applied
2. Check that exclude_cols includes both leakage features
3. Regenerate features_reduced.csv (run Step 4)
4. Re-run Step 7 with fresh features

---

## Best Practices

### Before Running Pipeline

- ✅ Run Step 0 independently first
- ✅ Review Step 0 recommendations
- ✅ Check logs for any warnings
- ✅ Verify input file format and encoding
- ✅ Confirm sufficient disk space (5-10GB)

### During Pipeline Execution

- ✅ Monitor console output for status
- ✅ Check individual logs if step seems slow
- ✅ Don't interrupt execution (can leave partial data)
- ✅ Note execution times for future planning

### After Pipeline Completion

- ✅ Review summary report
- ✅ Check log files for warnings/errors
- ✅ Validate Phase 1 improvements
- ✅ Save important outputs and logs
- ✅ Document any issues found

---

## Advanced Usage

### Running with Different Input Files

Edit `config.py`:
```python
INPUT_FILE = 'data/your_other_file.xlsx'
```

Then run pipeline as normal:
```bash
python run_pipeline.py
```

### Running Specific Steps Only

```bash
# Run only data validation steps
python 00_input_data_source_analysis.py
python 01_data_profiling.py

# Run only feature engineering
python 03_feature_engineering.py

# Run only model training
python 06_temporal_pof_model.py
python 07_chronic_classifier.py
```

### Parallel Analysis

Run EDA in parallel with pipeline:
```bash
# Terminal 1: Run main pipeline
python run_pipeline.py

# Terminal 2: Run exploratory analysis (in parallel)
python analysis/exploratory/04_eda.py
```

---

## Integration with Phase 1 Fixes

The pipeline now automatically includes:

1. **Phase 1.1**: Equipment ID consistency handling in Step 2
2. **Phase 1.2**: Leakage feature exclusion in Step 7
3. **Phase 1.3**: Enhanced leakage detection in Step 4
4. **Phase 1.4**: Mixed dataset training in Step 6
5. **Phase 1.5**: Imputation analysis in Step 3

No additional configuration needed - just run:

```bash
python run_pipeline.py
```

---

## Summary

**New in v4.0**:
- ✅ Step 0 automatically validates input data first
- ✅ Early detection of data issues before processing
- ✅ Clear validation checklist and recommendations
- ✅ Phase 1 fixes integrated throughout
- ✅ Better error handling and logging
- ✅ Reproducible validation workflow

**Benefits**:
- 🎯 Catch problems early
- ⏱️ Save computation time
- 📊 Understand data baseline
- 🔍 Clear audit trail
- ✅ Confidence in results

---

**Status**: Ready to run pipeline with Phase 1 fixes integrated
**Next**: Execute `python run_pipeline.py` to validate everything
