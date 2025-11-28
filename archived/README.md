# Archived Scripts

**Date Archived**: 2025-11-25

This folder contains scripts that are NOT part of the main production pipeline (`run_pipeline.py`) but may be useful for research, diagnostics, or future development.

---

## Archived Scripts

### 1. **07_walkforward_validation.py** (~180 lines)
**Purpose**: Walk-forward validation with expanding window

**Why Archived**:
- Not used in production pipeline
- Walk-forward validation is more time-consuming than standard k-fold CV
- Standard CV in 06_temporal_pof_model.py is sufficient for model evaluation

**When to Use**:
- If you need to validate temporal model stability over time
- For research on how model performance changes as training window expands
- Useful for academic papers or detailed model diagnostics

**Inputs**:
- `data/features_reduced.csv`
- `outputs/features_with_targets.csv`

**Outputs**:
- `results/walkforward_validation_results.csv`
- Validation performance metrics over time

---

### 2. **08_class_imbalance_analysis.py** (~150 lines)
**Purpose**: Analyze class imbalance and recommend sampling strategies

**Why Archived**:
- Not part of production pipeline
- Class imbalance is already addressed in main pipeline via `scale_pos_weight` (XGBoost) and `auto_class_weights` (CatBoost)
- SMOTE not recommended for this dataset (see analysis results)

**When to Use**:
- If you want to experiment with different class imbalance handling strategies
- For diagnostic analysis of class distribution across horizons
- To generate reports on failure rate distributions

**Inputs**:
- `data/features_reduced.csv`
- `outputs/features_with_targets.csv`

**Outputs**:
- Class distribution analysis
- Recommendations for sampling strategies

---

### 3. **09_train_with_smote.py** (~200 lines)
**Purpose**: Train models with SMOTE oversampling

**Why Archived**:
- Not used in production pipeline
- SMOTE often degrades performance on real-world imbalanced datasets
- Native class weighting (`scale_pos_weight`) performs better for this use case
- Creates synthetic samples that may not reflect real equipment behavior

**When to Use**:
- For comparison experiments: SMOTE vs class weights
- If you want to test if synthetic oversampling improves recall
- Academic research comparing sampling strategies

**Inputs**:
- `data/features_reduced.csv`
- `outputs/features_with_targets.csv`

**Outputs**:
- `models/smote_xgboost_*.pkl` (SMOTE-trained models)
- Performance comparison metrics

---

### 4. **logger.py** (~273 lines)
**Purpose**: Centralized logging module with structured logging helpers

**Why Archived**:
- **NEVER USED** - Fully implemented but not imported anywhere in the codebase
- Pipeline uses print statements and subprocess output capture instead
- Was designed for structured logging but remained unused

**When to Use**:
- If you want to add proper logging infrastructure to the pipeline
- For production deployments requiring log aggregation
- To replace print() statements with proper logging levels

**Functions Provided**:
- `get_logger(name)` - Get configured logger
- `log_script_start(logger, script_name, version)` - Log script start
- `log_script_end(logger, script_name)` - Log script completion
- `setup_logging(log_file, level)` - Configure logging

**Restoration**:
- If restoring, update all pipeline scripts to import and use logger
- Remove print() statements and replace with logger.info(), logger.warning(), etc.
- Update run_pipeline.py to aggregate log files instead of capturing stdout

---

### 5. **check_data_availability.py** (~100 lines) - **ARCHIVED 2025-11-28**
**Purpose**: Check if required input data files exist

**Why Archived**:
- **NEVER IMPORTED** - Not used anywhere in the codebase
- Functionality replaced by `00_input_data_source_analysis.py` (comprehensive validation)
- Redundant with newer, more complete validation script

**When to Use**:
- Simple file existence check before running pipeline
- Quick validation script for automation

---

### 6. **diagnose_data_issues.py** (~265 lines) - **ARCHIVED 2025-11-28**
**Purpose**: Diagnostic tool for identifying data quality issues

**Why Archived**:
- **NEVER IMPORTED** - Not used anywhere in the codebase
- Functionality overlaps with `01_data_profiling.py` and validation framework
- Was likely an early diagnostic tool replaced by more comprehensive scripts

**When to Use**:
- Debugging specific data quality issues
- One-off data diagnostics
- Historical reference for data issue patterns

---

### 7. **diagnostic_model_audit.py** (~218 lines) - **ARCHIVED 2025-11-28**
**Purpose**: Audit model training process and outputs

**Why Archived**:
- **NEVER IMPORTED** - Not used anywhere in the codebase
- Functionality covered by `pipeline_validation.py` and individual model scripts
- Redundant with current validation framework

**When to Use**:
- Detailed model output auditing
- Debugging model training issues
- Historical reference for validation patterns

---

## Related Scripts (Already in Analysis Folders)

These scripts are also optional/diagnostic but were already organized into subfolders:

### **analysis/exploratory/04_eda.py** (~275 lines)
- **Purpose**: 16 exploratory data analyses (distributions, correlations, temporal patterns)
- **Status**: Optional - run separately for research/understanding
- **Note**: Marked as OPTIONAL in pipeline documentation

### **analysis/diagnostics/06b_logistic_baseline.py** (~400 lines)
- **Purpose**: Train logistic regression baseline for comparison
- **Status**: Optional - baseline comparison only
- **Note**: Useful for validating that complex models (XGBoost) outperform simple baselines

### **analysis/explore_infant_mortality.py**
- **Purpose**: Analyze early-life failure patterns
- **Status**: Research/diagnostic script

---

## Usage Guidelines

### If you need to run an archived script:

1. **Run directly from archived folder**:
   ```bash
   python archived/07_walkforward_validation.py
   ```

2. **Or temporarily copy back to root**:
   ```bash
   cp archived/09_train_with_smote.py .
   python 09_train_with_smote.py
   rm 09_train_with_smote.py
   ```

3. **Check script dependencies**:
   - Most archived scripts require `data/features_reduced.csv`
   - Some require `outputs/features_with_targets.csv` (created by Step 5)
   - Run pipeline through Step 5 first if needed

---

## Restoration

If you decide a script should be part of the main pipeline:

1. Move it back to root: `git mv archived/SCRIPT.py .`
2. Add to `run_pipeline.py` PIPELINE_STEPS
3. Update `PIPELINE_EXECUTION_ORDER.md`
4. Test full pipeline integration

---

## Notes

- These scripts were removed from the main pipeline to reduce complexity and runtime
- They are fully functional and can be run independently
- Outputs are saved to `results/` and `outputs/` folders
- Check git history for detailed change log: `git log -- archived/SCRIPT.py`

---

**Last Updated**: 2025-11-25
