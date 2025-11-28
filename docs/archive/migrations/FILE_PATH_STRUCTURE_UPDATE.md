# File Path Structure Update
**Date**: 2025-11-28
**Status**: ✅ COMPLETED - New directory structure implemented
**Commit**: 220e192
**Impact**: Cleaner project organization, centralized path management

---

## New Directory Structure

```
PoF2/
├── data/
│   ├── inputs/                          # INPUT FILES (read-only)
│   │   ├── fault_merged_data.xlsx       # All fault records
│   │   └── health_merged_data.xlsx      # Healthy equipment data
│   │
│   └── outputs/                         # INTERMEDIATE FILES (generated)
│       ├── equipment_level_data.csv     # Step 2 output
│       ├── features_engineered.csv      # Step 3 output
│       ├── features_reduced.csv         # Step 4 output
│       ├── health_equipment_prepared.csv # Step 2a output (optional)
│       ├── feature_documentation.csv    # Documentation
│       ├── feature_catalog.csv          # Feature catalog
│       └── high_risk_equipment.csv      # High risk equipment list
│
├── outputs/                             # FINAL OUTPUTS (reports, predictions)
│   └── (created by pipeline)
│
├── models/                              # TRAINED MODELS
│   └── (created by pipeline)
│
├── predictions/                         # PREDICTION OUTPUTS
│   └── (created by pipeline)
│
├── results/                             # RESULTS & REPORTS
│   └── (created by pipeline)
│
├── logs/                                # LOG FILES
│   └── (created by pipeline)
│
└── config.py                            # ⭐ CENTRALIZED PATH CONFIGURATION

```

---

## Config.py Path Definitions

All paths are now centralized in `config.py`:

```python
# Data directories
DATA_DIR = Path('data')
DATA_INPUTS_DIR = DATA_DIR / 'inputs'      # NEW: Input files
DATA_OUTPUTS_DIR = DATA_DIR / 'outputs'    # NEW: Intermediate files

# Input files (from data/inputs/ folder)
INPUT_FILE = DATA_INPUTS_DIR / 'fault_merged_data.xlsx'
HEALTHY_EQUIPMENT_FILE = DATA_INPUTS_DIR / 'health_merged_data.xlsx'

# Intermediate files (saved to data/outputs/ folder)
EQUIPMENT_LEVEL_FILE = DATA_OUTPUTS_DIR / 'equipment_level_data.csv'
FEATURES_ENGINEERED_FILE = DATA_OUTPUTS_DIR / 'features_engineered.csv'
FEATURES_REDUCED_FILE = DATA_OUTPUTS_DIR / 'features_reduced.csv'
FEATURES_WITH_TARGETS_FILE = DATA_OUTPUTS_DIR / 'features_with_targets.csv'
```

---

## Changes by Script

### Core Pipeline Scripts

**1. config.py** ✅
- Added `DATA_INPUTS_DIR` pointing to `data/inputs/`
- Added `DATA_OUTPUTS_DIR` pointing to `data/outputs/`
- Updated all INPUT file paths to use `DATA_INPUTS_DIR`
- Updated all intermediate/output paths to use `DATA_OUTPUTS_DIR`
- Single source of truth for all file locations

**2. 02a_healthy_equipment_loader.py** ✅
- Import `HEALTHY_EQUIPMENT_FILE` from config
- Import `DATA_OUTPUTS_DIR` from config
- Output file: `DATA_OUTPUTS_DIR / 'health_equipment_prepared.csv'`
- Removed hardcoded path definitions

**3. smart_feature_selection.py** ✅
- Updated `run_smart_selection()` function
- Default input: `FEATURES_ENGINEERED_FILE` from config
- Default output: `FEATURES_REDUCED_FILE` from config
- No hardcoded paths

### Audit & Validation Scripts

**4. 05_equipment_id_audit.py** ✅
- Import paths from config: `INPUT_FILE`, `FEATURES_REDUCED_FILE`, `EQUIPMENT_LEVEL_FILE`
- Removed hardcoded `'data/equipment_level_data.csv'`
- Removed hardcoded `'data/features_reduced.csv'`

**5. diagnostic_model_audit.py** ✅
- Import paths from config: `FEATURES_REDUCED_FILE`, `EQUIPMENT_LEVEL_FILE`
- Removed hardcoded paths
- Uses config paths for data loading

**6. pipeline_validation.py** ✅
- Import `DATA_OUTPUTS_DIR` from config
- Updated Step 2a validation to check `DATA_OUTPUTS_DIR / 'health_equipment_prepared.csv'`
- All validation paths now use config

---

## Input File Changes

### OLD Paths
```
INPUT_FILE = 'data/combined_data_son.xlsx'
HEALTHY_EQUIPMENT_FILE = 'data/healthy_equipment.xlsx'
```

### NEW Paths
```
INPUT_FILE = 'data/inputs/fault_merged_data.xlsx'
HEALTHY_EQUIPMENT_FILE = 'data/inputs/health_merged_data.xlsx'
```

**Why?**:
- Cleaner organization
- Input files in `inputs/` folder (read-only, version controlled separately)
- Easier to maintain multiple input versions
- Clear separation between inputs and generated files

---

## Intermediate Files Changes

### OLD Paths
```
EQUIPMENT_LEVEL_FILE = 'data/equipment_level_data.csv'
FEATURES_ENGINEERED_FILE = 'data/features_engineered.csv'
FEATURES_REDUCED_FILE = 'data/features_reduced.csv'
FEATURES_WITH_TARGETS_FILE = 'outputs/features_with_targets.csv'
FEATURE_DOCS_FILE = 'data/feature_documentation.csv'
FEATURE_CATALOG_FILE = 'data/feature_catalog.csv'
HIGH_RISK_FILE = 'data/high_risk_equipment.csv'
```

### NEW Paths
```
EQUIPMENT_LEVEL_FILE = 'data/outputs/equipment_level_data.csv'
FEATURES_ENGINEERED_FILE = 'data/outputs/features_engineered.csv'
FEATURES_REDUCED_FILE = 'data/outputs/features_reduced.csv'
FEATURES_WITH_TARGETS_FILE = 'data/outputs/features_with_targets.csv'
FEATURE_DOCS_FILE = 'data/outputs/feature_documentation.csv'
FEATURE_CATALOG_FILE = 'data/outputs/feature_catalog.csv'
HIGH_RISK_FILE = 'data/outputs/high_risk_equipment.csv'
```

**Why?**:
- All pipeline-generated intermediate files in one place
- Easy to clean up (delete `data/outputs/` to rerun pipeline)
- Separate from input data (which doesn't change)
- Clear distinction: `inputs/` = stable, `outputs/` = regenerated

---

## Directory Creation

The required directories are automatically created by the pipeline, but can also be manually created:

```bash
mkdir -p data/inputs
mkdir -p data/outputs
```

These folders already exist in the repository with:
- `data/inputs/` - Ready for your input files
- `data/outputs/` - Ready for generated intermediate files

---

## Benefits of This Structure

### ✅ Cleaner Organization
```
Before:
- 7 files in data/
- 3 files in outputs/
- Confusing: which is input vs generated?

After:
- 2 files in data/inputs/
- 7 files in data/outputs/
- Crystal clear separation
```

### ✅ Centralized Path Management
```
Before:
- Hardcoded paths in 6+ scripts
- Change input path? Update multiple files
- Error-prone maintenance

After:
- config.py is single source of truth
- Change path? Update config.py once
- All scripts automatically use new path
```

### ✅ Easier Cleanup
```
Before:
- Delete data/feature*.csv, data/equipment*.csv individually

After:
- Delete entire data/outputs/ folder
- Pipeline automatically recreates when needed
```

### ✅ Better Version Control
```
Before:
- Input files mixed with generated files

After:
- Only commit: data/inputs/, rest of code
- Never commit: data/outputs/ (always regenerated)
- .gitignore: data/outputs/
```

---

## Migration Steps (Already Completed)

### Step 1: Update config.py ✅
- Added `DATA_INPUTS_DIR` and `DATA_OUTPUTS_DIR`
- Updated all path definitions

### Step 2: Update Scripts ✅
- 02a_healthy_equipment_loader.py
- smart_feature_selection.py
- 05_equipment_id_audit.py
- diagnostic_model_audit.py
- pipeline_validation.py
- 00_input_data_source_analysis.py (already done in Phase 1.8)

### Step 3: Validate ✅
- All scripts compile successfully
- No syntax errors
- Ready to execute

### Step 4: Create Directories ✅
- `data/inputs/` created
- `data/outputs/` created

### Step 5: Commit & Push ✅
- All changes committed
- Pushed to remote branch

---

## Before Running Pipeline

### Prepare Input Files

Move your data files to the correct location:

```bash
# Place your fault data
data/inputs/fault_merged_data.xlsx

# Place your health equipment data (if available)
data/inputs/health_merged_data.xlsx
```

### Run Pipeline

```bash
python run_pipeline.py
```

The pipeline will:
1. ✓ Read from `data/inputs/`
2. ✓ Generate intermediate files in `data/outputs/`
3. ✓ Create final outputs in `outputs/`, `models/`, `predictions/`, `results/`
4. ✓ Write logs to `logs/`

---

## File Access Pattern

```
Step 00: Input Data Analysis
  ├─ Read: data/inputs/fault_merged_data.xlsx
  └─ Read: data/inputs/health_merged_data.xlsx (if exists)

Step 01: Data Profiling
  └─ Read: data/inputs/fault_merged_data.xlsx

Step 02: Data Transformation
  ├─ Read: data/inputs/fault_merged_data.xlsx
  ├─ Read: data/inputs/health_merged_data.xlsx (if exists)
  └─ Write: data/outputs/equipment_level_data.csv

Step 02a: Health Equipment Loader (Optional)
  ├─ Read: data/inputs/health_merged_data.xlsx
  └─ Write: data/outputs/health_equipment_prepared.csv

Step 03: Feature Engineering
  ├─ Read: data/outputs/equipment_level_data.csv
  └─ Write: data/outputs/features_engineered.csv

Step 04: Feature Selection
  ├─ Read: data/outputs/features_engineered.csv
  └─ Write: data/outputs/features_reduced.csv

Steps 05+: Model Training & Evaluation
  └─ Read: data/outputs/features_reduced.csv
```

---

## Git Considerations

### .gitignore Update (Recommended)

Add to `.gitignore`:

```
# Generated intermediate files
data/outputs/

# Pipeline outputs
outputs/
models/
predictions/
results/
logs/
```

This ensures:
- Input files committed (source of truth)
- Generated files not committed (always regenerated)
- Cleaner repository history

---

## Summary

✅ **Completed**: All file paths centralized in `config.py`
✅ **Completed**: New directory structure (`data/inputs/` and `data/outputs/`)
✅ **Completed**: All scripts updated to use config paths
✅ **Completed**: No hardcoded paths remaining in scripts
✅ **Completed**: All directories created and ready

**Status**: READY FOR DATA FILES & PIPELINE EXECUTION
