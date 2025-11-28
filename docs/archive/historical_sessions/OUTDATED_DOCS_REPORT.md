# Outdated Documentation Report
**Date**: 2025-11-25
**After**: Pipeline cleanup (archived 4 scripts, removed 9 config params)

---

## Summary

Found **3 documentation files** that are outdated after recent cleanup changes.

---

## 🔴 CRITICAL: Requires Immediate Update

### 1. **docs/PIPELINE_USAGE.md** - SEVERELY OUTDATED

**Issues**:
1. ❌ References **old script names** that don't exist:
   - `05b_remove_leaky_features.py` (merged into 05_feature_selection.py months ago)
   - `06_model_training.py` (should be multiple 06_*.py scripts)
   - `09_survival_analysis.py` (actual: 06_survival_model.py)

2. ❌ Describes **old pipeline structure** (9 steps vs current 10 steps):
   ```
   Current doc says:
   Step 4: 04_eda.py
   Step 6: 05b_remove_leaky_features.py  ← DOESN'T EXIST
   Step 7: 06_model_training.py          ← DOESN'T EXIST
   Step 8: 09_survival_analysis.py       ← WRONG NAME
   ```

3. ❌ References **logger.py functionality** (now archived):
   - Lines 49-67 describe logs/ directory structure
   - Mentions LOG_LEVEL, LOG_DIR (removed from config.py)
   - Says pipeline creates timestamped log directories (it doesn't)

4. ❌ References **old output files**:
   - `risk_assessment_3M.csv`, `risk_assessment_24M.csv` (3M excluded, 24M removed)
   - Missing actual outputs like `predictions/predictions_*.csv`

**Impact**: Users following this guide will encounter errors

**Recommendation**: ✅ **REWRITE COMPLETELY** based on current run_pipeline.py

**Correct Pipeline Structure** (from run_pipeline.py):
```
Step 1:  01_data_profiling.py
Step 2:  02_data_transformation.py
Step 3:  03_feature_engineering.py
Step 4:  05_feature_selection.py
Step 5:  10_equipment_id_audit.py
Step 6:  06_temporal_pof_model.py
Step 7:  06_chronic_classifier.py
Step 8:  07_explainability.py
Step 9:  08_calibration.py
Step 10: 06_survival_model.py
Step 11: 10_consequence_of_failure.py
```

---

## ⚠️  MEDIUM: Contains Stale Information

### 2. **OPTION_B_PROGRESS.md** - References Archived logger.py

**Issues**:
- ✅ Lists "Add Logging Infrastructure" (logger.py) as COMPLETED task
- ❌ logger.py is now ARCHIVED (never imported anywhere)
- Lines 8-20 describe how to use logger.py
- Misleading: suggests logging infrastructure is active

**Outdated Sections**:
```markdown
### **1. ✅ Add Logging Infrastructure**
**File:** `logger.py`
- Easy to use: `from logger import get_logger`

Example usage:
from logger import get_logger, log_script_start, log_script_end
logger = get_logger(__name__)
```

**Impact**: Minor - this is a progress report from previous session

**Recommendation**: ⚠️ **ADD NOTICE** at top of file:
```markdown
> **⚠️ NOTE (2025-11-25)**: logger.py was later archived (never used in production).
> Pipeline uses print() statements + subprocess output capture instead.
> See archived/README.md for details.
```

**Alternative**: Archive this entire file to `archived/` or `docs/archive/`

---

### 3. **PIPELINE_RUNNER_FIX.md** - Minor Inaccuracy

**Issues**:
- Says pipeline has "10 total steps" (line 26)
- Actually has **11 steps** now (including 10_equipment_id_audit.py)
- Otherwise accurate about the fixes applied

**Outdated Line**:
```markdown
**Fix:** ✅ Added all 3 missing steps (now 10 total steps)
```

**Impact**: Very minor - count is off by 1

**Recommendation**: ⚠️ **QUICK FIX** - Change "10 total steps" to "11 total steps"

**Alternative**: Archive to docs/archive/ (historical progress report)

---

## ✅ UP-TO-DATE: No Changes Needed

### 4. **PIPELINE_EXECUTION_ORDER.md** - ✅ ACCURATE
- No references to archived scripts
- Accurately describes current pipeline structure
- No mentions of removed config parameters

### 5. **PIPELINE_REVIEW_SUMMARY.md** - ✅ STILL RELEVANT
- From previous session analyzing different issues (MTBF, target creation)
- No references to archived scripts
- Historical analysis still valid

### 6. **Recently Created Docs** - ✅ CURRENT
- DUAL_MODELING_ANALYSIS.md (created today)
- UNUSED_CONFIG_ANALYSIS.md (created today)
- PIPELINE_CLEANUP_SUMMARY.md (created today)
- archived/README.md (created today)

---

## Recommendation Summary

| File | Priority | Action | Effort |
|------|----------|--------|--------|
| **docs/PIPELINE_USAGE.md** | 🔴 CRITICAL | Rewrite completely | 1-2h |
| **OPTION_B_PROGRESS.md** | ⚠️ MEDIUM | Add deprecation notice OR archive | 5m |
| **PIPELINE_RUNNER_FIX.md** | ⚠️ LOW | Fix step count OR archive | 2m |

---

## Proposed Actions

### Option 1: Update Outdated Docs (Recommended)

1. **Rewrite docs/PIPELINE_USAGE.md** (1-2h)
   - Base on current run_pipeline.py structure
   - Remove logger.py references
   - Update output file paths
   - Add reference to archived/ scripts

2. **Update OPTION_B_PROGRESS.md** (5m)
   - Add deprecation notice at top about logger.py
   - Keep as historical record

3. **Fix PIPELINE_RUNNER_FIX.md** (2m)
   - Change "10 total steps" → "11 total steps"

**Total Effort**: ~2 hours

---

### Option 2: Archive Old Progress Reports (Quick)

1. **Create docs/archive/ folder**
2. **Move historical progress reports**:
   - OPTION_B_PROGRESS.md → docs/archive/
   - PIPELINE_RUNNER_FIX.md → docs/archive/
   - PIPELINE_SIMPLIFICATION_COMPLETE.md → docs/archive/
   - Other historical progress reports

3. **Rewrite docs/PIPELINE_USAGE.md** (still required)

**Benefit**: Cleaner root directory, clear separation of current vs historical docs

**Total Effort**: ~2 hours

---

## Detailed Fix for docs/PIPELINE_USAGE.md

### Current Issues (Line References):

| Line | Issue | Fix Needed |
|------|-------|------------|
| 22-30 | Old script names | Replace with current pipeline |
| 35-45 | Wrong pipeline steps table | Update to 11 current steps |
| 49-67 | logger.py log structure | Remove/replace with actual output |
| 80-86 | Old output files | Update to current predictions/ structure |
| 91-116 | Old visualization paths | Verify current output structure |

### Required Updates:

1. **Pipeline Steps Table** - Replace lines 35-45:
   ```markdown
   | Step | Script | Purpose | Estimated Time |
   |------|--------|---------|----------------|
   | 1 | `01_data_profiling.py` | Load and profile raw fault data | ~30s |
   | 2 | `02_data_transformation.py` | Transform to equipment-level data | ~1min |
   | 3 | `03_feature_engineering.py` | Create failure prediction features | ~2min |
   | 4 | `05_feature_selection.py` | Feature selection + leakage removal | ~1min |
   | 5 | `10_equipment_id_audit.py` | Validate ID consolidation | ~30s |
   | 6 | `06_temporal_pof_model.py` | Train XGBoost/CatBoost temporal models | ~2min |
   | 7 | `06_chronic_classifier.py` | Train chronic repeater classifier | ~1min |
   | 8 | `07_explainability.py` | SHAP feature importance analysis | ~2min |
   | 9 | `08_calibration.py` | Calibrate probability predictions | ~3min |
   | 10 | `06_survival_model.py` | Cox proportional hazards model | ~2min |
   | 11 | `10_consequence_of_failure.py` | Risk assessment (PoF × CoF) | ~1min |
   ```

2. **Log Files Section** - Replace lines 49-67:
   ```markdown
   ## Pipeline Output

   When using `run_pipeline.py`, all console output is captured to:

   ```
   logs/
   └── pipeline_run_YYYYMMDD_HHMMSS.log  # Complete pipeline execution log
   ```

   The runner script also creates a summary report at the end.
   ```

3. **Output Files** - Replace lines 80-86:
   ```markdown
   ### Predictions (CSV Files)
   ```
   predictions/
   ├── predictions_3m.csv             # 3-month temporal PoF
   ├── predictions_6m.csv             # 6-month temporal PoF
   ├── predictions_12m.csv            # 12-month temporal PoF
   ├── chronic_repeaters.csv          # Chronic equipment classification
   ├── pof_multi_horizon_predictions.csv  # Survival model multi-horizon
   └── capex_priority_list.csv        # Top 100 equipment for CAPEX
   ```
   ```

4. **Add Optional Scripts Section**:
   ```markdown
   ## Optional Analysis Scripts

   Not part of main pipeline but useful for research/diagnostics:

   ### Exploratory Data Analysis
   ```bash
   python analysis/exploratory/04_eda.py  # 16 exploratory analyses (~5 min)
   ```

   ### Archived Scripts
   See `archived/README.md` for scripts not in production pipeline:
   - Walk-forward validation
   - Class imbalance analysis
   - SMOTE training experiments
   ```

---

## Next Steps

1. **Review this report**
2. **Choose**: Option 1 (update docs) or Option 2 (archive historical docs)
3. **Prioritize**: Start with docs/PIPELINE_USAGE.md (critical for users)
4. **Test**: Verify all file paths and script names after updates

---

**Last Updated**: 2025-11-25
