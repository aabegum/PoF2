# Documentation Archive

This folder contains historical progress reports and documentation from previous pipeline development phases.

**Date Archived**: 2025-11-25

---

## Archived Documents

### Progress Reports (Chronological)

#### **OPTION_B_PROGRESS.md**
**Date**: ~Nov 20, 2025
**Topic**: Complete Pipeline Cleanup - Progress Report

**Contents**:
- Phase 1 cleanup tasks (logging infrastructure, config centralization, EDA documentation)
- Shows logger.py as "completed infrastructure"

**Why Archived**:
- logger.py was later archived (never actually imported/used)
- Pipeline uses print() + subprocess capture instead
- Historical record of cleanup attempt

---

#### **PIPELINE_RUNNER_FIX.md**
**Date**: Nov 20, 2025
**Topic**: run_pipeline.py Critical Fixes

**Contents**:
- Fixed broken 05b_remove_leaky_features.py reference
- Added missing production steps
- Removed EDA from main pipeline

**Why Archived**:
- Minor step count inaccuracy (says 10 steps, actually 11)
- Otherwise historical record of fixes applied

---

#### **PIPELINE_SIMPLIFICATION_COMPLETE.md**
**Date**: Previous session
**Topic**: Pipeline simplification completion report

**Why Archived**:
- Historical progress report
- May contain outdated script references

---

### Architecture Documents

#### **DUAL_MODEL_ARCHITECTURE.md**
**Date**: Previous session
**Topic**: Three-model architecture overview (Temporal, Chronic, Survival)

**Why Archived**:
- **Contains outdated script names**:
  - References `06_model_training.py` (doesn't exist)
  - References `06_chronic_repeater.py` (actual: `06_chronic_classifier.py`)
  - References `09_survival_analysis.py` (actual: `06_survival_model.py`)
- Architecture concept is still valid, but implementation details are wrong

**Replacement**:
- See current `DUAL_MODELING_ANALYSIS.md` for accurate analysis
- See `PIPELINE_EXECUTION_ORDER.md` for correct script names

---

## Still Current (Not Archived)

These documents remain in the root directory as they're up-to-date:

- **PIPELINE_EXECUTION_ORDER.md** - Main pipeline documentation (accurate)
- **DUAL_MODELING_ANALYSIS.md** - Current dual model analysis (2025-11-25)
- **PIPELINE_CLEANUP_SUMMARY.md** - Cleanup summary (2025-11-25)
- **UNUSED_CONFIG_ANALYSIS.md** - Config parameter analysis (2025-11-25)
- **OUTDATED_DOCS_REPORT.md** - Documentation audit (2025-11-25)
- **COLUMN_NAMING_STANDARD.md** - Naming conventions (still relevant)
- **FEATURE_OPTIMIZATION_PLAN.md** - Feature optimization plan
- **PIPELINE_REVIEW_SUMMARY.md** - MTBF & target analysis (historical but valid)

---

## Usage

If you need to reference historical decisions or progress:

1. **Check git history** for detailed change timeline:
   ```bash
   git log --follow docs/archive/FILENAME.md
   ```

2. **Compare with current docs** to see what changed:
   ```bash
   diff docs/archive/DUAL_MODEL_ARCHITECTURE.md DUAL_MODELING_ANALYSIS.md
   ```

3. **Restore if needed** (rarely necessary):
   ```bash
   git mv docs/archive/FILENAME.md .
   ```

---

**Last Updated**: 2025-11-25
