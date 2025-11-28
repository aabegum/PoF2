# Unused Configuration Parameters Analysis
**Date**: 2025-11-25

## Summary

Found **14 unused configuration parameters** in `config.py` (348 lines total).

---

## Analysis by Category

### 1. Logging Configuration (7 parameters) - ❌ REMOVE
**Status**: All unused - logger.py archived and never imported

```python
LOG_LEVEL = 'INFO'                    # Line 227
LOG_DIR = Path('logs')                # Line 230
LOG_FILE = LOG_DIR / 'pipeline.log'  # Line 231
LOG_FORMAT = '...'                    # Line 234
LOG_DATE_FORMAT = '...'               # Line 235
CONSOLE_LOG_LEVEL = 'INFO'            # Line 238
```

**Reason for Removal**:
- logger.py module (273 lines) was archived - never imported anywhere
- Pipeline uses print() statements + subprocess output capture instead
- All 7 logging parameters are orphaned

**Action**: ✅ **SAFE TO REMOVE** (30 lines including comments)

---

### 2. Visualization Configuration (3 parameters) - ⚠️  KEEP OR REMOVE

```python
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Line 245
FIGURE_SIZE = (12, 6)                 # Line 248
FIGURE_DPI = 300                      # Line 251
```

**Current Usage**:
- NOT imported anywhere
- Scripts use matplotlib defaults or define locally:
  - 07_explainability.py: `plt.figure(figsize=(16, 10))`
  - 06_temporal_pof_model.py: `plt.savefig(..., dpi=300)`
  - 08_calibration.py: defines own figure sizes

**Options**:
1. **Remove** - Scripts define their own plot parameters already
2. **Keep** - Useful if standardizing all plots later
3. **Fix** - Import and use these constants consistently

**Recommendation**: ⚠️  **KEEP FOR NOW** - Low overhead (3 lines), useful for future standardization

---

### 3. Feature Engineering Flags (2 parameters) - ✅ PARTIALLY USED

```python
USE_CLASS_WEIGHTS = True                       # Line 92
USE_FIRST_WORKORDER_FALLBACK = True           # Line 99
```

**Analysis**:

#### `USE_CLASS_WEIGHTS` - ❌ REMOVE
- NOT imported or used anywhere
- Class weighting is hardcoded in model training:
  - XGBOOST_PARAMS: `'scale_pos_weight': 1.0` (calculated dynamically)
  - CATBOOST_PARAMS: `'auto_class_weights': 'Balanced'` (always enabled)
- Flag is misleading - suggests it's optional but it's always active

**Action**: ✅ **REMOVE** - Hardcoded behavior, flag adds no value

#### `USE_FIRST_WORKORDER_FALLBACK` - ⚠️  KEEP
- NOT currently imported
- Related to age calculation strategy in 03_feature_engineering.py
- Script currently hardcodes the fallback logic (lines ~520-550)
- This flag was intended to control that behavior

**Options**:
1. **Remove** - Fallback is hardcoded anyway
2. **Keep + Use** - Import in 03_feature_engineering.py and honor the flag
3. **Keep** - Low overhead, documents intent

**Recommendation**: ⚠️  **KEEP** - Documents design intent (1 line), potential future use

---

### 4. Feature Selection Threshold (1 parameter) - ❌ REMOVE

```python
IMPORTANCE_THRESHOLD = 0.001  # Keep features contributing > 0.1%  (Line 118)
```

**Analysis**:
- NOT imported anywhere
- Feature importance filtering is NOT used in pipeline
- Feature selection uses VIF + correlation only (smart_feature_selection.py)
- PROTECTED_FEATURES list determines what to keep (not importance threshold)

**Action**: ✅ **REMOVE** - Unused, misleading (1 line)

---

### 5. Output File Paths (2 parameters) - ✅ KEEP (RECENTLY ADDED)

```python
FEATURES_WITH_TARGETS_FILE = OUTPUT_DIR / 'features_with_targets.csv'  # Line 71
HIGH_RISK_FILE = DATA_DIR / 'high_risk_equipment.csv'                 # Line 76
```

**Analysis**:

#### `FEATURES_WITH_TARGETS_FILE` - ✅ KEEP
- **RECENTLY ADDED** in Phase 1 cleanup (2025-11-25)
- Intended to centralize file path for features_with_targets.csv
- Currently NOT imported but should be used in:
  - 06_temporal_pof_model.py (line 334) - saves to OUTPUT_DIR / 'features_with_targets.csv'
  - Multiple scripts reference this file (needs centralization)

**Action**: ✅ **KEEP + TODO** - Import and use in 06_temporal_pof_model.py

#### `HIGH_RISK_FILE` - ⚠️  CHECK USAGE
- Path for high-risk equipment output
- Need to verify if 06_temporal_pof_model.py or other scripts save high-risk lists

**Action**: ⚠️  **INVESTIGATE** - Check if high_risk output is still generated

---

## Recommendations Summary

| Parameter | Lines | Action | Reason |
|-----------|-------|--------|--------|
| **Logging Config (7 params)** | 30 | ✅ **REMOVE** | logger.py archived, never used |
| **USE_CLASS_WEIGHTS** | 1 | ✅ **REMOVE** | Hardcoded behavior, misleading flag |
| **IMPORTANCE_THRESHOLD** | 1 | ✅ **REMOVE** | Not used in feature selection |
| **USE_FIRST_WORKORDER_FALLBACK** | 1 | ⚠️  **KEEP** | Documents intent, low overhead |
| **FEATURES_WITH_TARGETS_FILE** | 1 | ✅ **KEEP** | Recently added, needs integration |
| **HIGH_RISK_FILE** | 1 | ⚠️  **CHECK** | Verify if still used |
| **Plot Config (3 params)** | 3 | ⚠️  **KEEP** | Future standardization |

**Total Savings**: ~32 lines (9% of config.py)

---

## Detailed Removal Plan

### Phase 1: Safe Removals (32 lines) ⭐

Remove these parameters - no risk:

```python
# Lines 226-238: LOGGING CONFIGURATION (DELETE SECTION)
# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = 'INFO'

# Log file location
LOG_DIR = Path('logs')
LOG_FILE = LOG_DIR / 'pipeline.log'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Console logging
CONSOLE_LOG_LEVEL = 'INFO'

# Line 92: USE_CLASS_WEIGHTS (DELETE)
USE_CLASS_WEIGHTS = True

# Line 118: IMPORTANCE_THRESHOLD (DELETE)
IMPORTANCE_THRESHOLD = 0.001  # Keep features contributing > 0.1%
```

### Phase 2: Update docstrings

Remove references to removed parameters from:
- run_pipeline.py docstring (line 7-12) - mentions logger.py usage

### Phase 3: Optional - Standardize plot parameters

If keeping visualization config, update all plotting scripts to import and use:
- `PLOT_STYLE`
- `FIGURE_SIZE`
- `FIGURE_DPI`

Otherwise remove these 3 lines as well.

---

## Implementation Steps

1. **Backup**: Ensure git commit before changes
2. **Remove logging config**: Delete lines 226-238 from config.py
3. **Remove unused flags**: Delete USE_CLASS_WEIGHTS (line 92) and IMPORTANCE_THRESHOLD (line 118)
4. **Update run_pipeline.py**: Remove logger.py references from docstring (lines 7-12)
5. **Test**: Run pipeline to ensure no import errors
6. **Commit**: "Remove 9 unused config parameters (logger.py orphaned)"

---

## Notes

- Config file will shrink from 348 to ~316 lines (9% reduction)
- No functional impact - removed parameters were never used
- Improves maintainability by removing dead code
- Reduces confusion for future developers

---

**Last Updated**: 2025-11-25
