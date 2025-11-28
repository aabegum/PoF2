# Voltage Feature Multicollinearity Resolution
**Date**: 2025-11-28
**Status**: CRITICAL - Blocking Smart Feature Selection
**Impact**: VIF=∞ multicollinearity preventing feature selection from running

---

## Executive Summary

Two voltage features exist with **identical values** causing perfect multicollinearity (VIF=∞):
- **`component_voltage`** - Raw equipment voltage from source data (PROTECTED_FEATURES_EN line 465)
- **`voltage_level`** - Created as copy of component_voltage in feature engineering (03_feature_engineering.py line 587)

**Result**: Both map to Turkish name 'Gerilim_Seviyesi', VIF=∞, perfectly correlated

**Solution**: Remove `voltage_level` (redundant), keep `component_voltage` (original source)

---

## Problem Analysis

### Where Voltage Features Exist

#### 1. **component_voltage** (Raw Voltage)
```python
# Source: Raw input data
# Location: In PROTECTED_FEATURES_EN (line 465 of column_mapping.py)
# Values: [34500.0, 15800.0, 400.0, 0.4] (in volts)
# Meaning: Equipment's operating voltage level specification
# Status: PROTECTED - cannot be removed by smart feature selection
```

#### 2. **voltage_level** (Redundant Copy)
```python
# Source: Created in 03_feature_engineering.py (line 587)
# Creation: df['voltage_level'] = df['component_voltage']
# Values: [34500.0, 15800.0, 400.0, 0.4] (identical)
# Meaning: Same as component_voltage
# Status: NOT in PROTECTED_FEATURES_EN, but leaks through via Turkish mapping
```

#### 3. **Voltage_Class** (Derived Category)
```python
# Source: Derived classification in 03_feature_engineering.py (lines 600-621)
# Values: ['AG', 'OG', 'YG']
# Meaning: Categorical voltage level (Low/Medium/High)
# Status: PROTECTED - valuable categorical feature
# NOTE: This is INDEPENDENT of raw voltage values
```

### Turkish Name Mapping Issue

**column_mapping.py lines 31-33:**
```python
'component_voltage': 'Gerilim_Seviyesi',
'Voltage_Class': 'Gerilim_Sınıfı',
'voltage_level': 'Gerilim_Seviyesi',  # ⚠️ DUPLICATE - maps to same name!
```

**column_mapping.py lines 165-166 (PROTECTED_FEATURES_TR):**
```python
'Gerilim_Seviyesi',     # Maps to BOTH component_voltage AND voltage_level
'Gerilim_Sınıfı',
```

**Problem**: When smart_feature_selection checks `is_protected_feature('voltage_level')`:
- English version: NOT protected (not in PROTECTED_FEATURES_EN)
- Turkish version: Protected as 'Gerilim_Seviyesi'
- Result: INCONSISTENT protection status

---

## Multicollinearity Evidence

### Identical Values
```
Equipment 1: component_voltage = 34500.0  |  voltage_level = 34500.0 ✓ SAME
Equipment 2: component_voltage = 15800.0  |  voltage_level = 15800.0 ✓ SAME
Equipment 3: component_voltage = 400.0    |  voltage_level = 400.0    ✓ SAME
Equipment 4: component_voltage = 0.4      |  voltage_level = 0.4      ✓ SAME
```

### Statistical Evidence
- **Correlation**: 1.0 (perfect)
- **VIF**: ∞ (infinite multicollinearity)
- **Redundancy**: 100% (voltage_level is exact duplicate)
- **Information Loss**: None if one is removed

### Impact on Models
```
Without removal:
  - Regression coefficients unstable
  - Small changes in data → huge coefficient changes
  - Feature importance unreliable
  - Predictions less robust

With removal:
  - Clean, mathematically sound features
  - Stable coefficients
  - Reliable feature importance
  - Robust predictions
```

---

## Resolution Decision Matrix

### Option 1: REMOVE voltage_level (RECOMMENDED ✅)

**Rationale**:
- `voltage_level` is a created copy, not original source
- `component_voltage` is the raw equipment specification (more fundamental)
- Losing nothing by removing the copy
- Keeps protected feature (component_voltage) that might have domain importance
- Simplifies feature set

**Action**:
```python
# In 03_feature_engineering.py - CHANGE THIS:
df['voltage_level'] = df['component_voltage']

# TO THIS:
# REMOVED: voltage_level is redundant copy of component_voltage
# Keeping: component_voltage (original source)
# Reason: Both have VIF=∞, one must be removed
```

**Impact**:
- ✅ Removes multicollinearity
- ✅ No information loss (identical values)
- ✅ Keeps original component_voltage
- ✅ Voltage classification (Voltage_Class) still created independently

---

### Option 2: REMOVE component_voltage (NOT RECOMMENDED ❌)

**Issues**:
- Removes original equipment specification data
- component_voltage is protected (has domain importance)
- Loses direct voltage representation
- voltage_level is derived copy

**Not chosen because**: Removes more fundamental feature

---

### Option 3: REMOVE BOTH (NOT RECOMMENDED ❌)

**Issues**:
- Loses voltage information entirely
- Voltage_Class created from voltage, but categorical loss of detail
- Equipment characteristics incomplete

**Not chosen because**: Loses valuable domain information

---

## Implementation

### Step 1: Remove voltage_level Creation in Feature Engineering

**File**: `03_feature_engineering.py`

**Current code (lines 585-597)**:
```python
if 'component_voltage' in df.columns:
    # Rename for clarity
    df['voltage_level'] = df['component_voltage']  # ❌ REMOVE THIS LINE
```

**New code**:
```python
if 'component_voltage' in df.columns:
    # PHASE 1.6 FIX: Removed voltage_level (redundant copy of component_voltage)
    # Both had VIF=∞ (perfect multicollinearity)
    # Keeping: component_voltage (original equipment specification)
    # Removed: voltage_level (created as copy, no unique information)

    voltage_level = df['component_voltage']  # For local classification use only
```

### Step 2: Fix Turkish Name Mapping

**File**: `column_mapping.py` line 33

**Current**:
```python
'voltage_level': 'Gerilim_Seviyesi',  # Duplicate - maps to same
```

**New**:
```python
# PHASE 1.6 FIX: Removed voltage_level mapping
# voltage_level was redundant copy of component_voltage
# Keeping only: component_voltage → 'Gerilim_Seviyesi'
```

### Step 3: Update smart_feature_selection Logic

**File**: `smart_feature_selection.py`

**Impact**: Once voltage_level is removed from features_engineered.csv, it won't appear in selection pipeline

---

## Voltage Feature Final State

### After Implementation

| Feature | Role | Status | VIF | Comment |
|---------|------|--------|-----|---------|
| **component_voltage** | Raw equipment voltage | ✅ KEEP | Normal | Original spec; protected domain feature |
| **voltage_level** | (Copy of component_voltage) | ❌ REMOVE | ∞ | Redundant; no unique information |
| **Voltage_Class** | Categorical voltage tier | ✅ KEEP | Normal | Derived independently; valuable for classification |
| **Is_MV, Is_LV, Is_HV** | Voltage tier flags | ✅ KEEP | Normal | Created independently; useful for modeling |

---

## Why This Matters for Hybrid Staged Selection

With voltage_level removed:

1. **Stage 1 (Statistical Rules)**:
   - ✅ component_voltage has normal VIF (not infinite)
   - ✅ No multicollinearity to fight
   - ✅ Algorithm can apply rules consistently
   - ✅ PROTECTED_FEATURES not needed as override

2. **Stage 2 (Domain Review)**:
   - ✅ Smart selection works without exceptions
   - ✅ Features removed for valid statistical reasons
   - ✅ Domain experts review clean feature set
   - ✅ No circular "protect because protected" logic

3. **Stage 3 (Audit Trail)**:
   - ✅ Clear decision trail
   - ✅ No mathematical fighting
   - ✅ Reproducible results

---

## Validation Checklist

Before pipeline execution:

- [ ] voltage_level line removed from 03_feature_engineering.py
- [ ] voltage_level mapping removed from column_mapping.py
- [ ] features_engineered.csv does NOT contain voltage_level column
- [ ] features_reduced.csv successfully created with reduced features
- [ ] component_voltage present in both feature sets
- [ ] Voltage_Class present in both feature sets
- [ ] VIF analysis shows component_voltage has normal VIF (not ∞)
- [ ] Smart feature selection completes without "fighting" high-VIF features
- [ ] No leakage or multicollinearity issues detected

---

## Testing Strategy

### Test 1: Feature Creation
```bash
# Run: python 03_feature_engineering.py
# Expected:
#   ✓ features_engineered.csv created
#   ✓ voltage_level NOT in columns
#   ✓ component_voltage in columns
#   ✓ Voltage_Class in columns
```

### Test 2: Feature Selection
```bash
# Run: python 04_smart_feature_selection.py
# Expected:
#   ✓ features_reduced.csv created
#   ✓ component_voltage KEPT (not removed)
#   ✓ No multicollinearity fighting
#   ✓ Normal VIF for component_voltage
```

### Test 3: Full Pipeline
```bash
# Run: python run_pipeline.py
# Expected:
#   ✓ All steps complete without VIF warnings
#   ✓ Model training successful
#   ✓ No mathematical instability warnings
```

---

## Summary

**What we're doing**: Removing voltage_level (redundant copy) to eliminate VIF=∞ multicollinearity

**What we're keeping**: component_voltage (original specification) with normal VIF

**Why**: Enables proper smart feature selection without PROTECTED_FEATURES override logic

**Outcome**: Clean, mathematically sound feature set ready for Hybrid Staged Selection

---

**Status**: Ready to implement
**Timeline**: ~15 minutes to implement + test
**Priority**: CRITICAL - Blocking Hybrid Staged Selection implementation
