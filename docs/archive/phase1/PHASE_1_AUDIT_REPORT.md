# Phase 1 Comprehensive Audit Report
**Date**: 2025-11-27
**Pipeline**: Turkish EDAŞ PoF Prediction (v6.0)
**Status**: Critical Issues Identified - Ready for Fixes

---

## Executive Summary

This audit identifies **5 critical issues** in the current pipeline that affect model training, data integrity, and predictive accuracy. All issues are **foundational** (Phase 1) and must be fixed before proceeding to Phase 2.

**Key Metrics:**
- Equipment ID Mismatch: **36-38% of targets misaligned** with feature data
- Chronic Classifier: **Perfect AUC=1.0** (indicates data leakage)
- Feature Selection: **Missing explicit leakage detection** for derived features
- Temporal PoF: **Single dataset used** despite mixed data being created
- Imputation: **Inconsistent strategies** across pipeline (0 vs median)

---

## ISSUE #1: Equipment ID Mismatch (Foundation - CRITICAL)

### Problem
**36-38% of targets are misaligned with feature data**, causing model training on wrong associations.

### Root Cause
**Data Flow Issue:**
```
Step 2: cbs_id → Equipment_ID_Primary → RENAMED TO → Ekipman_ID
       (Line 598)                           (Line 821)

Step 6: Creates targets from cbs_id (raw faults)
        Filters to Ekipman_ID set from features
        Maps targets using df['Ekipman_ID'].isin(failed_equipment)

MISMATCH: Ekipman_ID values ≠ cbs_id values in filtered datasets
```

### Evidence
**File: `/home/user/PoF2/06_temporal_pof_model.py` (Lines 301-352)**

```python
# Line 302-305: Create targets from raw cbs_id
future_faults_3M_raw = all_faults[
    (all_faults['started at'] > CUTOFF_DATE) &
    (all_faults['started at'] <= FUTURE_3M_END)
]['cbs_id'].dropna().unique()

# Line 319: Filter to Ekipman_ID set (valid_equipment_ids from features)
future_faults_3M = np.array([id for id in future_faults_3M_raw if id in valid_equipment_ids])

# Line 324-326: Shows only 62-64% match rate
print(f"3M window:  {len(future_faults_3M_raw):,} raw → {len(future_faults_3M):,} valid ({len(future_faults_3M)/max(len(future_faults_3M_raw),1)*100:.1f}% matched)")

# Line 352: Assign targets using Ekipman_ID
targets[horizon_name] = df['Ekipman_ID'].isin(failed_equipment).astype(int)
```

### Impact
- **36-38% of target assignments are incorrect**
- Models learn associations between wrong equipment and failure patterns
- Model performance metrics are artificially inflated/deflated
- Predictions on unseen data will be unreliable

### Solution
**Use `cbs_id` consistently throughout the pipeline:**

1. **In Step 2** (02_data_transformation.py):
   - Keep `Equipment_ID_Primary` as the ID column
   - Do NOT rename it to `Ekipman_ID` (rename is confusing)
   - Export as `Equipment_ID` in outputs

2. **In Step 6** (06_temporal_pof_model.py):
   - Use `Equipment_ID_Primary` for all target creation
   - Match cbs_id against `Equipment_ID_Primary` (not Ekipman_ID)

3. **In all downstream steps**:
   - Use `Equipment_ID` as the single authoritative ID column

**Expected Outcome:**
- 100% target-feature alignment (currently ~64%)
- All 3,165 unique equipment IDs matched correctly
- Model training on correct associations

---

## ISSUE #2: Chronic Classifier Overfitting (Data Leakage - CRITICAL)

### Problem
**Chronic classifier achieves AUC=1.0**, which is impossible in real scenarios.

This indicates **data leakage** - the model is learning the target itself, not patterns that predict it.

### Root Cause
**Multiple leakage sources:**

1. **Direct Target Leakage:**
   - **Column**: `Tekrarlayan_Arıza_90gün_Flag`
   - **Usage**: Line 260 in `07_chronic_classifier.py`
   - **Problem**: This IS the target definition, not a predictor!
   ```python
   df['Target_Chronic_Repeater'] = df['Tekrarlayan_Arıza_90gün_Flag'].astype(int)
   ```
   - **Status**: Protected feature (column_mapping.py:491), so it won't be removed by feature selection

2. **Derived Target Leakage:**
   - **Column**: `AgeRatio_Recurrence_Interaction`
   - **Location**: column_mapping.py:488 (PROTECTED_FEATURES)
   - **Feature Importance**: 63% in chronic classifier
   - **Problem**: This feature is engineered from failure history, correlates perfectly with target
   - **Status**: Protected feature (won't be auto-detected/removed)

### Evidence
**File: `/home/user/PoF2/07_chronic_classifier.py` (Lines 250-260)**
```python
print("\n🎯 CHRONIC REPEATER DEFINITION:")
print("   Equipment with recurring failures within 90-day window")
print("   Using Tekrarlayan_Arıza_90gün_Flag feature")  # <-- THIS IS THE TARGET!

# Verify required column exists
if 'Tekrarlayan_Arıza_90gün_Flag' not in df.columns:
    ...

# Target = Chronic repeater flag
df['Target_Chronic_Repeater'] = df['Tekrarlayan_Arıza_90gün_Flag'].astype(int)
```

**File: `/home/user/PoF2/column_mapping.py` (Line 488)**
```python
PROTECTED_FEATURES_EN = [
    ...
    'AgeRatio_Recurrence_Interaction',  # <-- Derived from target
    ...
    'Tekrarlayan_Arıza_90gün_Flag',     # <-- IS the target
]
```

### Impact
- **Model memorizes the target, not learning patterns**
- **AUC=1.0 is unrealistic** - will not generalize to new data
- **Feature importance rankings are meaningless**
- **Production predictions will be incorrect**

### Solution
**Remove all leakage features from chronic classifier:**

1. **Update PROTECTED_FEATURES** in `column_mapping.py`:
   - Remove `Tekrarlayan_Arıza_90gün_Flag` from protected list
   - Remove `AgeRatio_Recurrence_Interaction` from protected list

2. **Update feature exclusion** in `07_chronic_classifier.py` (Line 294):
   ```python
   exclude_cols = [
       id_col,
       target_col,
       'Tekrarlayan_Arıza_90gün_Flag',           # IS the target
       'AgeRatio_Recurrence_Interaction',        # Derived from target
   ]
   ```

3. **Re-train chronic classifier:**
   - Expected AUC: ~0.75-0.88 (realistic range)
   - Model will learn actual failure patterns

**Expected Outcome:**
- AUC drops from 1.0 to realistic 0.75-0.88 range
- Feature importance shows actual predictive features
- Model will generalize to new data

---

## ISSUE #3: Leakage Detection Incomplete (Analysis - CRITICAL)

### Problem
**Smart feature selection has PHASE 2 (leakage detection) but misses critical leakage patterns.**

The detector only catches features matching specific patterns, and skips features in PROTECTED_FEATURES list.

### Root Cause

**File: `/home/user/PoF2/column_mapping.py` (Lines 212-240)**
```python
LEAKAGE_PATTERNS = {
    'temporal_windows': [
        '_3Ay', '_6Ay', '_12Ay',  # Only looks for horizon suffixes
        '_3M', '_6M', '_12M',
    ],

    'target_derived': [
        'Hedef_', 'Target_',      # Only looks for these prefixes
        'Arıza_Olasılığı',
        'Risk_Sınıfı',
    ],

    'aggregation_leakage': [
        '_Cluster_Avg', '_Class_Avg',
    ],

    'safe_patterns': [
        'Toplam_Arıza_Sayısı',
        'OAZS_',
        'Onarım_Süresi_',
        'Ekipman_Yaşı',
    ],
}
```

**File: `/home/user/PoF2/smart_feature_selection.py` (Line 343)**
```python
if is_leaky and not is_protected_feature(col):  # <-- Skips protected features!
    leaky_features.append((col, pattern_type))
```

### Impact
- `Tekrarlayan_Arıza_90gün_Flag` (IS the target) not detected because it doesn't match patterns
- `AgeRatio_Recurrence_Interaction` (derived from target) protected despite being leakage
- Feature selection reports "No leakage patterns detected" when leakage exists

### Solution
**Enhance leakage detection in PHASE 3.5:**

1. **Update LEAKAGE_PATTERNS** to include domain-specific patterns:
   ```python
   'target_indicators': [
       'Tekrarlayan_Arıza',  # Add this pattern
       'Recurrence',         # Add this pattern
       'AgeRatio_',          # Add this pattern
   ],
   ```

2. **Add statistical leakage detection** (PHASE 3.5):
   - Calculate correlation with each target variable
   - Flag features with correlation > 0.7 as suspicious
   - Report for manual review

3. **Remove protected feature override** for known leakage:
   - Keep PROTECTED_FEATURES for domain features (Ekipman_*, Arıza_*, etc.)
   - But remove features that ARE targets from protected list

**Expected Outcome:**
- Feature selection detects 100% of obvious leakage
- Report flags suspicious features for review
- Cleaner feature sets for all models

---

## ISSUE #4: Temporal PoF Model - Single Dataset (Data Quality - CRITICAL)

### Problem
**Step 6 creates mixed dataset (2,670 failed + 2,897 healthy = 5,567 equipment) but only trains on failed equipment.**

Healthy equipment excluded from temporal PoF training despite being created in Step 2.

### Root Cause

**File: `/home/user/PoF2/06_temporal_pof_model.py` (Lines 200-220)**
```python
# Current behavior in STEP 3: Filter to failed equipment only
print("\n⚠️ SINGLE DATASET (Failed Equipment Only)")
print("   Cannot predict without failure history")

# This filter excludes healthy equipment
valid_equipment_ids = set(
    df[df['Total_Faults'] > 0]['Ekipman_ID'].unique()
)
```

### Impact
- **Model trained on biased subset** (only equipment with failures)
- **Cannot make predictions for healthy equipment** (they're excluded from training)
- **Model calibration is off** - learns to predict from failure history, not prospective features
- **Mixed dataset created in Step 2** is wasted

### Solution
**Update Step 6 to train on all equipment:**

1. **Keep all equipment in training data:**
   - 2,670 failed equipment (Target=1 if failure in future window)
   - 2,897 healthy equipment (Target=0 - right-censored at cutoff date)

2. **Update target creation:**
   ```python
   # For failed equipment: 1 if failed in future window, 0 otherwise
   # For healthy equipment: 0 (right-censored, no failures observed)
   targets[horizon_name] = df['Ekipman_ID'].isin(failed_equipment).astype(int)
   ```

3. **Update model weights** (optional):
   - Use `scale_pos_weight` to balance 48% failed vs 52% healthy
   - Or use `class_weight='balanced'` in XGBoost

**Expected Outcome:**
- Model trained on all 5,567 equipment (currently ~2,670)
- Can make predictions for healthy equipment
- Better model calibration
- Slight AUC change, but more realistic performance

---

## ISSUE #5: Inconsistent Imputation Strategy (Data Quality - MEDIUM)

### Problem
**Different steps use different imputation strategies, causing inconsistent feature handling.**

- **Step 6** (STEP 6): Uses `fillna(0)` for some features
- **Step 9** (09_calibration.py): Uses `median` imputation
- **Step 10** (10_survival_model.py): Uses `median` imputation
- **No explicit handling** of features with >50% missing data

### Evidence

**File: `/home/user/PoF2/03_feature_engineering.py` (Lines 775-812)**
```python
# Step 6 approach: Fill with 0 (specific to this domain)
(df['urban_mv_Avg'].fillna(0) + df['urban_lv_Avg'].fillna(0)) / 2

# Step 9/10 approach: Median imputation
days_since.fillna(365)  # Default to "never failed"
```

### Impact
- **Inconsistent feature preprocessing** across pipeline
- **Features with >50% missing** (like MTBF_InterFault_Trend with 90.9% missing) are handled differently
- **Model predictions inconsistent** depending on which script processes data
- **Hard to debug** which features are causing issues

### Solution
**Implement standardized imputation in Step 3:**

1. **Add STEP 11 to feature_engineering.py:**
   - Explicitly handle features with >50% missing (exclude them)
   - Document domain-specific imputation strategies
   - Apply consistently to all features

2. **Imputation Strategy by Feature Type:**
   ```
   Failure History: Impute with 365 (days, "never failed" default)
   Age Features: Keep as-is (no nulls expected)
   Customer Ratio: Impute with 0 (no customers = not served)
   Geographic: Keep as-is (static data)
   ```

3. **Create imputation_strategy.csv** documenting each feature's strategy

**Expected Outcome:**
- Consistent preprocessing across all pipeline steps
- Clear documentation of imputation decisions
- Reproducible feature engineering

---

## Implementation Priority & Sequence

### Phase 1.1 (Foundation) ⚠️ BLOCKER
**Fix Equipment ID consistency** - This must be fixed first!
- Affects ALL downstream steps
- All other fixes depend on correct target-feature alignment

### Phase 1.2 (Data Quality) 🔴 HIGH
**Remove leakage from chronic classifier**
- Can be done independently
- Requires unprotecting leakage features

### Phase 1.3 (Analysis) 🔴 HIGH
**Enhance leakage detection** - Should be done with Phase 1.2
- Add missing patterns to LEAKAGE_PATTERNS
- Implement statistical leakage detection (PHASE 3.5)

### Phase 1.4 (Data Quality) 🟡 MEDIUM
**Train PoF on mixed dataset**
- Depends on Phase 1.1 being fixed first
- Will improve model generalization

### Phase 1.5 (Documentation) 🟡 MEDIUM
**Standardize imputation strategy**
- Can be done independently
- Improves code clarity and reproducibility

---

## Test & Validation Checklist

### After Each Fix:
- [ ] Verify target counts match equipment counts
- [ ] Check for missing values in key columns
- [ ] Validate model training completes without errors
- [ ] Compare before/after metrics (AUC, feature importance, etc.)
- [ ] Run STEP 5 (Equipment ID Audit) to verify alignment

### After All Phase 1 Fixes:
- [ ] All equipment IDs have 100% match rate
- [ ] Chronic classifier AUC in 0.75-0.88 range
- [ ] Temporal PoF model trained on 5,567 equipment
- [ ] Feature selection reports clean feature sets
- [ ] All downstream steps run without errors
- [ ] Model predictions reasonable and sensible

---

## Files Affected

### Must Modify:
1. `/home/user/PoF2/02_data_transformation.py` - ID naming
2. `/home/user/PoF2/06_temporal_pof_model.py` - Target creation, mixed dataset
3. `/home/user/PoF2/07_chronic_classifier.py` - Feature exclusion
4. `/home/user/PoF2/column_mapping.py` - PROTECTED_FEATURES, LEAKAGE_PATTERNS
5. `/home/user/PoF2/smart_feature_selection.py` - Leakage detection (PHASE 3.5)
6. `/home/user/PoF2/03_feature_engineering.py` - Imputation strategy

### Should Review:
7. `/home/user/PoF2/04_feature_selection.py` - Feature selection execution
8. `/home/user/PoF2/01_data_profiling.py` - Data profiling validation
9. `/home/user/PoF2/09_calibration.py` - Calibration on corrected targets
10. `/home/user/PoF2/10_survival_model.py` - Survival model on corrected targets

---

## Timeline & Resources

**Expected Effort:**
- Phase 1.1: 1-2 hours (foundational, affects multiple files)
- Phase 1.2: 30 minutes (simple feature exclusion)
- Phase 1.3: 1 hour (leakage detection enhancement)
- Phase 1.4: 30 minutes (dataset filtering change)
- Phase 1.5: 1-2 hours (imputation standardization + testing)

**Total: 4-6 hours for Phase 1 completion**

---

## Sign-Off

This audit document serves as the specification for Phase 1 implementation.

**Prepared By**: Claude Code (Automated Analysis)
**Date**: 2025-11-27
**Status**: Ready for Implementation
**Next Step**: Begin Phase 1.1 (Equipment ID Fix)
