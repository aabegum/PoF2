# 🎯 FINAL FIX: Exclude Equipment with No Pre-Cutoff Failures
**Turkish EDAŞ PoF Prediction Project**
**Date:** 2025-01-17
**Issue:** 184 equipment with NaN Son_Arıza_Gun_Sayisi filled with median causing data leakage
**Solution:** ✅ Explicitly exclude these equipment from temporal PoF training

---

## 🔍 PROBLEM DISCOVERED

### **The Discovery**

Running diagnostic verification revealed:
```
Son_Arıza_Gun_Sayisi statistics:
  Missing values: 184
  Median: 354 days

Verification results:
  Mismatches: 184 (equipment with feature value=354 vs manual=9999)
```

### **User's Critical Insight**

**Question:** "We shouldn't have equipment with no failures since all data is fault data. What do we mean by equipment with no failures?"

**Answer:** These 184 equipment **DO have failures**, but their **FIRST failure happened AFTER the cutoff date (2024-06-25)**!

```
Timeline Example:
  Installation:    2020
  Cutoff date:     2024-06-25  ← Feature calculation cutoff
  First failure:   2024-08-15  ← AFTER cutoff!

  Result:
    Pre-cutoff failures: 0
    Son_Arıza_Gun_Sayisi: NaN (no failures before cutoff)
    Target_6M: 1 (equipment fails in Aug 2024)
```

---

## 🚨 THE DATA LEAKAGE MECHANISM

### **Step-by-Step Leakage Chain**

**Step 1 - 02_data_transformation.py** ✅ CORRECT
```python
Equipment with NO pre-cutoff failures:
  Son_Arıza_Gun_Sayisi = NaN
  # Correctly indicates no failure history before cutoff
```

**Step 2 - 05_feature_selection.py** ❌ THE PROBLEM!
```python
# Line 191 (OLD CODE):
df[col].fillna(df[col].median(), inplace=True)

Result:
  Son_Arıza_Gun_Sayisi: NaN → 354 (median)
```

**Step 3 - Model Training** 🚨 DATA LEAKAGE!
```python
Equipment with first failure AFTER cutoff:
  Son_Arıza_Gun_Sayisi: 354 (filled median)
  Target_6M: 1 (they fail in Aug/Sep 2024)

Model learns:
  "354 days since last failure" → high probability to fail

But this is BACKWARDS!
  - Equipment with MORE days should be LESS likely to fail
  - But these 184 equipment with value=354 ALL fail (Target=1)
  - Perfect correlation → AUC stays high!
```

---

## ✅ THE SOLUTION

### **Option 1: Exclude from Temporal PoF (IMPLEMENTED)**

**Rationale:** You cannot predict **WHEN** equipment will fail if you have **NO historical failures** to learn from!

These 184 equipment should be handled differently:
- They are "new to failure"
- Use age-based risk or equipment-type-based rules
- Or flag for inspection/monitoring
- **NOT suitable for temporal PoF prediction**

---

## 🔧 FIXES IMPLEMENTED

### **File 1: `05_feature_selection.py`**

**Lines 187-200:**
```python
# 🔧 FIX: Do NOT fill Son_Arıza_Gun_Sayisi or MTBF_Gün with median
# Equipment with NaN have no pre-cutoff failures → cannot predict with temporal PoF
temporal_features = ['Son_Arıza_Gun_Sayisi', 'MTBF_Gün']

print("\n✓ Strategy: Filling missing values with median (except temporal features)")
print(f"   Note: {temporal_features} will remain NaN for equipment with no pre-cutoff failures")

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        if col in temporal_features:
            # Keep NaN for temporal features - indicates no failure history
            print(f"   ⚠️  Skipping {col} (will filter out NaN equipment later)")
        else:
            df[col].fillna(df[col].median(), inplace=True)
```

**Lines 566-585:**
```python
# Add reporting for excluded equipment
if 'Son_Arıza_Gun_Sayisi' in df_selected.columns:
    no_history_mask = df_selected['Son_Arıza_Gun_Sayisi'].isna()
    no_history_count = no_history_mask.sum()

    if no_history_count > 0:
        print(f"\n⚠️  IMPORTANT: Found {no_history_count} equipment with NO pre-cutoff failures")
        print(f"   These had their first failure AFTER 2024-06-25")
        print(f"   They will be EXCLUDED from temporal PoF training")

        # Save list of excluded equipment
        excluded_equipment = df_selected[no_history_mask][['Ekipman_ID']].copy()
        excluded_equipment['Exclusion_Reason'] = 'No pre-cutoff failures'
        excluded_equipment['First_Failure'] = 'After 2024-06-25'

        excluded_path = Path('data/excluded_equipment_no_history.csv')
        excluded_equipment.to_csv(excluded_path, index=False)
        print(f"   ✓ Saved excluded equipment list: {excluded_path}")
```

### **File 2: `06_model_training.py`**

**Lines 201-221:**
```python
# 🔧 FIX: Filter out equipment with no pre-cutoff failure history
if 'Son_Arıza_Gun_Sayisi' in df.columns:
    before_count = len(df)
    no_history_mask = df['Son_Arıza_Gun_Sayisi'].isna()
    no_history_count = no_history_mask.sum()

    if no_history_count > 0:
        print(f"\n⚠️  Excluding {no_history_count} equipment with NO pre-cutoff failures")
        print(f"   These had their first failure AFTER 2024-06-25")
        print(f"   Reason: Cannot predict temporal PoF without failure history")

        # Keep only equipment with failure history
        df = df[~no_history_mask].copy()

        print(f"   ✓ Equipment for temporal PoF: {len(df)} (excluded {no_history_count})")
        print(f"   ✓ Exclusion rate: {no_history_count/before_count*100:.1f}%")
```

---

## 🚀 NEXT STEPS (RUN ON YOUR MACHINE)

### **Step 1: Pull Latest Fixes**
```bash
git pull origin claude/review-asset-pipeline-01QCgELdMGzkZjyWr1Qh6B52
```

### **Step 2: Delete Old Feature Files**
```bash
# Windows PowerShell
del data\features_selected.csv
del data\features_selected_clean.csv
del data\features_reduced.csv
```

### **Step 3: Re-run Complete Pipeline**
```bash
# Regenerate features (05_feature_selection now skips filling temporal features)
python 05_feature_selection.py

# Remove leaky features
python 05b_remove_leaky_features.py

# Remove redundant features
python 05c_reduce_feature_redundancy.py

# Train model (will exclude 184 equipment with no history)
python 06_model_training.py
```

---

## 📊 EXPECTED RESULTS

### **Step 3: 05_feature_selection.py Output**
```
✓ Strategy: Filling missing values with median (except temporal features)
   Note: ['Son_Arıza_Gun_Sayisi', 'MTBF_Gün'] will remain NaN for equipment with no pre-cutoff failures
   ⚠️  Skipping Son_Arıza_Gun_Sayisi (will filter out NaN equipment later)
   ⚠️  Skipping MTBF_Gün (will filter out NaN equipment later)

⚠️  IMPORTANT: Found 184 equipment with NO pre-cutoff failures
   These had their first failure AFTER 2024-06-25
   They will be EXCLUDED from temporal PoF training
   (Cannot predict when equipment will fail if no failure history)
   ✓ Saved excluded equipment list: data\excluded_equipment_no_history.csv
```

### **Step 6: 06_model_training.py Output**
```
✓ Loaded: 789 equipment × 19 features

⚠️  Excluding 184 equipment with NO pre-cutoff failures
   These had their first failure AFTER 2024-06-25
   Reason: Cannot predict temporal PoF without failure history
   ✓ Equipment for temporal PoF: 605 (excluded 184)
   ✓ Exclusion rate: 23.3%

--- Creating Binary Temporal Targets ---
6M Target (will fail in next 180 days):
   Will fail (1):     ~120-140 equipment
   Won't fail (0):    ~465-485 equipment

XGBoost 6M Test Set Results:
   AUC: 0.75-0.85  ← DOWN from 0.9456! ✅

XGBoost 12M Test Set Results:
   AUC: 0.80-0.90  ← DOWN from 0.9934! ✅
```

### **Verification Script**
```bash
python verify_feature_calculation.py
```

**Expected:**
```
✅ GOOD: Son_Arıza_Gun_Sayisi uses only pre-cutoff faults!
   Exact matches: 605
   Mismatches: 0  ← DOWN from 184! ✅
```

---

## 📋 NEW FILES CREATED

1. **data/excluded_equipment_no_history.csv**
   - List of 184 equipment with no pre-cutoff failures
   - Columns: Ekipman_ID, Exclusion_Reason, First_Failure
   - Use this for:
     - Separate "new to failure" risk model
     - Age-based or equipment-type-based rules
     - Flagging for monitoring/inspection

---

## 📊 IMPACT SUMMARY

### **Before Fix**
```
Dataset: 789 equipment
  - 605 with pre-cutoff failures (valid for temporal PoF)
  - 184 with NO pre-cutoff failures (invalid for temporal PoF)

Feature Processing:
  Son_Arıza_Gun_Sayisi: NaN → 354 (median) ❌ DATA LEAKAGE

Model Training:
  Uses all 789 equipment
  AUC 6M: 0.9456 (too high - leakage)
  AUC 12M: 0.9934 (too high - leakage)
```

### **After Fix**
```
Dataset: 789 equipment
  - 605 with pre-cutoff failures → Used for temporal PoF ✅
  - 184 with NO pre-cutoff failures → Excluded ✅

Feature Processing:
  Son_Arıza_Gun_Sayisi: NaN → stays NaN ✅ CORRECT

Model Training:
  Uses only 605 equipment (with failure history)
  AUC 6M: 0.75-0.85 (realistic) ✅
  AUC 12M: 0.80-0.90 (realistic) ✅
```

---

## 🎯 KEY INSIGHTS

1. **Not all equipment are suitable for temporal PoF:**
   - Equipment need failure history to predict future failures
   - 184 equipment had their FIRST failure after cutoff
   - These should use different risk assessment method

2. **Filling NaN with median is NOT always safe:**
   - For temporal features, NaN has semantic meaning
   - NaN = "no failure history" ≠ "median failure history"
   - Must preserve NaN and handle explicitly

3. **Smaller dataset with clean data > Larger dataset with leakage:**
   - Using 605 equipment with history: AUC 0.75-0.85 (useful)
   - Using 789 equipment with median-filled NaN: AUC 0.99 (useless)

---

## 💡 RECOMMENDATIONS FOR EXCLUDED EQUIPMENT

The 184 excluded equipment still need risk assessment! Recommended approaches:

### **Option A: Age-Based Risk**
```python
# For equipment with no failure history, use age as primary indicator
if equipment_age > 15_years:
    risk = "High - replacement candidate"
elif equipment_age > 10_years:
    risk = "Medium - monitor closely"
else:
    risk = "Low - normal monitoring"
```

### **Option B: Equipment-Type Rules**
```python
# Different equipment types have different baseline risk
if equipment_type in ['OG/AG Trafo', 'Rekortman']:
    baseline_risk = "High"
elif equipment_type in ['AG Pano Box']:
    baseline_risk = "Medium"
```

### **Option C: Hybrid Approach**
```python
# Combine age + type + customer impact
risk_score = (
    age_factor * 0.4 +
    type_factor * 0.3 +
    customer_impact_factor * 0.3
)
```

---

## ✅ SUCCESS CRITERIA

You'll know the fix worked when:

1. ✅ `05_feature_selection.py` reports: "Skipping Son_Arıza_Gun_Sayisi"
2. ✅ `excluded_equipment_no_history.csv` created with 184 equipment
3. ✅ `06_model_training.py` shows: "Excluding 184 equipment"
4. ✅ Model trains on **605 equipment** (not 789)
5. ✅ AUC 6M: 0.75-0.85 (realistic)
6. ✅ AUC 12M: 0.80-0.90 (realistic)
7. ✅ `verify_feature_calculation.py` shows: 0 mismatches

---

**Last Updated:** 2025-01-17
**Status:** ✅ FINAL FIX IMPLEMENTED
**Action Required:** Run steps 1-3 above and verify results
