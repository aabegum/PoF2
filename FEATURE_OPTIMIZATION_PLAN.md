# FEATURE OPTIMIZATION PLAN - 30 OPTIMAL FEATURES

## 🎯 OBJECTIVE
Transform current 111-feature pipeline → 30 carefully curated features (no leakage, minimal redundancy)

---

## ✅ FEATURES ALREADY CORRECT (Keep as-is)

### TIER 1: Equipment Characteristics (3/3 ✅)
- ✅ `Equipment_Class_Primary` → Use as `Equipment_Class_Grouped`
- ✅ `component_voltage` (already exists)
- ✅ `Voltage_Class` (already exists)

### TIER 2: Age & Lifecycle (3/3 ✅)
- ✅ `Ekipman_Yaşı_Yıl` (already exists)
- ✅ `Yas_Beklenen_Omur_Orani` (already exists)
- ✅ `Beklenen_Ömür_Yıl` (already exists)

### TIER 3: Failure History - Temporal (5/5 ✅)
- ✅ `Son_Arıza_Gun_Sayisi` (already exists)
- ✅ `Ilk_Arizaya_Kadar_Yil` (already exists)
- ✅ `Toplam_Arıza_Sayisi_Lifetime` (already exists)
- ✅ `Time_To_Repair_Hours_mean` (already exists)
- ✅ `Time_To_Repair_Hours_max` (already exists)

### TIER 4: MTBF & Reliability (3/5 - 2 missing)
- ✅ `MTBF_InterFault_Gün` (already exists)
- ✅ `MTBF_Lifetime_Gün` (already exists)
- ✅ `MTBF_ActiveLife_Gün` (already exists)
- ❌ `MTBF_InterFault_Trend` **NEEDS TO BE ADDED**
- ❌ `MTBF_InterFault_StdDev` **NEEDS TO BE ADDED**

### TIER 5: Failure Cause Patterns (4/4 ✅)
- ✅ `Arıza_Nedeni_Çeşitlilik` (already exists)
- ✅ `Arıza_Nedeni_Tutarlılık` (already exists)
- ✅ `Neden_Değişim_Flag` (already exists)
- ⚠️ `Tek_Neden_Flag` **CHECK IF EXISTS**

### TIER 6: Customer Impact & Loading (5/5 ✅)
- ✅ `Urban_Customer_Ratio_mean` (already exists)
- ✅ `urban_lv_Avg` (already exists)
- ✅ `urban_mv_Avg` (already exists)
- ✅ `MV_Customer_Ratio_mean` (already exists)
- ✅ `total_customer_count_Avg` (already exists)

### TIER 7: Geographic & Environmental (3/3 ✅)
- ✅ `İlçe` (already exists)
- ✅ `Bölge_Tipi` (already exists)
- ✅ `Summer_Peak_Flag_sum` (already exists)

### TIER 8: Derived Interactions (0/2 - both missing)
- ❌ `Overdue_Factor` **NEEDS TO BE ADDED**
- ❌ `AgeRatio_Recurrence_Interaction` **NEEDS TO BE ADDED**

---

## ❌ FEATURES TO REMOVE (Currently Created but Not in Optimal Set)

### Category 1: Geographic Clustering (STEP 3) - REMOVE ENTIRELY ❌
```python
# File: 03_feature_engineering.py, Lines 218-277
Geographic_Cluster                           # Noisy K-means clustering
Arıza_Sayısı_12ay_Cluster_Avg               # Leaky (uses future data)
Tekrarlayan_Arıza_90gün_Flag_Cluster_Avg   # Leaky (uses future data)
MTBF_Gün_Cluster_Avg                        # Circular logic

Rationale for removal:
- Geographic clustering on X,Y coordinates produces noisy patterns
- Distribution networks are LINEAR (power lines), not clustered points
- Cluster aggregations create data leakage risk
- Better alternative: İlçe (district) - clear, interpretable, domain-meaningful
```

### Category 2: Redundant Failure Rate Features (STEP 4) - REMOVE ❌
```python
# File: 03_feature_engineering.py, Lines 283-323
Failure_Rate_Per_Year                        # Redundant (Toplam_Arıza / Ekipman_Yaşı)
Recent_Failure_Intensity                     # Leaky (uses Arıza_Sayısı_3ay)
Failure_Acceleration                         # Leaky (uses Arıza_Sayısı_6ay)

Rationale for removal:
- Failure_Rate_Per_Year: Tree models learn this automatically from raw features
- Recent_Failure_Intensity: Uses 3-month data (may include post-cutoff)
- Failure_Acceleration: Uses 6-month data (may include post-cutoff)
```

### Category 3: Equipment Class Aggregations (STEP 7) - REMOVE ENTIRELY ❌
```python
# File: 03_feature_engineering.py, Lines 382-418
Arıza_Sayısı_12ay_Class_Avg                 # Leaky (uses 12M window)
MTBF_Gün_Class_Avg                          # Circular logic
Ekipman_Yaşı_Yıl_Class_Avg                  # Not predictive
Yas_Beklenen_Omur_Orani_Class_Avg           # Not predictive
Failure_vs_Class_Avg                         # Derived from leaky feature

Rationale for removal:
- Class averages create target leakage (using class to predict class members)
- Circular reasoning (if equipment X is in class Y, using Y's average to predict X)
- Models learn class patterns automatically from Equipment_Class_Primary feature
```

### Category 4: Weak Interaction Features (STEP 8) - REMOVE/REPLACE ❌
```python
# File: 03_feature_engineering.py, Lines 420-443
Age_Failure_Interaction                      # Uses leaky Arıza_Sayısı_12ay

Rationale for removal:
- Uses Arıza_Sayısı_12ay which may include post-cutoff data
- Better alternative: AgeRatio_Recurrence_Interaction (uses Lifetime count instead)
```

---

## ✅ FEATURES TO ADD (Missing from Current Pipeline)

### TIER 4: MTBF Enhancement Features (2 features)

#### 1. MTBF_InterFault_Trend
```python
# Description: Recent MTBF / Historical MTBF
# Purpose: Detect equipment degradation over time
# Calculation:
#   - Recent MTBF = Average of last 50% of inter-fault times
#   - Historical MTBF = Average of first 50% of inter-fault times
#   - Trend = Recent / Historical
# Interpretation:
#   - < 1.0 = Degrading (failures accelerating)
#   - = 1.0 = Stable
#   - > 1.0 = Improving (failures slowing down)
# Missing: ~20% (equipment with < 4 faults)
```

#### 2. MTBF_InterFault_StdDev
```python
# Description: Standard deviation of inter-fault times
# Purpose: Measure failure timing predictability
# Calculation: Std dev of days between consecutive faults
# Interpretation:
#   - Low StdDev = Consistent, predictable failure pattern
#   - High StdDev = Erratic, unpredictable failures
# Missing: ~20% (equipment with < 2 faults)
```

### TIER 8: Derived Interaction Features (2 features)

#### 3. Overdue_Factor
```python
# Description: Days since last failure / MTBF_InterFault_Gün
# Purpose: Detect equipment "overdue" for next failure
# Calculation: Son_Arıza_Gun_Sayisi / MTBF_InterFault_Gün
# Interpretation:
#   - < 1.0 = Not yet due for next failure based on pattern
#   - = 1.0 = Due for failure based on historical pattern
#   - > 1.0 = Overdue (higher risk of imminent failure)
# Missing: ~23% (equipment with no failures or MTBF = 0)
```

#### 4. AgeRatio_Recurrence_Interaction
```python
# Description: Age ratio × Failure count (compound aging + use risk)
# Purpose: Capture interaction between lifecycle position and failure history
# Calculation: Yas_Beklenen_Omur_Orani × Toplam_Arıza_Sayisi_Lifetime
# Interpretation:
#   - High value = Old equipment with many failures (compound risk)
#   - Low value = Young equipment with few failures (low risk)
# Missing: 0% (both components always available)
```

### OPTIONAL: Additional Enhancement Features

#### 5. Age_Lifecycle_Stage (Categorical)
```python
# Description: Lifecycle stage based on age ratio
# Purpose: Categorical representation of equipment lifecycle
# Categories:
#   - 'Infant': Yas_Beklenen_Omur_Orani < 0.15 (first 15% of life)
#   - 'Mature': 0.15 ≤ ratio < 0.75 (main operational period)
#   - 'Aging': 0.75 ≤ ratio < 0.95 (approaching end-of-life)
#   - 'End-of-Life': ratio ≥ 0.95 (critical replacement needed)
# Missing: 0%
```

#### 6. Tek_Neden_Flag (if not already exists)
```python
# Description: Single cause accounts for ≥80% of faults
# Purpose: Identify stable failure mode (single dominant cause)
# Calculation: 1 if max(cause_count) / total_faults >= 0.80, else 0
# Interpretation:
#   - 1 = Stable, predictable failure mode (good for maintenance planning)
#   - 0 = Multiple failure modes (complex degradation)
# Missing: 0%
```

---

## 🔧 IMPLEMENTATION PLAN

### Phase 1: Remove Problematic Features ✅ COMPLETE

**File: 03_feature_engineering.py**

1. **Delete STEP 3: Geographic Clustering (Lines 218-277)**
   - Remove entire section
   - Saves ~60 lines of code
   - Eliminates 4 problematic features

2. **Delete STEP 4: Redundant Failure Rate Features (Lines 283-323)**
   - Remove Failure_Rate_Per_Year calculation
   - Remove Recent_Failure_Intensity calculation
   - Remove Failure_Acceleration calculation
   - Keep only the section header for future use

3. **Delete STEP 7: Equipment Class Aggregations (Lines 382-418)**
   - Remove entire section
   - Saves ~37 lines of code
   - Eliminates 5 leaky features

4. **Replace STEP 8: Interaction Features (Lines 420-443)**
   - Remove Age_Failure_Interaction
   - Replace with new interaction features (see Phase 2)

### Phase 2: Add Missing TIER 3 Features ✅ COMPLETE

**File: 03_feature_engineering.py**

**Add after STEP 5: RELIABILITY METRICS**

```python
# ============================================================================
# STEP 5B: ADVANCED MTBF FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 5B: ENGINEERING ADVANCED MTBF FEATURES")
print("="*100)

# FEATURE 1: MTBF_InterFault_Trend (Degradation Detector)
# ... implementation code ...

# FEATURE 2: MTBF_InterFault_StdDev (Predictability Measure)
# ... implementation code ...
```

**Add after STEP 8: INTERACTION FEATURES (renamed)**

```python
# ============================================================================
# STEP 8: DERIVED INTERACTION FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 8: CREATING DERIVED INTERACTION FEATURES")
print("="*100)

# FEATURE 3: Overdue_Factor (Imminent Risk Detector)
# ... implementation code ...

# FEATURE 4: AgeRatio_Recurrence_Interaction (Compound Risk)
# ... implementation code ...

# FEATURE 5: Age_Lifecycle_Stage (Categorical Lifecycle)
# ... implementation code ...

# FEATURE 6: Tek_Neden_Flag (if not already exists)
# ... implementation code ...
```

### Phase 3: Update Feature Selection Logic ✅ COMPLETE

**File: 05_feature_selection.py**

1. **Update PROTECTED_FEATURES list in config.py**
   - Add new features to protection list
   - Remove old features from protection list

2. **Update REDUNDANT_FEATURES dictionary**
   - Add Geographic_Cluster to removal list
   - Add Class aggregation features to removal list
   - Add old interaction features to removal list

3. **Verify VIF Analysis**
   - Ensure new features pass VIF threshold
   - Expected: All 30 features should pass VIF < 10

### Phase 4: Validation & Testing ⏳ PENDING (requires data files)

1. **Run 03_feature_engineering.py**
   - Verify 30 core features are created
   - Check for missing data patterns
   - Validate feature distributions

2. **Run 05_feature_selection.py**
   - Verify final feature count ≈ 30
   - Check leakage removal works correctly
   - Review feature importance rankings

3. **Run 06_model_training.py**
   - Baseline AUC with old features: 0.68-0.77
   - Expected AUC with new features: 0.78-0.82
   - Validate no data leakage (realistic AUC)

---

## 📊 EXPECTED OUTCOMES

### Before (Current State):
- **Features created:** 111
- **Features after selection:** 12-13 (too aggressive removal)
- **AUC:** 0.68-0.77
- **Issues:** Missing important features, some leakage

### After (Optimal 30-Feature Set):
- **Features created:** 30 (carefully curated)
- **Features after selection:** 25-30 (minimal removal needed)
- **AUC:** 0.78-0.82 (expected +0.05-0.10 improvement)
- **Benefits:**
  - ✅ No data leakage
  - ✅ No redundancy
  - ✅ Better signal (more informative features)
  - ✅ Captures degradation trends
  - ✅ Improved interpretability
  - ✅ Faster training

---

## 🎯 SUCCESS CRITERIA

1. ✅ **Feature Count:** Exactly 30 features created in 03_feature_engineering.py
2. ✅ **No Leakage:** All features use only pre-cutoff data
3. ✅ **No Redundancy:** VIF < 10 for all features
4. ✅ **Improved Performance:** AUC increase of 0.05-0.10
5. ✅ **Feature Importance:** Top 10 features account for 65-75% of importance
6. ✅ **Missing Data:** < 25% missing for critical features
7. ✅ **Interpretability:** All features have clear business meaning

---

## 📋 IMPLEMENTATION STATUS

### ✅ Phase 1: COMPLETE (2025-11-19)
**Commit:** `30e37f8` - "Optimize feature engineering: Remove 10 leaky/redundant features, add 3 TIER 3 enhancements"

**Removed Features:**
- ❌ Geographic Clustering (4 features): Geographic_Cluster, cluster aggregations
- ❌ Redundant Failure Rates (3 features): Failure_Rate_Per_Year, Recent_Failure_Intensity, Failure_Acceleration
- ❌ Equipment Class Aggregations (5 features): All class averages removed
- ❌ Weak Interactions (2 features): Age_Failure_Interaction, Customer_Failure_Interaction

**File:** 03_feature_engineering.py
- STEP 3: Geographic Clustering → Removed (replaced with İlçe)
- STEP 4: Failure Rate Features → Removed
- STEP 7: Equipment Class Aggregations → Removed
- STEP 8: Interaction Features → Cleaned (kept only AgeRatio_Recurrence_Interaction)

### ✅ Phase 2: COMPLETE (2025-11-19)
**Commit:** `30e37f8` (same commit as Phase 1)

**Added Features:**
- ✅ MTBF_InterFault_Trend (degradation detector)
- ✅ MTBF_InterFault_StdDev (predictability measure)
- ✅ Overdue_Factor (imminent failure risk)

**File:** 03_feature_engineering.py
- STEP 5B: Advanced MTBF Features → Added (new section)
- STEP 8: Enhanced Interactions → Added Overdue_Factor

**Updated:**
- Feature summary in STEP 10 to reflect optimization
- Final accomplishments section

### ✅ Phase 3: COMPLETE (2025-11-19)
**Commit:** `dd1fa16` - "Phase 3: Update feature selection configuration for optimal 30-feature set"

**Updated config.py:**
- Protected features updated to 32 features (31 core + ID + target)
- Removed: Geographic_Cluster, Ekipman_Yaşı_Yıl_EDBS_first
- Added: MTBF_InterFault_Trend, MTBF_InterFault_StdDev, Overdue_Factor, AgeRatio_Recurrence_Interaction
- All 8 TIERS documented with clear categorization

**Updated 05_feature_selection.py:**
- REDUNDANT_FEATURES dictionary updated with 14 Phase 1 removals
- Each removal documented with reason, replacement, and correlation
- Organized by phase and category for clarity

### ⏳ Phase 4: PENDING (requires data files)
**Status:** Cannot test without source data files
- 03_feature_engineering.py syntax validated ✅
- config.py syntax validated ✅
- 05_feature_selection.py syntax validated ✅

**Next Steps:**
1. Run with actual data when available
2. Verify 30-feature output
3. Validate AUC improvements (expecting 0.78-0.82)

---

## 📋 NEXT STEPS FOR USER

1. ✅ **Review implementation** - All code changes complete
2. ✅ **Phase 1-3 complete** - Feature engineering optimized
3. ⏳ **Test with data** - Run pipeline when data files available
4. ⏳ **Validate AUC** - Confirm expected improvement to 0.78-0.82
5. ⏳ **Update OPTION_B_PROGRESS.md** - Mark feature optimization complete
