# MODULE IMPLEMENTATION PLAN
**Turkish EDAŞ - Varlık Yönetimi Arıza İlişkisi**

## 📋 CURRENT STATUS vs MODULE REQUIREMENTS

### **MODULE 1: PoF + Recurring Failures**

| Requirement | Status | Current Implementation | Action Needed |
|-------------|--------|----------------------|---------------|
| **Risk scores (3/6/12/24M)** | ⚠️ PARTIAL | 6M, 12M only | Add 3M, 24M horizons |
| **Risk-ranked lists** | ✅ DONE | 387 high-risk equipment | None |
| **Geographic heat maps** | ❌ MISSING | No geo visualization | Create heat maps |
| **Survival curves** | ❌ MISSING | No survival analysis | Add Kaplan-Meier curves |
| **Chronic equipment (30/90d)** | ⚠️ PARTIAL | Lifetime-based repeaters | Add 30/90 day windows |
| **Hot-spot maps** | ❌ MISSING | No geographic clustering | Create hot-spot maps |
| **Pattern analysis** | ⚠️ PARTIAL | Equipment patterns only | Add crew/location patterns |

**Overall**: 25% complete → Target: 100%

---

### **MODULE 3: Fault Classification**

| Requirement | Status | Data Source | Action Needed |
|-------------|--------|-------------|---------------|
| **Hierarchical classification** | ❌ MISSING | "cause code" column | Create fault taxonomy |
| **Equipment fault profiles** | ❌ MISSING | cause code × equipment class | Build profiles |
| **Automated classification** | ❌ MISSING | ML classifier | Train classifier |
| **Root cause knowledge base** | ❌ MISSING | cause code patterns | Build knowledge base |

**Overall**: 0% complete → Target: Phase 2 (after Module 1 complete)

---

## 🎯 IMPLEMENTATION PRIORITY

### **PHASE 1: Complete Module 1 PoF Model** (HIGH PRIORITY)

#### **Task 1.1: Add "cause code" Feature** ⭐ IMMEDIATE
**File**: `02_data_transformation.py`
**Changes**:
```python
# Line ~350: Add cause code to aggregation
agg_dict.update({
    'cause code_first': 'Arıza_Nedeni_İlk',      # First fault cause
    'cause code_last': 'Arıza_Nedeni_Son',       # Most recent fault cause
    'cause code_mode': 'Arıza_Nedeni_Sık',       # Most common cause
})

# Count cause code distribution per equipment
cause_distribution = df.groupby(['Ekipman_ID', 'cause code']).size().unstack(fill_value=0)
# Add dominant cause type
equipment_df['Dominant_Cause'] = cause_distribution.idxmax(axis=1)
equipment_df['Dominant_Cause_Pct'] = cause_distribution.max(axis=1) / cause_distribution.sum(axis=1)
```

**File**: `03_feature_engineering.py`
**Changes**:
```python
# Add cause code features
df['Cause_Diversity'] = (cause_distribution > 0).sum(axis=1)  # How many different causes
df['Cause_Consistency'] = df['Dominant_Cause_Pct']  # Consistency of cause (high = always same cause)

# Cause × Equipment Class interaction
df['Cause_Class_Risk'] = df['Dominant_Cause'] + '_' + df['Equipment_Class_Primary']
```

**Expected Impact**: +3-5% model accuracy

---

#### **Task 1.2: Add Recurring Failure Logic (30/90 day)** ⭐ IMMEDIATE
**File**: `03_feature_engineering.py`
**New Features**:
```python
# Calculate repeat failures within windows
df['Repeat_30d_Count'] = count_repeats_within_days(fault_data, 30)
df['Repeat_90d_Count'] = count_repeats_within_days(fault_data, 90)

# Chronic equipment flags
df['Is_Chronic_30d'] = (df['Repeat_30d_Count'] >= 2).astype(int)  # Failed twice in 30 days
df['Is_Chronic_90d'] = (df['Repeat_90d_Count'] >= 2).astype(int)  # Failed twice in 90 days

# Average repeat interval
df['Avg_Repeat_Interval_Days'] = (
    (df['Son_Arıza_Tarihi'] - df['İlk_Arıza_Tarihi']).dt.days /
    df['Toplam_Arıza_Sayisi_Lifetime'].clip(lower=1)
)
```

**Output**: Separate chronic equipment lists for 30/90 day repeaters

---

#### **Task 1.3: Add 3M and 24M Horizons** ⚠️ OPTIONAL
**File**: `06_model_training.py`
**Issue**: Current model removed 3M (100% positive class issue)

**Solution**:
```python
# Option 1: Use temporal split (requires diagnostic)
# Run 00_temporal_diagnostic.py to check feasibility

# Option 2: Keep current approach, adjust thresholds
HORIZONS = {
    '3M': 90,
    '6M': 180,
    '12M': 365,
    '24M': 730
}

TARGET_THRESHOLDS = {
    '3M': 3,   # >= 3 lifetime failures (more stringent)
    '6M': 2,   # >= 2 lifetime failures
    '12M': 2,   # >= 2 lifetime failures
    '24M': 1    # >= 1 lifetime failure (less stringent)
}
```

**Recommendation**: Keep 6M/12M for now (most actionable for maintenance planning)

---

#### **Task 1.4: Geographic Visualizations** 🗺️
**File**: `09_geographic_analysis.py` (NEW SCRIPT)

**Requirements**:
- Geographic heat maps (failure density by location)
- Hot-spot maps (clusters of high-risk equipment)
- Pattern analysis by location

**Data Needed**:
- Coordinates (KOORDINAT_X, KOORDINAT_Y columns)
- Location hierarchy (İl, İlçe, Mahalle)

**Visualizations**:
```python
# 1. Failure density heat map
create_heatmap(
    lat=equipment_df['KOORDINAT_Y'],
    lon=equipment_df['KOORDINAT_X'],
    intensity=equipment_df['PoF_Score']
)

# 2. Hot-spot clustering
hot_spots = identify_hotspots(
    equipment_df,
    min_cluster_size=5,
    radius_km=1.0
)

# 3. Geographic risk distribution
plot_choropleth(
    regions=['İl', 'İlçe'],
    metric='Avg_PoF_Score'
)
```

---

#### **Task 1.5: Survival Curves** 📈
**File**: `10_survival_analysis.py` (NEW SCRIPT)

**Requirements**:
- Kaplan-Meier survival curves by equipment class
- Time-to-failure distributions
- Hazard rates

**Libraries**: `lifelines` package
```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(
    durations=equipment_df['Time_To_First_Failure'],
    event_observed=equipment_df['Has_Failed']
)

kmf.plot_survival_function()
```

---

### **PHASE 2: Module 3 Fault Classification** (FUTURE)

**Defer to Phase 2** - Focus on completing Module 1 first

---

## 📅 IMPLEMENTATION TIMELINE

### **IMMEDIATE (Today)**
1. ✅ Add "cause code" feature to 02_data_transformation.py
2. ✅ Add cause code features to 03_feature_engineering.py
3. ✅ Add 30/90 day recurring failure logic to 03_feature_engineering.py
4. ✅ Re-run pipeline: 02 → 03 → 04 (EDA) → 05 (feature selection) → 06 (model training)

### **SHORT TERM (Next 2-3 days)**
5. Create geographic visualization script (09_geographic_analysis.py)
6. Create survival analysis script (10_survival_analysis.py)
7. Validate outputs against Module 1 requirements

### **OPTIONAL (Future Enhancement)**
8. Add 3M/24M horizons (if needed)
9. Implement Module 3 fault classification
10. Add pattern analysis (crew/location)

---

## 🎯 EXPECTED OUTCOMES

After completing Phase 1:

### **Module 1 PoF Outputs** ✅
```
✓ Risk scores per equipment (6M, 12M horizons)
✓ Risk-ranked lists (top 387 high-risk equipment)
✓ Geographic heat maps (failure density visualization)
✓ Survival curves (Kaplan-Meier by equipment class)
✓ Chronic equipment lists (30/90 day repeaters identified)
✓ Hot-spot geographic maps (high-risk clusters)
✓ Pattern analysis (equipment/location patterns)
```

### **Model Improvements**
- Baseline: AUC 0.88
- With cause code: AUC 0.90-0.93 (estimated)
- With recurring logic: Better chronic equipment identification
- With geo analysis: Spatial risk patterns identified

---

## ❓ QUESTIONS FOR USER

1. **Cause Code Column**: Can you run this to confirm column exists and show sample values?
   ```bash
   python 01_data_profiling.py > profile_output.txt
   ```
   Then search for "cause" in the output.

2. **Coordinates**: Do you have KOORDINAT_X, KOORDINAT_Y columns for geographic maps?

3. **Priority**: Should I implement tasks 1.1-1.3 (cause code + recurring failures) first, then geo visualizations?

4. **3M/24M Horizons**: Do you really need 3M and 24M, or are 6M/12M sufficient for your maintenance planning?

---

## 🚀 READY TO IMPLEMENT

I'm ready to modify your pipeline scripts to add:
1. ✅ Cause code features (mekanik, elektriksel, çevresel classification)
2. ✅ 30/90 day recurring failure detection
3. ✅ Enhanced Module 1 outputs

**Shall I proceed with Tasks 1.1-1.3?** This will complete 75% of Module 1 requirements.
