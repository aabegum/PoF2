# HEALTHY EQUIPMENT INTEGRATION PLAN
**Date**: 2025-11-25
**Status**: 🔄 IMPLEMENTATION READY
**Priority**: HIGH (Improves model quality significantly)

---

## Executive Summary

**Current Problem**:
- Pipeline only uses **failed equipment** (all positive samples)
- No true negative samples for model training
- Class imbalance leads to poor generalization
- Models cannot distinguish healthy vs at-risk equipment

**Solution**:
- Add **healthy equipment data** as separate input file
- Merge healthy + failed equipment datasets
- Create balanced training with positive + negative samples
- Improve model accuracy and reliability

**Expected Benefits**:
- ✅ Better probability calibration
- ✅ Reduced false positives
- ✅ More realistic risk scores
- ✅ Improved model generalization

---

## Current Pipeline Analysis

### Data Flow (Current State)

```
INPUT: data/combined_data_son.xlsx (fault-level records)
  ↓
02_data_transformation.py → Transforms fault-level to equipment-level
  ↓
OUTPUT: data/equipment_level_data.csv (ONLY FAILED EQUIPMENT)
  ↓
03_feature_engineering.py → Creates advanced features
  ↓
OUTPUT: data/features_engineered.csv
  ↓
04_feature_selection.py → Removes leaky features
  ↓
OUTPUT: data/features_selected_clean.csv
  ↓
06_temporal_pof_model.py → Trains PoF models (all positive samples)
07_chronic_classifier.py → Identifies chronic repeaters
10_survival_model.py → Cox PH survival analysis
09_calibration.py → Calibrates probabilities
11_consequence_of_failure.py → Risk assessment
```

### Key Issues Identified

1. **02_data_transformation.py** (Lines 114-116):
   - Only loads fault records from Excel
   - No healthy equipment data source

2. **03_feature_engineering.py**:
   - Assumes all equipment have failure history
   - Calculates MTBF, fault counts (won't work for 0 faults)

3. **06_temporal_pof_model.py** (Lines 174-192):
   - **CRITICAL**: Explicitly excludes equipment with NO pre-cutoff failures
   - This is designed ONLY for failed equipment
   - Target creation assumes failures exist

4. **07_chronic_classifier.py**:
   - Identifies chronic repeaters (90-day recurrence)
   - Only relevant for failed equipment

5. **10_survival_model.py** (Lines 204-236):
   - Creates time-to-event from consecutive failures
   - Censors LAST failure only
   - No handling for equipment with 0 failures

---

## Implementation Plan

### Phase 1: Data Integration (NEW SCRIPT)

**Create: `02a_healthy_equipment_loader.py`**

**Purpose**: Load and prepare healthy equipment data

**Inputs**:
- `data/healthy_equipment.csv` (NEW - user provides)
  - Required columns:
    - `cbs_id` (Equipment ID)
    - `Şebeke Unsuru` (Equipment Type)
    - `Sebekeye_Baglanma_Tarihi` (Grid Connection Date)
    - Optional: Geographic info, customer impact metrics

**Outputs**:
- `data/healthy_equipment_prepared.csv`

**Key Logic**:
```python
# Load healthy equipment
df_healthy = pd.read_csv('data/healthy_equipment.csv')

# Add metadata
df_healthy['Has_Failure_History'] = 0
df_healthy['Total_Faults'] = 0
df_healthy['Last_Fault_Date'] = pd.NaT

# Calculate age
df_healthy['Ekipman_Yaşı_Yıl'] = calculate_age(df_healthy['Sebekeye_Baglanma_Tarihi'])

# Set zero-fault features
df_healthy['Fault_Count_3M'] = 0
df_healthy['Fault_Count_6M'] = 0
df_healthy['Fault_Count_12M'] = 0
df_healthy['MTBF_Gün'] = np.nan  # No failures → no MTBF
df_healthy['Son_Arıza_Gun_Sayisi'] = np.nan  # No last failure
```

---

### Phase 2: Update Data Transformation

**Modify: `02_data_transformation.py`**

**Changes**:

1. **After Equipment-Level Transformation** (Line ~1271):
```python
# ============================================================================
# STEP 13: MERGE WITH HEALTHY EQUIPMENT (NEW)
# ============================================================================
print("\n[Step 13/13] Merging with Healthy Equipment Data...")

healthy_file = DATA_DIR / 'healthy_equipment_prepared.csv'

if healthy_file.exists():
    print(f"\n✓ Loading healthy equipment from: {healthy_file}")
    df_healthy = pd.read_csv(healthy_file)
    print(f"✓ Loaded: {len(df_healthy):,} healthy equipment")

    # Ensure column compatibility
    failed_cols = set(equipment_df.columns)
    healthy_cols = set(df_healthy.columns)

    # Add missing columns to healthy equipment (with safe defaults)
    missing_in_healthy = failed_cols - healthy_cols
    for col in missing_in_healthy:
        if 'Fault_Count' in col or 'MTBF' in col:
            df_healthy[col] = 0  # Zero faults
        else:
            df_healthy[col] = np.nan

    # Merge datasets
    equipment_df_combined = pd.concat([equipment_df, df_healthy], ignore_index=True)
    print(f"\n✓ Combined dataset: {len(equipment_df_combined):,} total equipment")
    print(f"  • Failed equipment: {len(equipment_df):,} ({len(equipment_df)/len(equipment_df_combined)*100:.1f}%)")
    print(f"  • Healthy equipment: {len(df_healthy):,} ({len(df_healthy)/len(equipment_df_combined)*100:.1f}%)")

    # Use combined dataset
    equipment_df = equipment_df_combined
else:
    print(f"\n⚠️  WARNING: Healthy equipment file not found at {healthy_file}")
    print(f"  Pipeline will continue with ONLY failed equipment (current behavior)")
```

2. **Update Output Path**:
```python
# Save combined dataset
output_path = DATA_DIR / 'equipment_level_data_combined.csv'
equipment_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Saved: {output_path} ({len(equipment_df):,} equipment)")
```

---

### Phase 3: Update Feature Engineering

**Modify: `03_feature_engineering.py`**

**Changes**:

1. **Handle Zero-Fault Equipment** (After loading, ~Line 107):
```python
# ============================================================================
# HANDLE HEALTHY EQUIPMENT (ZERO FAULTS)
# ============================================================================
print("\n--- Handling Equipment with Zero Faults ---")

# Identify healthy equipment
has_failures = df['Has_Failure_History'] == 1 if 'Has_Failure_History' in df.columns else df['Total_Faults'] > 0
healthy_count = (~has_failures).sum()

print(f"  Healthy equipment (0 faults): {healthy_count:,} ({healthy_count/len(df)*100:.1f}%)")
print(f"  Failed equipment (>0 faults): {has_failures.sum():,} ({has_failures.sum()/len(df)*100:.1f}%)")

# For healthy equipment, set safe defaults for fault-based features
if healthy_count > 0:
    print(f"\n  Setting safe defaults for {healthy_count:,} healthy equipment...")

    # Fault counts → 0
    fault_count_cols = [col for col in df.columns if 'Fault_Count' in col]
    for col in fault_count_cols:
        df.loc[~has_failures, col] = 0

    # MTBF → NaN (cannot calculate without failures)
    mtbf_cols = [col for col in df.columns if 'MTBF' in col]
    for col in mtbf_cols:
        df.loc[~has_failures, col] = np.nan

    # Time since last fault → Max (very old/no fault)
    if 'Son_Arıza_Gun_Sayisi' in df.columns:
        df.loc[~has_failures, 'Son_Arıza_Gun_Sayisi'] = np.nan

    print(f"  ✓ Safe defaults applied")
```

2. **Update Expected Life Ratio Calculation** (~Line 176):
```python
def calculate_age_ratio(row):
    """Calculate ratio of current age to expected life"""
    age = row['Ekipman_Yaşı_Yıl']
    expected_life = row['Beklenen_Ömür_Yıl']

    if pd.notna(age) and expected_life > 0:
        return age / expected_life
    return None  # Healthy equipment may have ratio < 0.5 (young)

df['Yas_Beklenen_Omur_Orani'] = df.apply(calculate_age_ratio, axis=1)
```

---

### Phase 4: Update Temporal PoF Model

**Modify: `06_temporal_pof_model.py`**

**Changes**:

1. **REMOVE Equipment Exclusion** (Lines 174-192):
```python
# 🔧 UPDATED: Include ALL equipment (both failed and healthy)
# Previously: Excluded equipment with NO pre-cutoff failures
# Now: Keep all equipment - healthy equipment are true negatives!

print(f"\n✓ Including ALL equipment for balanced training")
print(f"   Equipment with failure history: {df['Son_Arıza_Gun_Sayisi'].notna().sum():,}")
print(f"   Equipment with NO failures (healthy): {df['Son_Arıza_Gun_Sayisi'].isna().sum():,}")
print(f"   Total equipment for PoF modeling: {len(df):,}")
```

2. **Update Target Creation** (Lines 196-250):
```python
# Load fault-level data for target creation
df_faults = pd.read_excel(INPUT_FILE)

# Parse dates
df_faults['started at'] = pd.to_datetime(df_faults['started at'], errors='coerce')

# For each horizon, create target
for horizon, days in HORIZONS.items():
    target_col = f'Target_{horizon}'

    # Default: All equipment start as negative (0)
    df[target_col] = 0

    # Mark equipment that FAILED in this window
    future_start = CUTOFF_DATE
    future_end = CUTOFF_DATE + timedelta(days=days)

    # Find faults in prediction window
    future_faults = df_faults[
        (df_faults['started at'] > future_start) &
        (df_faults['started at'] <= future_end)
    ]

    # Mark equipment as positive (1) if they failed
    failed_equipment = future_faults['cbs_id'].unique()
    df.loc[df['Ekipman_ID'].isin(failed_equipment), target_col] = 1

    # Class distribution
    positive_count = df[target_col].sum()
    negative_count = len(df) - positive_count
    positive_pct = positive_count / len(df) * 100

    print(f"\n{horizon} Target Distribution:")
    print(f"  Positive (will fail): {positive_count:,} ({positive_pct:.1f}%)")
    print(f"  Negative (healthy): {negative_count:,} ({100-positive_pct:.1f}%)")
    print(f"  Class balance: {positive_count/negative_count:.3f}")
```

3. **Update Class Weights** (~Line 400):
```python
# Calculate balanced class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train
)

scale_pos_weight = class_weights[1] / class_weights[0]

print(f"\n⚖️  Class Weight Strategy:")
print(f"   Negative weight: {class_weights[0]:.3f}")
print(f"   Positive weight: {class_weights[1]:.3f}")
print(f"   Scale pos weight: {scale_pos_weight:.3f}")
```

---

### Phase 5: Update Chronic Classifier

**Modify: `07_chronic_classifier.py`**

**Changes**:

1. **Filter to Failed Equipment Only** (Early in script):
```python
# ============================================================================
# FILTER TO FAILED EQUIPMENT (Chronic classifier only for equipment with failures)
# ============================================================================
print("\n--- Filtering to Failed Equipment ---")

# Chronic classifier is ONLY relevant for equipment with failure history
has_failures = df['Total_Faults'] > 0
healthy_count = (~has_failures).sum()

if healthy_count > 0:
    print(f"\n⚠️  Excluding {healthy_count:,} healthy equipment (no failures)")
    print(f"   Chronic classifier requires failure history")
    df = df[has_failures].copy()
    print(f"   ✓ Equipment for chronic classification: {len(df):,}")
```

2. **Update Documentation**:
```python
"""
CHRONIC FAILURE CLASSIFIER
Turkish EDAŞ PoF Prediction Project

Purpose:
- Identify chronic repeaters (90-day recurrence pattern)
- ONLY applicable to equipment with failure history
- Healthy equipment are excluded (no failures to classify)

Input:  data/features_selected_clean.csv (filtered to failed equipment)
Output: models/chronic_classifier.pkl, predictions/chronic_predictions.csv
"""
```

---

### Phase 6: Update Survival Analysis

**Modify: `10_survival_model.py`**

**Changes**:

1. **Add Right-Censored Observations for Healthy Equipment** (After Line 248):
```python
# ============================================================================
# ADD HEALTHY EQUIPMENT AS RIGHT-CENSORED OBSERVATIONS
# ============================================================================
print("\n--- Adding Healthy Equipment as Right-Censored Observations ---")

# Identify healthy equipment (no failures in fault data)
equipment_with_faults = df_faults['Ekipman_ID'].unique()
all_equipment = df_features['Ekipman_ID'].unique()
healthy_equipment = set(all_equipment) - set(equipment_with_faults)

print(f"\n  Equipment with faults: {len(equipment_with_faults):,}")
print(f"  Healthy equipment: {len(healthy_equipment):,}")

# Add censored observations for healthy equipment
for equip_id in healthy_equipment:
    # Get installation date
    equip_data = df_features[df_features['Ekipman_ID'] == equip_id].iloc[0]

    if 'Ekipman_Yaşı_Yıl' in equip_data and pd.notna(equip_data['Ekipman_Yaşı_Yıl']):
        age_years = equip_data['Ekipman_Yaşı_Yıl']
        time_to_event = int(age_years * 365)  # Convert years to days

        # Add as censored observation (event_occurred = 0)
        survival_records.append({
            'Ekipman_ID': equip_id,
            'Observation_Date': REFERENCE_DATE - timedelta(days=time_to_event),
            'Time_To_Event': time_to_event,
            'Event_Occurred': 0,  # Censored (no failure observed)
            'Failure_Number': 0
        })

# Recreate dataframe
df_survival = pd.DataFrame(survival_records)

print(f"\n✓ Total survival observations: {len(df_survival):,}")
print(f"  Events (failures): {df_survival['Event_Occurred'].sum():,} ({df_survival['Event_Occurred'].sum()/len(df_survival)*100:.1f}%)")
print(f"  Censored (healthy): {(df_survival['Event_Occurred'] == 0).sum():,} ({(df_survival['Event_Occurred'] == 0).sum()/len(df_survival)*100:.1f}%)")
```

---

### Phase 7: Update Configuration

**Modify: `config.py`**

**Add new configuration**:
```python
# ============================================================================
# HEALTHY EQUIPMENT DATA
# ============================================================================

# Input file for healthy equipment (NEW)
HEALTHY_EQUIPMENT_FILE = DATA_DIR / 'healthy_equipment.csv'

# Whether to include healthy equipment in training
INCLUDE_HEALTHY_EQUIPMENT = True

# Expected healthy equipment data columns
REQUIRED_HEALTHY_COLUMNS = [
    'cbs_id',  # Equipment ID
    'Şebeke Unsuru',  # Equipment Type
    'Sebekeye_Baglanma_Tarihi'  # Grid Connection Date
]

OPTIONAL_HEALTHY_COLUMNS = [
    'Enlem',  # Latitude
    'Boylam',  # Longitude
    'Musteri_Sayisi',  # Customer count
    'Trafo_Güç_kVA'  # Transformer capacity
]
```

---

## Data Requirements for Healthy Equipment

### Required Format

**File**: `data/healthy_equipment.csv`

**Required Columns**:
1. `cbs_id` (int) - Equipment ID (must match failed equipment ID system)
2. `Şebeke Unsuru` (str) - Equipment Type (e.g., "Ayırıcı", "Kesici", "OG/AG Trafo")
3. `Sebekeye_Baglanma_Tarihi` (date) - Grid Connection Date (for age calculation)

**Optional Columns** (improve model quality):
4. `Enlem` (float) - Latitude
5. `Boylam` (float) - Longitude
6. `Musteri_Sayisi` (int) - Customer count
7. `Trafo_Güç_kVA` (float) - Transformer capacity
8. Any other metadata columns

**Example CSV**:
```csv
cbs_id,Şebeke Unsuru,Sebekeye_Baglanma_Tarihi,Enlem,Boylam,Musteri_Sayisi
50001,Ayırıcı,2015-03-20,41.0082,28.9784,150
50002,Kesici,2018-07-12,41.0123,29.0012,200
50003,OG/AG Trafo,2012-11-05,40.9876,28.8956,500
```

### Data Quality Checks

Before providing healthy equipment data:

1. **ID Consistency**: Ensure `cbs_id` values do NOT overlap with failed equipment
2. **Equipment Type**: Use same categories as failed equipment (check `Şebeke Unsuru` column)
3. **Date Format**: Use `YYYY-MM-DD` or `DD-MM-YYYY` format
4. **Completeness**: At least 80% of required columns should have values
5. **Representativeness**: Healthy equipment should match distribution of failed equipment types

### Recommended Sample Size

**Target Ratio**: 1:1 to 2:1 (healthy:failed)

**Current Failed Equipment**: ~1,313 equipment
**Recommended Healthy Equipment**: 1,300 - 2,600 equipment

**Benefits**:
- 1:1 ratio → Balanced training
- 2:1 ratio → More realistic distribution (most equipment are healthy)

---

## Testing Strategy

### Phase 1: Unit Testing

Test each modified script independently:

```bash
# Test healthy equipment loader
python 02a_healthy_equipment_loader.py

# Test data transformation with merged data
python 02_data_transformation.py

# Test feature engineering with zero-fault equipment
python 03_feature_engineering.py

# Test temporal PoF with balanced data
python 06_temporal_pof_model.py
```

### Phase 2: Integration Testing

Run full pipeline end-to-end:

```bash
python run_pipeline.py
```

### Phase 3: Validation Checks

1. **Data Distribution**:
   - Check failed vs healthy equipment ratio
   - Verify target class balance (expect 10-30% positive class)

2. **Feature Values**:
   - Verify healthy equipment have 0 fault counts
   - Verify healthy equipment have NaN MTBF
   - Verify healthy equipment have age values

3. **Model Performance**:
   - Check AUC (expect 0.75-0.90)
   - Check precision/recall trade-off
   - Verify predictions range from 0-1

4. **Survival Analysis**:
   - Verify censored observations for healthy equipment
   - Check time-to-event distributions

---

## Expected Outcomes

### Model Performance Improvements

**Before** (Only Failed Equipment):
- AUC: 0.85-0.95 (overfitted, no true negatives)
- Precision: Low (many false positives)
- Recall: High (finds all failures but also flags healthy equipment)
- Calibration: Poor (probabilities not realistic)

**After** (Mixed Healthy + Failed):
- AUC: 0.75-0.88 (realistic, true negatives learned)
- Precision: High (fewer false positives)
- Recall: Moderate (better trade-off)
- Calibration: Good (probabilities reflect true risk)

### Risk Score Distribution

**Before**:
- Most equipment: 60-90% risk (all failed equipment)
- Few low-risk equipment

**After**:
- Healthy equipment: 5-30% risk (true negatives)
- At-risk equipment: 40-70% risk (moderate)
- High-risk equipment: 70-95% risk (true positives)
- Better distribution for CAPEX prioritization

---

## Implementation Timeline

| Phase | Task | Estimated Time |
|-------|------|----------------|
| **1** | Create healthy equipment loader script | 2 hours |
| **2** | Update data transformation | 2 hours |
| **3** | Update feature engineering | 2 hours |
| **4** | Update temporal PoF model | 3 hours |
| **5** | Update chronic classifier | 1 hour |
| **6** | Update survival analysis | 2 hours |
| **7** | Update configuration | 1 hour |
| **8** | Testing & validation | 3 hours |
| **9** | Documentation | 2 hours |
| **Total** | | **18 hours** |

---

## Next Steps

1. **User Action Required**:
   - ✅ Provide healthy equipment data file (`data/healthy_equipment.csv`)
   - ✅ Verify data format matches requirements above
   - ✅ Confirm healthy equipment count (target: 1,300-2,600)

2. **Implementation Order**:
   - Phase 1: Create healthy equipment loader
   - Phase 2-7: Update pipeline scripts sequentially
   - Phase 8: End-to-end testing
   - Phase 9: Document changes

3. **Validation Checkpoints**:
   - After Phase 3: Verify merged dataset structure
   - After Phase 4: Verify model training with balanced data
   - After Phase 6: Verify survival analysis with censoring
   - After Phase 8: Validate full pipeline output

---

## Questions for User

1. **Healthy Equipment Data Source**:
   - Where will healthy equipment data come from? (CBS system, manual selection?)
   - What date range should we consider? (active equipment as of cutoff date?)

2. **Healthy Equipment Definition**:
   - Should "healthy" mean: (a) zero failures ever, OR (b) no failures in last X months?
   - Recommended: No failures in last 12 months AND < 2 lifetime failures

3. **Sample Size**:
   - How many healthy equipment can you provide? (target: 1,300-2,600)
   - Should we match equipment type distribution? (recommended: yes)

4. **Geographic Coverage**:
   - Should healthy equipment come from same regions as failed equipment?
   - This helps control for environmental factors

---

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Healthy equipment data unavailable | HIGH | Start with equipment that have very old last failure (>24 months) as pseudo-healthy |
| ID mismatch between datasets | HIGH | Implement strict ID validation and mapping |
| Imbalanced equipment types | MEDIUM | Sample healthy equipment to match failed equipment type distribution |
| Feature calculation errors for zero-fault equipment | MEDIUM | Implement comprehensive null handling and safe defaults |
| Model performance degradation | LOW | Expected - initial drop in AUC is normal, improves generalization |

---

## References

- **COMPREHENSIVE_ROADMAP.md**: Overall pipeline improvements
- **DUAL_MODELING_ANALYSIS.md**: Modeling approach documentation
- **config.py**: Configuration parameters
- **02_data_transformation.py**: Current data loading logic
- **06_temporal_pof_model.py**: Current model training logic
- **10_survival_model.py**: Current survival analysis logic

---

**Status**: ✅ Plan Complete - Ready for User Data
**Next Action**: User provides `data/healthy_equipment.csv` → Start Phase 1 implementation
