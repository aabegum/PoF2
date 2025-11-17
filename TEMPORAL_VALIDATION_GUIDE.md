# Temporal Transformation Validation Guide
## Fix for Critical Temporal Leakage Issue

---

## Problem Statement

**CRITICAL BUG IDENTIFIED**: The original pipeline (`02_data_transformation.py`) calculated features using ALL available data, including the target prediction period. This causes **temporal leakage** - the model "sees the future" during training.

### Original (WRONG) Approach:
```python
# 02_data_transformation.py (LEAKY)
reference_date = df['started at'].max()  # Uses latest fault date (2025-06-25)
cutoff_3m = reference_date - pd.Timedelta(days=90)
df['Fault_Last_3M'] = (df['started at'] >= cutoff_3m).astype(int)

# Problem: If we're predicting 2024-2025, features include data from 2024-2025!
# Model will have unrealistically high accuracy that won't generalize
```

### New (CORRECT) Approach:
```python
# 02_data_transformation_temporal.py (NO LEAKAGE)
cutoff_date = pd.Timestamp('2024-06-25')  # User-specified training cutoff
historical_data = df[df['started at'] <= cutoff_date]  # ONLY historical data
df['Fault_Last_3M'] = (historical_data['started at'] >= cutoff_3m).astype(int)

# Target from FUTURE data:
target_data = df[df['started at'] > cutoff_date]
```

---

## Usage Instructions

### Basic Usage

```bash
# Create training data with cutoff at 2024-06-25
python 02_data_transformation_temporal.py --cutoff_date 2024-06-25

# Predict 6 months ahead instead of default 12 months
python 02_data_transformation_temporal.py --cutoff_date 2024-06-25 --prediction_horizon 180

# Add custom suffix to output filename
python 02_data_transformation_temporal.py --cutoff_date 2024-06-25 --output_suffix "_2024Q2"
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--cutoff_date` | **YES** | None | Cutoff date (YYYY-MM-DD). Features use ONLY data before this date. |
| `--prediction_horizon` | No | 365 | Days after cutoff to predict (e.g., 365 = 12 months) |
| `--output_suffix` | No | `_YYYYMMDD` | Custom suffix for output files (e.g., `_2024Q2`) |

---

## Walk-Forward Validation

To properly validate the model with **no temporal leakage**, create multiple training windows:

### Example: 3 Training Windows

```bash
# Window 1: Train on 2021-2023, predict 2023-2024
python 02_data_transformation_temporal.py \
    --cutoff_date 2023-06-25 \
    --prediction_horizon 365 \
    --output_suffix "_window1_2023"

# Window 2: Train on 2021-2024H1, predict 2024H1-2024H2
python 02_data_transformation_temporal.py \
    --cutoff_date 2024-01-01 \
    --prediction_horizon 365 \
    --output_suffix "_window2_2024Q1"

# Window 3: Train on 2021-2024H1, predict 2024H2-2025H2
python 02_data_transformation_temporal.py \
    --cutoff_date 2024-06-25 \
    --prediction_horizon 365 \
    --output_suffix "_window3_2024Q2"
```

**Purpose**: Each window tests whether the model generalizes to truly unseen future data.

### Recommended Validation Strategy

1. **Production Deployment Window** (Most recent)
   - Cutoff: Latest safe date (e.g., 2024-06-25)
   - Use: Train final production model

2. **Validation Window** (6 months before latest)
   - Cutoff: 2024-01-01
   - Use: Validate performance on known outcomes (2024-2025)

3. **Historical Window** (1 year before latest)
   - Cutoff: 2023-06-25
   - Use: Additional validation on older time period

---

## Output Validation Checklist

After running the temporal transformation script, **ALWAYS verify** the following:

### ✅ 1. Check Temporal Filter Console Output

Look for this in the console output:
```
================================================================================
STEP 1B: TEMPORAL FILTERING (NO LEAKAGE)
================================================================================

Filtering faults to BEFORE cutoff date (2024-06-25)...
  Historical faults (for features): 4,523 (72.3%)
  Future faults (excluded): 1,732 (27.7%)

✓ TEMPORAL FILTER APPLIED
  Features will be calculated from 4,523 historical faults only
  Reference date for all calculations: 2024-06-25
```

**Verify**:
- Historical faults percentage makes sense (should be 60-80% for 12M horizon)
- Future faults are explicitly excluded

---

### ✅ 2. Check Target Variable Creation

Look for this section:
```
================================================================================
STEP 13: CREATING TARGET VARIABLE (FROM FUTURE DATA)
================================================================================

Target Period: 2024-06-26 to 2025-06-25
  Faults in target period: 1,732

✓ Target Variable Created:
  Equipment with failures in target period: 287 (25.0%)
  Equipment without failures: 861 (75.0%)
  Total target faults: 1,732
  Avg faults per failing equipment: 6.03
```

**Verify**:
- Target period is AFTER cutoff date
- Failure rate (25%) is realistic (typical: 15-35%)
- No overlap between historical and target faults

---

### ✅ 3. Inspect Output Files

Three files are created:

**A. Main Output**:
```
data/equipment_level_data_temporal_20240625.csv
```

Load and check key columns:
```python
import pandas as pd
df = pd.read_csv('data/equipment_level_data_temporal_20240625.csv')

# Check temporal feature columns
print("\nTemporal Features (all relative to cutoff):")
print(df[['Arıza_Sayısı_3ay', 'Arıza_Sayısı_6ay', 'Arıza_Sayısı_12ay']].describe())

# Check target variable
print("\nTarget Variable (from future period):")
print(df['Target_Failure_Binary'].value_counts())
print(df['Target_Failure_Count'].describe())

# CRITICAL CHECK: Verify no impossible values
# (e.g., equipment with 0 historical faults but target=1 is OK)
# (e.g., Arıza_Sayısı_12ay > 0 should exist)
```

**B. Metadata File**:
```json
// data/equipment_level_data_temporal_20240625_metadata.json
{
  "cutoff_date": "2024-06-25",
  "prediction_horizon_days": 365,
  "target_start": "2024-06-26",
  "target_end": "2025-06-25",
  "historical_faults": 4523,
  "target_faults": 1732,
  "equipment_count": 1148,
  "failure_rate": 0.25,
  "features": 54
}
```

**Verify**:
- `target_start` > `cutoff_date` (no overlap)
- `failure_rate` is reasonable (0.15 - 0.35)
- `historical_faults` + `target_faults` = total faults in dataset

**C. Feature Documentation**:
```
data/feature_documentation_temporal_20240625.csv
```

Check for expected columns:
- `Arıza_Sayısı_3ay`, `Arıza_Sayısı_6ay`, `Arıza_Sayısı_12ay` (temporal features)
- `Ekipman_Yaşı_Yıl` (age as of cutoff)
- `Son_Arıza_Gun_Sayisi` (days since last failure as of cutoff)
- `Target_Failure_Binary`, `Target_Failure_Count` (future targets)

---

### ✅ 4. Validate Feature Values

**Critical validation queries**:

```python
import pandas as pd
df = pd.read_csv('data/equipment_level_data_temporal_20240625.csv')

# 1. Check age is calculated as of cutoff date
# (Should NOT show equipment with age > max possible given cutoff)
print("Age distribution (as of 2024-06-25):")
print(df['Ekipman_Yaşı_Yıl'].describe())
# All ages should be <= (2024 - earliest_install_year)

# 2. Check failure counts are from historical period only
print("\nFailure count distribution (historical):")
print(df[['Arıza_Sayısı_3ay', 'Arıza_Sayısı_6ay', 'Arıza_Sayısı_12ay']].describe())
# Should have reasonable counts (not impossibly high)

# 3. Check target variable is from future period
print("\nTarget variable (future):")
print(df['Target_Failure_Binary'].value_counts())
# Should have mix of 0s and 1s

# 4. CRITICAL: Check no equipment has target=1 but Arıza_Sayısı_12ay=0
# (This is ALLOWED - equipment can fail for the first time in target period)
# But most equipment with target=1 should have Arıza_Sayísı_12ay > 0
print("\nCross-check: Historical vs Future failures")
print(pd.crosstab(
    df['Arıza_Sayısı_12ay'] > 0,
    df['Target_Failure_Binary'],
    margins=True
))
# Expect high overlap but not 100%
```

---

### ✅ 5. Compare with Original (Leaky) Output

If you still have the original output from `02_data_transformation.py`:

```python
# Load both datasets
df_leaky = pd.read_csv('data/equipment_level_data.csv')  # Original (leaky)
df_temporal = pd.read_csv('data/equipment_level_data_temporal_20240625.csv')  # Fixed

# Compare failure counts (should be LOWER in temporal version)
print("Feature comparison:")
print(f"Leaky - Arıza_Sayısı_12ay mean: {df_leaky['Arıza_Sayısı_12ay'].mean():.2f}")
print(f"Temporal - Arıza_Sayısı_12ay mean: {df_temporal['Arıza_Sayısı_12ay'].mean():.2f}")

# Temporal should have LOWER counts (fewer months of data)
# This proves the temporal filter is working
```

---

## Model Training with Temporal Data

Once validated, use the temporal data for training:

### Option 1: Direct Training (Single Window)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load temporal data
df = pd.read_csv('data/equipment_level_data_temporal_20240625.csv')

# Separate features and target
X = df.drop(['Ekipman_ID', 'Target_Failure_Binary', 'Target_Failure_Count'], axis=1)
y = df['Target_Failure_Binary']

# Split (can use random split since temporal filtering already done)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate (this is TRULY out-of-sample performance)
print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")
```

### Option 2: Walk-Forward Validation (Multiple Windows)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Load all windows
windows = [
    'data/equipment_level_data_temporal_window1_2023.csv',
    'data/equipment_level_data_temporal_window2_2024Q1.csv',
    'data/equipment_level_data_temporal_window3_2024Q2.csv',
]

results = []

for i, window_path in enumerate(windows):
    df = pd.read_csv(window_path)

    X = df.drop(['Ekipman_ID', 'Target_Failure_Binary', 'Target_Failure_Count'], axis=1)
    y = df['Target_Failure_Binary']

    # Train on window i, test on window i+1
    if i < len(windows) - 1:
        df_next = pd.read_csv(windows[i+1])
        X_test = df_next.drop(['Ekipman_ID', 'Target_Failure_Binary', 'Target_Failure_Count'], axis=1)
        y_test = df_next['Target_Failure_Binary']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        results.append({
            'Window': f'{i} -> {i+1}',
            'AUC': auc
        })

print("\nWalk-Forward Validation Results:")
for r in results:
    print(f"  {r['Window']}: AUC = {r['AUC']:.3f}")
```

---

## Troubleshooting

### Issue 1: "Only X historical faults available"

**Symptom**:
```
⚠️  WARNING: Only 523 historical faults available!
   Consider using an earlier cutoff date for more training data.
```

**Solution**: Use an earlier cutoff date or reduce prediction horizon:
```bash
# Option A: Earlier cutoff (more training data)
python 02_data_transformation_temporal.py --cutoff_date 2023-06-25

# Option B: Shorter prediction horizon (keeps more historical data)
python 02_data_transformation_temporal.py --cutoff_date 2024-06-25 --prediction_horizon 180
```

---

### Issue 2: Failure rate too low/high

**Symptom**: Target failure rate is 5% (too low) or 60% (too high)

**Diagnosis**:
```python
# Check metadata
import json
with open('data/equipment_level_data_temporal_20240625_metadata.json') as f:
    meta = json.load(f)
    print(f"Failure rate: {meta['failure_rate']*100:.1f}%")
    print(f"Historical faults: {meta['historical_faults']}")
    print(f"Target faults: {meta['target_faults']}")
```

**Solution**:
- **Too low** (<10%): Increase prediction horizon (e.g., 365 -> 730 days)
- **Too high** (>40%): Decrease prediction horizon (e.g., 365 -> 180 days) or use earlier cutoff

---

### Issue 3: Missing equipment in output

**Symptom**: Original data has 1,500 equipment but output only has 1,000

**Cause**: Equipment with no faults before cutoff date are excluded

**Solution**: This is EXPECTED behavior. Equipment with no historical faults cannot have features calculated. To include them:

1. Add equipment master list (separate file with all equipment IDs)
2. Left join temporal output with master list
3. Fill missing feature values with 0 (no failures = 0 faults)

---

## Success Criteria

Your temporal transformation is **VALIDATED** if:

✅ **Console output shows**:
- "TEMPORAL FILTER APPLIED"
- Historical faults < 100% (future faults are excluded)
- Target period starts AFTER cutoff date

✅ **Output files contain**:
- `Target_Failure_Binary` and `Target_Failure_Count` columns
- Failure rate between 15-35%
- Feature counts (e.g., `Arıza_Sayısı_12ay`) are reasonable (not impossibly high)

✅ **Validation checks pass**:
- Feature values are lower than original (leaky) version
- No overlap between historical and target data
- Age calculated as of cutoff date

✅ **Model performance**:
- Performance on temporal data is LOWER than leaky data (this is EXPECTED and GOOD)
- Performance is consistent across walk-forward windows (generalization)

---

## Next Steps

After successful validation:

1. **Update feature engineering** (`03_feature_engineering.py`):
   - Accept `--input` parameter to use temporal output
   - Ensure all features respect cutoff date (no future data)

2. **Retrain feature selection** (`05_feature_selection.py`):
   - Use temporal data (not leaky data)
   - Adjust VIF threshold to 10 (current: 5 is too strict)

3. **Create production pipeline**:
   - Automate monthly temporal splits
   - Retrain model with latest cutoff date
   - Deploy for field operations

4. **Deploy Phase 1 (Rule-Based) immediately**:
   - Use risk scores from `03_feature_engineering.py`
   - Generate work orders for high-risk equipment
   - Don't wait for ML model fixes

---

**Document Version**: 1.0
**Created**: 2025-11-17
**Author**: Data Analytics Team
**Related Issue**: Temporal leakage fix (critical blocker for ML production)
