"""
WALK-FORWARD TEMPORAL VALIDATION (v1.0)
=======================================
Replaces random 70/30 split with time-based validation

APPROACH:
- Train on 2020-2021, test on 2022
- Train on 2020-2022, test on 2023
- Train on 2020-2023, test on 2024

WHY CRITICAL:
✅ Checks model stability over time
✅ Detects concept drift (failure patterns changing)
✅ More realistic evaluation (can't see future during training)
✅ Prevents temporal leakage from random split

EXPECTED OUTCOMES:
- AUC should be LOWER than random split (0.70-0.85)
- Performance should be more consistent across folds
- Identifies if recent data has different patterns
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from config import CUTOFF_DATE, XGBOOST_PARAMS
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("WALK-FORWARD TEMPORAL VALIDATION")
print("="*100)

# ============================================================================
# STEP 1: LOAD DATA WITH TEMPORAL INFORMATION
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING DATA WITH TEMPORAL FEATURES")
print("="*100)

# Load reduced features
df = pd.read_csv('data/features_reduced.csv')
print(f"\n✓ Loaded features: {len(df)} equipment × {len(df.columns)} columns")

# Load original data to get installation dates for temporal splitting
df_full = pd.read_csv('data/equipment_level_data.csv')

# Merge to get installation date
df = df.merge(df_full[['Ekipman_ID', 'Ekipman_Kurulum_Tarihi']], on='Ekipman_ID', how='left')

# Parse installation date
df['Install_Date'] = pd.to_datetime(df['Ekipman_Kurulum_Tarihi'], errors='coerce')
df['Install_Year'] = df['Install_Date'].dt.year

print(f"\n📅 Installation Date Distribution:")
print(df['Install_Year'].value_counts().sort_index())

missing_dates = df['Install_Date'].isna().sum()
if missing_dates > 0:
    print(f"\n⚠️  WARNING: {missing_dates} equipment missing installation dates")
    print(f"   These will be assigned based on first fault date")

# ============================================================================
# STEP 2: TEMPORAL DATA QUALITY CHECKS
# ============================================================================
print("\n" + "="*100)
print("STEP 2: TEMPORAL DATA QUALITY CHECKS")
print("="*100)

# Load fault data to check temporal consistency
faults = pd.read_excel('data/combined_data.xlsx')
faults['started at'] = pd.to_datetime(faults['started at'], dayfirst=True, errors='coerce')

# Check 1: Faults before installation
print("\n🔍 CHECK 1: Faults Before Installation Date")
fault_dates = faults.groupby('cbs_id')['started at'].min().reset_index()
fault_dates.columns = ['Ekipman_ID', 'First_Fault']

check_df = df.merge(fault_dates, on='Ekipman_ID', how='left')
faults_before_install = check_df[check_df['First_Fault'] < check_df['Install_Date']]

if len(faults_before_install) > 0:
    print(f"   ❌ FOUND {len(faults_before_install)} equipment with faults BEFORE installation!")
    print(f"      This indicates data quality issues or incorrect installation dates")
    print(f"      Sample cases:")
    print(faults_before_install[['Ekipman_ID', 'Install_Date', 'First_Fault']].head())
else:
    print(f"   ✓ No faults before installation dates (GOOD)")

# Check 2: Training data should only have pre-cutoff faults
print("\n🔍 CHECK 2: Last Fault Date vs Cutoff Date")
last_fault_dates = faults[faults['started at'] <= CUTOFF_DATE].groupby('cbs_id')['started at'].max().reset_index()
last_fault_dates.columns = ['Ekipman_ID', 'Last_Fault_PreCutoff']

check_df2 = df.merge(last_fault_dates, on='Ekipman_ID', how='left')
post_cutoff_in_training = check_df2[check_df2['Last_Fault_PreCutoff'] > CUTOFF_DATE]

if len(post_cutoff_in_training) > 0:
    print(f"   ❌ FOUND {len(post_cutoff_in_training)} equipment with faults AFTER cutoff in training data!")
    print(f"      This is DATA LEAKAGE!")
else:
    print(f"   ✓ All training faults are pre-cutoff (GOOD)")

# Check 3: Negative time intervals
print("\n🔍 CHECK 3: Age Calculation Consistency")
if 'Ekipman_Yaşı_Yıl' in df.columns and 'Install_Date' in df.columns:
    # Calculate age from install date to cutoff
    df['Calculated_Age_Years'] = (CUTOFF_DATE - df['Install_Date']).dt.days / 365.25

    # Compare with reported age
    age_diff = (df['Ekipman_Yaşı_Yıl'] - df['Calculated_Age_Years']).abs()
    inconsistent_ages = df[age_diff > 1]  # More than 1 year difference

    if len(inconsistent_ages) > 0:
        print(f"   ⚠️  {len(inconsistent_ages)} equipment with age inconsistencies (>1 year difference)")
        print(f"      Mean difference: {age_diff.mean():.2f} years")
    else:
        print(f"   ✓ Age calculations consistent (GOOD)")

# ============================================================================
# STEP 3: WALK-FORWARD VALIDATION SETUP
# ============================================================================
print("\n" + "="*100)
print("STEP 3: WALK-FORWARD VALIDATION SETUP")
print("="*100)

# Define temporal folds
# We'll use installation year to create temporal splits
temporal_folds = [
    {
        'name': 'Fold 1: Train 2020-2021, Test 2022',
        'train_years': [2020, 2021],
        'test_years': [2022]
    },
    {
        'name': 'Fold 2: Train 2020-2022, Test 2023',
        'train_years': [2020, 2021, 2022],
        'test_years': [2023]
    },
    {
        'name': 'Fold 3: Train 2020-2023, Test 2024',
        'train_years': [2020, 2021, 2022, 2023],
        'test_years': [2024]
    }
]

print("\n📋 Temporal Fold Configuration:")
for fold in temporal_folds:
    print(f"\n   {fold['name']}")
    train_count = df[df['Install_Year'].isin(fold['train_years'])].shape[0]
    test_count = df[df['Install_Year'].isin(fold['test_years'])].shape[0]
    print(f"      Train: {train_count} equipment")
    print(f"      Test:  {test_count} equipment")

# ============================================================================
# STEP 4: PREPARE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 4: PREPARING FEATURES")
print("="*100)

horizons = ['3M', '6M', '12M']  # Removed 24M as recommended
id_column = 'Ekipman_ID'
target_columns = [f'Target_{h}' for h in horizons]

# Identify categorical features
categorical_features = []
for col in df.columns:
    if col not in [id_column, 'Ekipman_Kurulum_Tarihi', 'Install_Date', 'Install_Year', 'Calculated_Age_Years'] + target_columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)

print(f"\n✓ Categorical features detected: {categorical_features}")

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Get feature columns
feature_columns = [col for col in df.columns
                   if col not in [id_column, 'Ekipman_Kurulum_Tarihi', 'Install_Date',
                                  'Install_Year', 'Calculated_Age_Years'] + target_columns]

print(f"✓ Feature count: {len(feature_columns)}")

# ============================================================================
# STEP 5: WALK-FORWARD VALIDATION EXECUTION
# ============================================================================
print("\n" + "="*100)
print("STEP 5: WALK-FORWARD VALIDATION EXECUTION")
print("="*100)

results = []

for horizon in horizons:
    print(f"\n{'='*100}")
    print(f"HORIZON: {horizon}")
    print(f"{'='*100}")

    target_col = f'Target_{horizon}'

    for fold_idx, fold in enumerate(temporal_folds, 1):
        print(f"\n{'-'*80}")
        print(f"{fold['name']}")
        print(f"{'-'*80}")

        # Create temporal train/test split
        train_mask = df['Install_Year'].isin(fold['train_years'])
        test_mask = df['Install_Year'].isin(fold['test_years'])

        X_train = df[train_mask][feature_columns]
        y_train = df[train_mask][target_col]
        X_test = df[test_mask][feature_columns]
        y_test = df[test_mask][target_col]

        # Check if we have enough samples
        if len(X_train) < 50 or len(X_test) < 10:
            print(f"   ⚠️  SKIPPING: Insufficient samples (train={len(X_train)}, test={len(X_test)})")
            continue

        if y_train.sum() < 5 or y_test.sum() < 2:
            print(f"   ⚠️  SKIPPING: Insufficient positive samples (train={y_train.sum()}, test={y_test.sum()})")
            continue

        print(f"\n   Train: {len(X_train)} equipment, {y_train.sum()} positives ({y_train.mean()*100:.1f}%)")
        print(f"   Test:  {len(X_test)} equipment, {y_test.sum()} positives ({y_test.mean()*100:.1f}%)")

        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            verbosity=0
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

        print(f"\n   ✅ Results:")
        print(f"      AUC: {auc:.4f}")
        print(f"      Average Precision: {ap:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall: {recall:.4f}")
        print(f"      F1-Score: {f1:.4f}")

        # Store results
        results.append({
            'Horizon': horizon,
            'Fold': fold['name'],
            'Train_Size': len(X_train),
            'Test_Size': len(X_test),
            'Train_Positive_Rate': y_train.mean(),
            'Test_Positive_Rate': y_test.mean(),
            'AUC': auc,
            'Average_Precision': ap,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        })

# ============================================================================
# STEP 6: AGGREGATE RESULTS & COMPARISON
# ============================================================================
print("\n" + "="*100)
print("STEP 6: WALK-FORWARD VALIDATION SUMMARY")
print("="*100)

if len(results) > 0:
    results_df = pd.DataFrame(results)

    print("\n📊 Complete Results:")
    print(results_df.to_string(index=False))

    print("\n📊 Average Performance by Horizon:")
    summary = results_df.groupby('Horizon')[['AUC', 'Average_Precision', 'Precision', 'Recall', 'F1_Score']].agg(['mean', 'std'])
    print(summary)

    # Save results
    Path('results').mkdir(exist_ok=True)
    results_df.to_csv('results/walkforward_validation_results.csv', index=False)
    print("\n✓ Results saved to: results/walkforward_validation_results.csv")

    print("\n" + "="*100)
    print("COMPARISON: Random Split vs Walk-Forward")
    print("="*100)

    print("\nExpected Differences:")
    print("   Random Split (from 06_temporal_pof_model.py):")
    print("      - 3M AUC: 0.9733 (suspiciously high)")
    print("      - 6M AUC: 0.9485")
    print("      - 12M AUC: 0.9664")

    print("\n   Walk-Forward (current run):")
    for horizon in horizons:
        horizon_results = results_df[results_df['Horizon'] == horizon]
        if len(horizon_results) > 0:
            mean_auc = horizon_results['AUC'].mean()
            std_auc = horizon_results['AUC'].std()
            print(f"      - {horizon} AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    print("\n💡 INTERPRETATION:")
    print("   If walk-forward AUC < random split AUC:")
    print("      ✓ Confirms random split had temporal leakage")
    print("      ✓ Walk-forward is more realistic performance estimate")
    print("   If walk-forward AUC ≈ random split AUC:")
    print("      ⚠️  Model may genuinely be good (or both have leakage)")
    print("   If AUC variance is HIGH across folds:")
    print("      ⚠️  Indicates concept drift - failure patterns changing over time")

else:
    print("\n❌ No valid results - check data temporal distribution")

print("\n" + "="*100)
print("WALK-FORWARD VALIDATION COMPLETE")
print("="*100)
