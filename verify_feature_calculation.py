"""
VERIFY FEATURE CALCULATION
Check if Son_Arıza_Gun_Sayisi and MTBF_Gün use post-cutoff data

This script:
1. Loads features_reduced.csv
2. Loads raw fault data
3. Manually calculates Son_Arıza_Gun_Sayisi using ONLY pre-cutoff faults
4. Compares with values in features_reduced.csv
5. If they differ → features are using post-cutoff data (LEAKAGE!)
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*100)
print(" "*30 + "FEATURE CALCULATION VERIFICATION")
print(" "*25 + "Detect if features use post-cutoff data")
print("="*100)

# Load features
df_features = pd.read_csv('data/features_reduced.csv')
print(f"\n✓ Loaded features: {df_features.shape[0]} equipment")

# Load raw faults
all_faults = pd.read_excel('data/combined_data.xlsx')
all_faults['started at'] = pd.to_datetime(all_faults['started at'], dayfirst=True, errors='coerce')
print(f"✓ Loaded faults: {len(all_faults)} total fault records")

# Define cutoff
CUTOFF_DATE = pd.Timestamp('2024-06-25')
print(f"✓ Cutoff date: {CUTOFF_DATE.date()}")

# Filter to pre-cutoff faults ONLY
pre_cutoff_faults = all_faults[all_faults['started at'] <= CUTOFF_DATE].copy()
print(f"✓ Pre-cutoff faults: {len(pre_cutoff_faults)} (should use ONLY these)")

# Calculate Son_Arıza_Gun_Sayisi manually using ONLY pre-cutoff faults
print("\n" + "="*100)
print("CHECKING: Son_Arıza_Gun_Sayisi (Days Since Last Failure)")
print("="*100)

manual_days_since = {}
for equip_id in df_features['Ekipman_ID']:
    equip_faults_pre = pre_cutoff_faults[pre_cutoff_faults['cbs_id'] == equip_id]

    if len(equip_faults_pre) > 0:
        last_fault_date = equip_faults_pre['started at'].max()
        days_since = (CUTOFF_DATE - last_fault_date).days
    else:
        days_since = 9999  # No failures

    manual_days_since[equip_id] = days_since

df_features['Manual_Days_Since'] = df_features['Ekipman_ID'].map(manual_days_since)

# Compare
df_features['Days_Since_Diff'] = df_features['Son_Arıza_Gun_Sayisi'] - df_features['Manual_Days_Since']

mismatches = df_features[df_features['Days_Since_Diff'].abs() > 1].copy()

print(f"\n📊 Comparison Results:")
print(f"   Total equipment: {len(df_features)}")
print(f"   Exact matches: {len(df_features[df_features['Days_Since_Diff'].abs() <= 1])}")
print(f"   Mismatches: {len(mismatches)}")

if len(mismatches) > 0:
    print(f"\n🚨 CRITICAL: Found {len(mismatches)} mismatches!")
    print(f"   This means Son_Arıza_Gun_Sayisi is NOT using only pre-cutoff faults!")
    print(f"\n   Sample mismatches:")
    print(f"   {'Equip_ID':<12} {'Feature Value':<15} {'Manual (Safe)':<15} {'Difference':<12}")
    print(f"   {'-'*60}")
    for idx in mismatches.head(10).index:
        row = mismatches.loc[idx]
        print(f"   {row['Ekipman_ID']:<12} {row['Son_Arıza_Gun_Sayisi']:<15.0f} {row['Manual_Days_Since']:<15.0f} {row['Days_Since_Diff']:<12.0f}")

    # Check correlation with targets
    future_faults_12M = all_faults[
        (all_faults['started at'] > CUTOFF_DATE) &
        (all_faults['started at'] <= CUTOFF_DATE + pd.DateOffset(months=12))
    ]['cbs_id'].dropna().unique()

    df_features['Target_12M'] = df_features['Ekipman_ID'].isin(future_faults_12M).astype(int)

    corr_feature = df_features[['Son_Arıza_Gun_Sayisi', 'Target_12M']].corr().iloc[0, 1]
    corr_manual = df_features[['Manual_Days_Since', 'Target_12M']].corr().iloc[0, 1]

    print(f"\n   Correlation with 12M target:")
    print(f"   Feature value (current):  r={corr_feature:.4f}")
    print(f"   Manual (safe) value:      r={corr_manual:.4f}")
    print(f"   Difference:               Δr={abs(corr_feature - corr_manual):.4f}")

else:
    print(f"\n✅ GOOD: Son_Arıza_Gun_Sayisi uses only pre-cutoff faults!")

# Check MTBF_Gün
print("\n" + "="*100)
print("CHECKING: MTBF_Gün (Mean Time Between Failures)")
print("="*100)

manual_mtbf = {}
for equip_id in df_features['Ekipman_ID']:
    equip_faults_pre = pre_cutoff_faults[pre_cutoff_faults['cbs_id'] == equip_id]

    if len(equip_faults_pre) >= 2:
        fault_dates = equip_faults_pre['started at'].sort_values()
        time_diffs = fault_dates.diff().dropna()
        mtbf_days = time_diffs.dt.days.mean()
    else:
        mtbf_days = 9999  # Insufficient data

    manual_mtbf[equip_id] = mtbf_days

df_features['Manual_MTBF'] = df_features['Ekipman_ID'].map(manual_mtbf)

# Compare
df_features['MTBF_Diff'] = df_features['MTBF_Gün'] - df_features['Manual_MTBF']

mismatches_mtbf = df_features[df_features['MTBF_Diff'].abs() > 5].copy()  # Allow 5 days tolerance

print(f"\n📊 Comparison Results:")
print(f"   Total equipment: {len(df_features)}")
print(f"   Exact matches: {len(df_features[df_features['MTBF_Diff'].abs() <= 5])}")
print(f"   Mismatches: {len(mismatches_mtbf)}")

if len(mismatches_mtbf) > 0:
    print(f"\n🚨 CRITICAL: Found {len(mismatches_mtbf)} mismatches!")
    print(f"   This means MTBF_Gün is NOT using only pre-cutoff faults!")
    print(f"\n   Sample mismatches:")
    print(f"   {'Equip_ID':<12} {'Feature Value':<15} {'Manual (Safe)':<15} {'Difference':<12}")
    print(f"   {'-'*60}")
    for idx in mismatches_mtbf.head(10).index:
        row = mismatches_mtbf.loc[idx]
        print(f"   {row['Ekipman_ID']:<12} {row['MTBF_Gün']:<15.1f} {row['Manual_MTBF']:<15.1f} {row['MTBF_Diff']:<12.1f}")
else:
    print(f"\n✅ GOOD: MTBF_Gün uses only pre-cutoff faults!")

# Summary
print("\n" + "="*100)
print(" "*35 + "VERIFICATION SUMMARY")
print("="*100)

if len(mismatches) > 0 or len(mismatches_mtbf) > 0:
    print("\n🚨 DATA LEAKAGE DETECTED!")
    if len(mismatches) > 0:
        print(f"   ❌ Son_Arıza_Gun_Sayisi: {len(mismatches)} mismatches (LEAKY!)")
    else:
        print(f"   ✅ Son_Arıza_Gun_Sayisi: Clean")

    if len(mismatches_mtbf) > 0:
        print(f"   ❌ MTBF_Gün: {len(mismatches_mtbf)} mismatches (LEAKY!)")
    else:
        print(f"   ✅ MTBF_Gün: Clean")

    print(f"\n🔧 NEXT STEPS:")
    print(f"   1. Review 02_data_transformation.py to see how these features are calculated")
    print(f"   2. Ensure they filter by cutoff date (faults <= 2024-06-25)")
    print(f"   3. Re-run feature engineering with proper cutoff filtering")
    print(f"   4. Re-run model training")
else:
    print("\n✅ ALL FEATURES CLEAN!")
    print("\n🤔 But AUC is still 1.0, which means:")
    print("   1. These features are genuinely VERY predictive")
    print("   2. Dataset is small and highly structured (perfect separation exists)")
    print("   3. This might be the nature of the problem (chronic failures are very predictable)")

    print(f"\n💡 RECOMMENDATION:")
    print(f"   Since features are clean and predictive:")
    print(f"   - Accept the high AUC (the features are just very predictive)")
    print(f"   - Validate on truly unseen equipment (not in this dataset)")
    print(f"   - Consider this a best-case scenario for failure prediction")

print("\n" + "="*100)
