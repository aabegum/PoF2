"""
DIAGNOSTIC: FIND REMAINING LEAKY FEATURES
Identify which features in features_reduced.csv still have data leakage

This script:
1. Loads features_reduced.csv (21 features)
2. Loads temporal targets (6M and 12M)
3. Calculates correlation between each feature and targets
4. Identifies features with suspiciously high correlation (r > 0.80)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print(" "*35 + "DIAGNOSTIC: FIND LEAKY FEATURES")
print(" "*25 + "Identify Features Causing AUC = 1.0")
print("="*100)

# Load reduced features
print("\n✓ Loading features_reduced.csv...")
df = pd.read_csv('data/features_reduced.csv')
print(f"  Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# Load temporal targets
print("\n✓ Loading temporal targets...")
CUTOFF_DATE = pd.Timestamp('2024-06-25')
FUTURE_6M_END = CUTOFF_DATE + pd.DateOffset(months=6)
FUTURE_12M_END = CUTOFF_DATE + pd.DateOffset(months=12)

# Load fault data
all_faults = pd.read_excel('data/combined_data.xlsx')
all_faults['started at'] = pd.to_datetime(all_faults['started at'], dayfirst=True, errors='coerce')

# Create targets
future_faults_6M = all_faults[
    (all_faults['started at'] > CUTOFF_DATE) &
    (all_faults['started at'] <= FUTURE_6M_END)
]['cbs_id'].dropna().unique()

future_faults_12M = all_faults[
    (all_faults['started at'] > CUTOFF_DATE) &
    (all_faults['started at'] <= FUTURE_12M_END)
]['cbs_id'].dropna().unique()

df['Target_6M'] = df['Ekipman_ID'].isin(future_faults_6M).astype(int)
df['Target_12M'] = df['Ekipman_ID'].isin(future_faults_12M).astype(int)

print(f"  6M target positive rate: {df['Target_6M'].mean()*100:.1f}%")
print(f"  12M target positive rate: {df['Target_12M'].mean()*100:.1f}%")

# Get numeric features only
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in ['Ekipman_ID', 'Target_6M', 'Target_12M']]

print(f"\n✓ Analyzing {len(numeric_features)} numeric features...")

# Calculate correlations
print("\n" + "="*100)
print("CORRELATION ANALYSIS")
print("="*100)

correlations_6M = []
correlations_12M = []

for feat in numeric_features:
    corr_6M = df[[feat, 'Target_6M']].corr().iloc[0, 1]
    corr_12M = df[[feat, 'Target_12M']].corr().iloc[0, 1]
    correlations_6M.append((feat, abs(corr_6M), corr_6M))
    correlations_12M.append((feat, abs(corr_12M), corr_12M))

# Sort by absolute correlation
correlations_6M.sort(key=lambda x: x[1], reverse=True)
correlations_12M.sort(key=lambda x: x[1], reverse=True)

# Display top correlations for 6M
print("\n--- Top 10 Feature Correlations with 6M Target ---")
print(f"{'Feature':<50} {'|r|':<10} {'r':<10} {'Status'}")
print("-"*100)

for feat, abs_corr, corr in correlations_6M[:10]:
    if abs_corr > 0.80:
        status = "🚨 CRITICAL LEAKAGE!"
    elif abs_corr > 0.60:
        status = "⚠️  HIGH - Investigate"
    elif abs_corr > 0.40:
        status = "⚠️  MODERATE"
    else:
        status = "✅ Normal"

    print(f"{feat:<50} {abs_corr:<10.4f} {corr:<10.4f} {status}")

# Display top correlations for 12M
print("\n--- Top 10 Feature Correlations with 12M Target ---")
print(f"{'Feature':<50} {'|r|':<10} {'r':<10} {'Status'}")
print("-"*100)

for feat, abs_corr, corr in correlations_12M[:10]:
    if abs_corr > 0.80:
        status = "🚨 CRITICAL LEAKAGE!"
    elif abs_corr > 0.60:
        status = "⚠️  HIGH - Investigate"
    elif abs_corr > 0.40:
        status = "⚠️  MODERATE"
    else:
        status = "✅ Normal"

    print(f"{feat:<50} {abs_corr:<10.4f} {corr:<10.4f} {status}")

# Identify critical leaky features
print("\n" + "="*100)
print("CRITICAL FINDINGS")
print("="*100)

leaky_6M = [feat for feat, abs_corr, _ in correlations_6M if abs_corr > 0.80]
leaky_12M = [feat for feat, abs_corr, _ in correlations_12M if abs_corr > 0.80]

all_leaky = list(set(leaky_6M + leaky_12M))

if all_leaky:
    print(f"\n🚨 CRITICAL: Found {len(all_leaky)} leaky feature(s):")
    for feat in all_leaky:
        corr_6M = next((corr for f, _, corr in correlations_6M if f == feat), 0)
        corr_12M = next((corr for f, _, corr in correlations_12M if f == feat), 0)
        print(f"\n   ❌ {feat}")
        print(f"      6M correlation: r={corr_6M:.4f}")
        print(f"      12M correlation: r={corr_12M:.4f}")
        print(f"      → This feature MUST be removed or recalculated without future data!")
else:
    print("\n✅ No critical leakage found (all |r| < 0.80)")
    print("   The high AUC may be due to complex feature interactions.")
    print("   Try training with even fewer features (top 10-15 only).")

# Save correlation report
report_data = []
for feat, abs_corr, corr in correlations_6M:
    corr_12M = next((c for f, _, c in correlations_12M if f == feat), 0)
    report_data.append({
        'Feature': feat,
        'Corr_6M': corr,
        'AbsCorr_6M': abs_corr,
        'Corr_12M': corr_12M,
        'AbsCorr_12M': abs(corr_12M),
        'MaxAbsCorr': max(abs_corr, abs(corr_12M)),
        'IsLeaky': max(abs_corr, abs(corr_12M)) > 0.80
    })

report_df = pd.DataFrame(report_data)
report_df = report_df.sort_values('MaxAbsCorr', ascending=False)

output_path = Path('outputs/feature_selection/correlation_diagnostic.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
report_df.to_csv(output_path, index=False)

print(f"\n💾 Detailed correlation report saved to: {output_path}")

print("\n" + "="*100)
print(" "*35 + "DIAGNOSTIC COMPLETE")
print("="*100)

if all_leaky:
    print("\n🔧 RECOMMENDED ACTIONS:")
    print(f"   1. Add these {len(all_leaky)} feature(s) to REDUNDANT_FEATURES in 05c_reduce_feature_redundancy.py")
    print(f"   2. Re-run: python 05c_reduce_feature_redundancy.py")
    print(f"   3. Re-run: python 06_model_training.py")
    print(f"   4. Expected: AUC should drop to 0.75-0.85")
else:
    print("\n🔧 RECOMMENDED ACTIONS:")
    print(f"   1. Review correlation_diagnostic.csv for features with |r| > 0.60")
    print(f"   2. Manually inspect how these features are calculated")
    print(f"   3. Verify they don't use data from after 2024-06-25")

print("\n" + "="*100)
