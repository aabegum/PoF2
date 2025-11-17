"""
FEATURE REDUNDANCY REDUCTION
Turkish EDAŞ PoF Prediction Project (v4.0)

Purpose:
- Remove highly correlated features to reduce overfitting
- Keep only ONE feature from each correlated group
- Improve model generalization to new equipment
- Reduce multicollinearity while preserving predictive power

Strategy:
- Remove Reliability_Score (keep MTBF_Gün - more interpretable)
- Remove Failure_Rate_Per_Year (highly correlated with failure counts)
- Remove cluster averages that duplicate individual features
- Remove Tekrarlayan_Arıza_90gün_Flag (data leakage - uses ALL faults)
- Remove Failure_Free_3M (data leakage - uses ALL faults)
- Keep interpretable, business-critical features

Input:  data/features_selected_clean.csv (26 features)
Output: data/features_reduced.csv (20 features)

Author: Data Analytics Team
Date: 2025
Version: 4.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*100)
print(" "*25 + "FEATURE REDUNDANCY REDUCTION")
print(" "*20 + "Remove Correlated Features | Improve Generalization")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Features to remove (redundant with other features)
REDUNDANT_FEATURES = {
    # Reliability Metrics (Keep MTBF_Gün, remove Reliability_Score)
    'Reliability_Score': {
        'reason': 'Derived from MTBF_Gün (r² > 0.95)',
        'keep_instead': 'MTBF_Gün',
        'correlation': 0.97
    },

    # Failure Rate (Keep failure counts, remove rate)
    'Failure_Rate_Per_Year': {
        'reason': 'Correlated with Son_Arıza_Gun_Sayisi and failure counts',
        'keep_instead': 'Son_Arıza_Gun_Sayisi',
        'correlation': 0.72
    },

    # Cluster averages that duplicate individual features
    'MTBF_Gün_Cluster_Avg': {
        'reason': 'Aggregation of MTBF_Gün (individual feature more important)',
        'keep_instead': 'MTBF_Gün',
        'correlation': 0.65
    },

    'Tekrarlayan_Arıza_90gün_Flag_Cluster_Avg': {
        'reason': 'Aggregation of chronic repeater flag (also calculated from future data)',
        'keep_instead': 'None (removed due to leakage)',
        'correlation': 0.58
    },

    # 🚨 DATA LEAKAGE: Chronic repeater flag calculated from FULL dataset
    'Tekrarlayan_Arıza_90gün_Flag': {
        'reason': '🚨 CRITICAL: Calculated using ALL faults (includes future failures after 2024-06-25)',
        'keep_instead': 'None (use in 06_chronic_repeater.py separately)',
        'correlation': 'N/A (causes AUC=1.0 data leakage)'
    },

    # 🚨 DATA LEAKAGE: Failure-free 3M flag calculated from FULL dataset
    'Failure_Free_3M': {
        'reason': '🚨 CRITICAL: Binary flag for no failures in last 3M (uses ALL faults including after 2024-06-25)',
        'keep_instead': 'Son_Arıza_Gun_Sayisi (days since last failure)',
        'correlation': 0.83  # r=-0.8281 with 12M target (inverse correlation)
    },
}

# Protected features (NEVER remove, even if correlated)
PROTECTED_FEATURES = [
    'Ekipman_ID',                      # ID column
    # NOTE: Tekrarlayan_Arıza_90gün_Flag REMOVED (data leakage - see REDUNDANT_FEATURES)
    # NOTE: Failure_Free_3M REMOVED (data leakage - see REDUNDANT_FEATURES)
    'MTBF_Gün',                        # Primary reliability metric
    'Son_Arıza_Gun_Sayisi',            # Recency - critical for temporal PoF
    'Composite_PoF_Risk_Score',        # Stakeholder communication
    'Ilk_Arizaya_Kadar_Yil',          # Time to first failure
    'Ekipman_Yaşı_Yıl_EDBS_first',    # Equipment age
    'Equipment_Class_Primary',         # Equipment type
    'Geographic_Cluster',              # Location
]

print("\n📋 Configuration:")
print(f"   Redundant features to remove: {len(REDUNDANT_FEATURES)}")
print(f"   Protected features (never remove): {len(PROTECTED_FEATURES)}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING SELECTED FEATURES")
print("="*100)

data_path = Path('data/features_selected_clean.csv')

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("Please run 05b_remove_leaky_features.py first!")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

original_features = df.columns.tolist()

# ============================================================================
# STEP 2: IDENTIFY REDUNDANT FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: IDENTIFYING REDUNDANT FEATURES")
print("="*100)

print("\n--- Redundant Features Analysis ---")

features_to_remove = []
removal_reasons = {}

for feat, info in REDUNDANT_FEATURES.items():
    if feat in df.columns:
        features_to_remove.append(feat)
        removal_reasons[feat] = info
        print(f"\n❌ {feat}")
        print(f"   Reason: {info['reason']}")
        print(f"   Keep instead: {info['keep_instead']}")
        # Handle both numeric and string correlation values
        corr = info['correlation']
        if isinstance(corr, (int, float)):
            print(f"   Correlation: r={corr:.2f}")
        else:
            print(f"   Correlation: {corr}")
    else:
        print(f"\n⚠️  {feat} - Not in dataset (already removed)")

print(f"\n📊 Summary:")
print(f"   Features to remove: {len(features_to_remove)}")
print(f"   Features to keep: {len(original_features) - len(features_to_remove)}")

# ============================================================================
# STEP 3: CALCULATE FEATURE CORRELATIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 3: CALCULATING FEATURE CORRELATIONS")
print("="*100)

# Get numeric features only
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID column from correlation
numeric_features = [f for f in numeric_features if f != 'Ekipman_ID']

print(f"\n✓ Calculating correlations for {len(numeric_features)} numeric features...")

# Calculate correlation matrix
corr_matrix = df[numeric_features].corr().abs()

# Find high correlations (excluding self-correlations)
high_corr_pairs = []
threshold = 0.85

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > threshold:
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            high_corr_pairs.append((feat1, feat2, corr_val))

if high_corr_pairs:
    print(f"\n⚠️  High Correlation Pairs Found (r > {threshold}):")
    for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
        removed = feat1 in features_to_remove or feat2 in features_to_remove
        status = "✅ One will be removed" if removed else "⚠️  Both kept (review)"
        print(f"   {feat1} ↔ {feat2}: r={corr_val:.3f} | {status}")
else:
    print(f"\n✅ No high correlations found (all < {threshold})")

# ============================================================================
# STEP 4: REMOVE REDUNDANT FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 4: REMOVING REDUNDANT FEATURES")
print("="*100)

# Verify no protected features are being removed
protected_in_removal = set(features_to_remove) & set(PROTECTED_FEATURES)
if protected_in_removal:
    print(f"\n⚠️  WARNING: Protected features in removal list: {protected_in_removal}")
    print("   Removing from removal list...")
    features_to_remove = [f for f in features_to_remove if f not in PROTECTED_FEATURES]

# Create reduced dataframe
df_reduced = df.drop(columns=features_to_remove)

print(f"\n--- Features Removed ---")
for feat in features_to_remove:
    info = removal_reasons[feat]
    print(f"   ❌ {feat:<45} (Keep: {info['keep_instead']})")

print(f"\n✓ Removed {len(features_to_remove)} redundant features")
print(f"✓ Retained {len(df_reduced.columns)} features")

# ============================================================================
# STEP 5: VALIDATE REDUCED FEATURE SET
# ============================================================================
print("\n" + "="*100)
print("STEP 5: VALIDATING REDUCED FEATURE SET")
print("="*100)

# Verify all protected features are present
missing_protected = set(PROTECTED_FEATURES) - set(df_reduced.columns)
if missing_protected:
    print(f"\n⚠️  WARNING: Missing protected features: {missing_protected}")
else:
    print(f"\n✅ All {len(PROTECTED_FEATURES)} protected features present")

# Categorize remaining features
print(f"\n--- Remaining Features by Category ---")

# Get feature categories
id_features = [col for col in df_reduced.columns if 'ID' in col or col == 'Ekipman_ID']
age_features = [col for col in df_reduced.columns if 'Yaş' in col or 'Age' in col or 'Ömür' in col]
historical_features = [col for col in df_reduced.columns if 'Son_Arıza' in col or 'Last_Failure' in col or 'Recurrence' in col or 'MTBF' in col or 'Ilk_Ariza' in col]
location_features = [col for col in df_reduced.columns if 'Geographic' in col or 'Cluster' in col or 'urban' in col or 'suburban' in col or 'rural' in col]
equipment_features = [col for col in df_reduced.columns if 'Equipment' in col or 'Ekipman' in col or 'Class' in col]
customer_features = [col for col in df_reduced.columns if 'Customer' in col or 'Müşteri' in col or 'Peak' in col]
risk_features = [col for col in df_reduced.columns if 'Risk' in col or 'Score' in col or 'Flag' in col]

print(f"\n1. ID Features ({len(id_features)}): {id_features}")
print(f"2. Age/Life Features ({len(age_features)}): {age_features}")
print(f"3. Historical Failure Features ({len(historical_features)}): {historical_features}")
print(f"4. Location Features ({len(location_features)}): {location_features}")
print(f"5. Equipment Type Features ({len(equipment_features)}): {equipment_features}")
print(f"6. Customer Impact Features ({len(customer_features)}): {customer_features}")
print(f"7. Risk/Score Features ({len(risk_features)}): {risk_features}")

# ============================================================================
# STEP 6: SAVE REDUCED FEATURE SET
# ============================================================================
print("\n" + "="*100)
print("STEP 6: SAVING REDUCED FEATURE SET")
print("="*100)

output_path = Path('data/features_reduced.csv')
print(f"\n💾 Saving to: {output_path}")
df_reduced.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ Successfully saved!")
print(f"   Records: {len(df_reduced):,}")
print(f"   Features: {len(df_reduced.columns)}")
print(f"   File size: {output_path.stat().st_size / 1024**2:.2f} MB")

# Save feature reduction report
print("\n📋 Creating feature reduction report...")

report_data = []
for feat in original_features:
    is_removed = feat in features_to_remove
    report_data.append({
        'Feature': feat,
        'Status': 'REMOVED' if is_removed else 'RETAINED',
        'Reason': removal_reasons[feat]['reason'] if is_removed else 'Not redundant',
        'Keep_Instead': removal_reasons[feat]['keep_instead'] if is_removed else '',
        'In_Reduced_Set': not is_removed
    })

report_df = pd.DataFrame(report_data)
report_path = Path('outputs/feature_selection/feature_reduction_report.csv')
report_path.parent.mkdir(parents=True, exist_ok=True)
report_df.to_csv(report_path, index=False)
print(f"✓ Feature reduction report saved to: {report_path}")

# ============================================================================
# STEP 7: SUMMARY
# ============================================================================
print("\n" + "="*100)
print("FEATURE REDUNDANCY REDUCTION COMPLETE")
print("="*100)

print(f"\n📊 REDUCTION SUMMARY:")
print(f"   Original features: {len(original_features)}")
print(f"   Redundant features removed: {len(features_to_remove)}")
print(f"   Final features: {len(df_reduced.columns)}")
print(f"   Reduction: {len(features_to_remove)/len(original_features)*100:.1f}%")

print(f"\n📂 OUTPUT FILES:")
print(f"   • {output_path} ({len(df_reduced.columns)} features)")
print(f"   • {report_path} (detailed analysis)")

print(f"\n✅ BENEFITS:")
print(f"   • Reduced multicollinearity")
print(f"   • Improved model generalization")
print(f"   • Faster training (fewer features)")
print(f"   • Clearer feature importance rankings")
print(f"   • Better interpretability for stakeholders")

print(f"\n💡 NEXT STEPS:")
print(f"   1. Review feature reduction report")
print(f"   2. Re-run model training with reduced features:")
print(f"      • python 06_model_training.py (use data/features_reduced.csv)")
print(f"      • python 06_chronic_repeater.py (use data/features_reduced.csv)")
print(f"   3. Compare AUC: Should be slightly lower but more realistic")
print(f"   4. Expected improvement: Better generalization to new equipment")

print(f"\n⚠️  EXPECTED CHANGES:")
print(f"   • AUC may drop by 0.02-0.05 (this is GOOD - means less overfitting)")
print(f"   • Feature importance rankings will be clearer")
print(f"   • Model will perform better on new/unseen equipment")

print("\n" + "="*100)
print(f"{'FEATURE REDUNDANCY REDUCTION COMPLETE':^100}")
print("="*100)
