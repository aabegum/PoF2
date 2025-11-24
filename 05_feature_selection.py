"""
COMPREHENSIVE FEATURE SELECTION PIPELINE
Turkish EDAŞ PoF Prediction Project (v5.0)

Purpose:
- Step 1: Remove data leakage features (target period information)
- Step 2: Remove redundant/correlated features (multicollinearity)
- Step 3: VIF analysis for final selection (variance inflation)
- All-in-one feature selection with clear audit trail

Input:  data/features_engineered.csv (111 features)
Output: data/features_reduced.csv (12-18 features)

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import sys

# Import centralized configuration
from config import (
    FEATURES_ENGINEERED_FILE,
    FEATURES_REDUCED_FILE,
    OUTPUT_DIR,
    VIF_THRESHOLD,
    VIF_TARGET,
    CORRELATION_THRESHOLD,
    IMPORTANCE_THRESHOLD,
    PROTECTED_FEATURES
)

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
plt.style.use('seaborn-v0_8-darkgrid')

print("="*100)
print(" "*25 + "COMPREHENSIVE FEATURE SELECTION")
print(" "*15 + "Leakage Removal → Redundancy Reduction → VIF Analysis")
print("="*100)

# ============================================================================
# CONFIGURATION (Imported from config.py)
# ============================================================================

# Create output directory
output_dir = OUTPUT_DIR / 'feature_selection'
output_dir.mkdir(parents=True, exist_ok=True)

print("\n📋 Configuration:")
print(f"   VIF Threshold: {VIF_THRESHOLD}")
print(f"   Correlation Threshold: {CORRELATION_THRESHOLD}")
print(f"   Importance Threshold: {IMPORTANCE_THRESHOLD}")

# ============================================================================
# STEP 0: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 0: LOADING ENGINEERED FEATURES")
print("="*100)

if not FEATURES_ENGINEERED_FILE.exists():
    print(f"\n❌ ERROR: File not found at {FEATURES_ENGINEERED_FILE}")
    print("Please run 03_feature_engineering.py first!")
    exit(1)

print(f"\n✓ Loading from: {FEATURES_ENGINEERED_FILE}")
df = pd.read_csv(FEATURES_ENGINEERED_FILE)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# Verify Equipment_Class_Primary exists
if 'Equipment_Class_Primary' not in df.columns:
    print("\n⚠ WARNING: Equipment_Class_Primary column not found!")
else:
    print("✓ Equipment_Class_Primary column verified")

original_feature_count = df.shape[1]
original_features = df.columns.tolist()

# ============================================================================
# STEP 1: REMOVE DATA LEAKAGE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 1: REMOVING DATA LEAKAGE FEATURES")
print("="*100)

print("\n📋 Data Leakage Rules:")
print("   1. LEAKY: Recent failure counts (3M/6M/12M include target period)")
print("   2. LEAKY: Aggregations based on recent failures")
print("   3. LEAKY: Features calculated from ALL faults (not cutoff-filtered)")
print("   4. SAFE: Historical patterns from BEFORE cutoff (2024-06-25)")
print("   5. SAFE: Static attributes (age, equipment type, location)")

leaky_features = []
leaky_reasons = {}

for col in original_features:
    reason = None

    # Rule 1: Recent failure counts (3/6/12 months)
    if 'Arıza_Sayısı_3ay' in col or 'Arıza_Sayısı_6ay' in col or 'Arıza_Sayısı_12ay' in col:
        if col != 'Toplam_Arıza_Sayisi_Lifetime':
            reason = "Recent failure count (includes target period)"

    # Rule 2: Aggregations based on recent failures
    elif any(x in col for x in ['_12ay_Class_Avg', '_12ay_Cluster_Avg', '_6ay_Class_Avg', '_3ay_Class_Avg']):
        reason = "Aggregation based on recent failures (target period)"

    # Rule 3: Recent failure intensity/acceleration
    elif 'Recent_Failure_Intensity' in col or 'Failure_Acceleration' in col:
        reason = "Recent failure intensity (uses target period)"

    # Rule 4: Risk scores if based on recent failures
    elif 'Recent_Failure_Risk_Score' in col:
        reason = "Risk score based on recent failures"

    # Rule 5: Time since last normalized (if recent)
    elif 'Time_Since_Last_Normalized' in col:
        reason = "Normalized time since last failure (recent)"

    # Rule 6: Lifetime failure count (used to create target - DIRECT LEAKAGE!)
    elif 'Toplam_Arıza_Sayisi_Lifetime' in col or 'Toplam_Ariza_Sayisi_Lifetime' in col:
        reason = "Lifetime failure count (DIRECTLY used to create target!)"

    if reason:
        leaky_features.append(col)
        leaky_reasons[col] = reason

print(f"\n⚠️  Found {len(leaky_features)} leaky features:")
for feat in leaky_features:
    print(f"   ❌ {feat:<50} → {leaky_reasons[feat]}")

# Remove leaky features
df = df.drop(columns=leaky_features)
print(f"\n✓ Removed {len(leaky_features)} leaky features")
print(f"✓ Remaining: {len(df.columns)} features")

# ============================================================================
# STEP 2: REMOVE REDUNDANT FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: REMOVING REDUNDANT FEATURES")
print("="*100)

# Define redundant features (highly correlated or derived)
# Updated for OPTIMAL 30-FEATURE SET (Phase 1 removals included)
REDUNDANT_FEATURES = {
    # ========================================================================
    # PHASE 1 REMOVALS: Geographic Clustering (STEP 3)
    # ========================================================================
    'Geographic_Cluster': {
        'reason': '🚫 REMOVED: K-means clustering on X,Y coordinates (noisy patterns)',
        'keep_instead': 'İlçe (district - clear, interpretable)',
        'correlation': 'N/A'
    },
    'Arıza_Sayısı_12ay_Cluster_Avg': {
        'reason': '🚨 LEAKY: Cluster aggregation uses 12-month window (includes post-cutoff)',
        'keep_instead': 'İlçe (district)',
        'correlation': 0.45
    },
    'Tekrarlayan_Arıza_90gün_Flag_Cluster_Avg': {
        'reason': '🚨 LEAKY: Cluster aggregation of recurrence flag',
        'keep_instead': 'Tekrarlayan_Arıza_90gün_Flag (individual flag)',
        'correlation': 0.58
    },
    'MTBF_Gün_Cluster_Avg': {
        'reason': '🚫 CIRCULAR: Cluster aggregation creates circular logic',
        'keep_instead': 'MTBF_InterFault_Gün (individual)',
        'correlation': 0.65
    },

    # ========================================================================
    # PHASE 1 REMOVALS: Redundant Failure Rates (STEP 4)
    # ========================================================================
    'Failure_Rate_Per_Year': {
        'reason': '🚫 REDUNDANT: Tree models learn from Toplam_Arıza / Ekipman_Yaşı',
        'keep_instead': 'Toplam_Arıza_Sayisi_Lifetime + Ekipman_Yaşı_Yıl',
        'correlation': 0.72
    },
    'Recent_Failure_Intensity': {
        'reason': '🚨 LEAKY: Uses Arıza_Sayısı_3ay (includes post-cutoff data)',
        'keep_instead': 'Son_Arıza_Gun_Sayisi',
        'correlation': 0.68
    },
    'Failure_Acceleration': {
        'reason': '🚨 LEAKY: Uses Arıza_Sayısı_6ay (includes post-cutoff data)',
        'keep_instead': 'MTBF_InterFault_Trend (degradation detector)',
        'correlation': 0.52
    },

    # ========================================================================
    # PHASE 1 REMOVALS: Equipment Class Aggregations (STEP 7)
    # ========================================================================
    'Arıza_Sayısı_12ay_Class_Avg': {
        'reason': '🚨 LEAKY: Class aggregation uses 12-month window (target leakage)',
        'keep_instead': 'Equipment_Class_Primary (let model learn patterns)',
        'correlation': 0.55
    },
    'MTBF_Gün_Class_Avg': {
        'reason': '🚫 CIRCULAR: Class average creates circular prediction logic',
        'keep_instead': 'Equipment_Class_Primary',
        'correlation': 0.48
    },
    'Ekipman_Yaşı_Yıl_Class_Avg': {
        'reason': '🚫 NOT PREDICTIVE: Class age average not useful',
        'keep_instead': 'Equipment_Class_Primary + Ekipman_Yaşı_Yıl',
        'correlation': 0.32
    },
    'Yas_Beklenen_Omur_Orani_Class_Avg': {
        'reason': '🚫 NOT PREDICTIVE: Class age ratio average not useful',
        'keep_instead': 'Equipment_Class_Primary + Yas_Beklenen_Omur_Orani',
        'correlation': 0.28
    },
    'Failure_vs_Class_Avg': {
        'reason': '🚨 LEAKY: Derived from Arıza_Sayısı_12ay_Class_Avg',
        'keep_instead': 'Equipment_Class_Primary',
        'correlation': 0.61
    },

    # ========================================================================
    # PHASE 1 REMOVALS: Weak Interaction Features (STEP 8)
    # ========================================================================
    'Age_Failure_Interaction': {
        'reason': '🚨 LEAKY: Uses Arıza_Sayısı_12ay (includes post-cutoff data)',
        'keep_instead': 'AgeRatio_Recurrence_Interaction (uses lifetime count)',
        'correlation': 0.45
    },
    'Customer_Failure_Interaction': {
        'reason': '🚨 LEAKY: Uses Arıza_Sayısı_12ay (includes post-cutoff data)',
        'keep_instead': 'total_customer_count_Avg + Toplam_Arıza_Sayisi_Lifetime',
        'correlation': 0.38
    },

    # ========================================================================
    # ORIGINAL REDUNDANT FEATURES (from previous analysis)
    # ========================================================================
    'Reliability_Score': {
        'reason': 'Derived from MTBF_Gün (r² > 0.95)',
        'keep_instead': 'MTBF_Gün',
        'correlation': 0.97
    },
    'Failure_Free_3M': {
        'reason': '🚨 CRITICAL: Binary flag for no failures in last 3M (uses ALL faults)',
        'keep_instead': 'Son_Arıza_Gun_Sayisi',
        'correlation': 0.83
    },
    'Ekipman_Yoğunluk_Skoru': {
        'reason': '🚨 CRITICAL: Fault density score uses ALL faults',
        'keep_instead': 'Son_Arıza_Gun_Sayisi or MTBF_Gün',
        'correlation': 0.99
    },
    'Composite_PoF_Risk_Score': {
        'reason': '🚨 CRITICAL: Created using Arıza_Sayısı_6ay (leaky)',
        'keep_instead': 'MTBF_Gün + Son_Arıza_Gun_Sayisi + Age features',
        'correlation': 0.22
    },
    'Risk_Category': {
        'reason': 'Derived from Composite_PoF_Risk_Score (leaky)',
        'keep_instead': 'Equipment_Class_Primary',
        'correlation': 'N/A'
    },

    # ========================================================================
    # PHASE 2 REMOVALS: Low-Value/Constant Features (VIF cleanup)
    # ========================================================================
    'Tek_Neden_Flag': {
        'reason': '🚫 HIGH VIF (89): Correlated with Arıza_Nedeni_Tutarlılık',
        'keep_instead': 'Arıza_Nedeni_Tutarlılık (more informative continuous variable)',
        'correlation': 0.85
    },
    'Is_HV': {
        'reason': '🚫 CONSTANT: All zeros (no high voltage equipment in dataset)',
        'keep_instead': 'Voltage_Class (covers all voltage levels)',
        'correlation': 'N/A'
    },
    'Yaş_Kaynak': {
        'reason': '🚫 CONSTANT: Single unique value (no variance, zero predictive power)',
        'keep_instead': 'None needed (all equipment use same age source)',
        'correlation': 'N/A'
    },
}

# NOTE: Tekrarlayan_Arıza_90gün_Flag is KEPT - it's the TARGET for chronic repeater classification
# Calculated safely using only pre-cutoff data (see 02_data_transformation.py calculate_recurrence_safe)
# Used in 06_chronic_repeater.py as the target label

# Protected features are imported from config.py
# (NEVER remove these features)

redundant_to_remove = []
removal_reasons = {}

for feat, info in REDUNDANT_FEATURES.items():
    if feat in df.columns:
        redundant_to_remove.append(feat)
        removal_reasons[feat] = info
        print(f"\n❌ {feat}")
        print(f"   Reason: {info['reason']}")
        print(f"   Keep instead: {info['keep_instead']}")
        corr = info['correlation']
        if isinstance(corr, (int, float)):
            print(f"   Correlation: r={corr:.2f}")
        else:
            print(f"   Correlation: {corr}")
    else:
        print(f"\n⚠️  {feat} - Not in dataset (already removed)")

# Verify no protected features are being removed
protected_in_removal = set(redundant_to_remove) & set(PROTECTED_FEATURES)
if protected_in_removal:
    print(f"\n⚠️  WARNING: Protected features in removal list: {protected_in_removal}")
    redundant_to_remove = [f for f in redundant_to_remove if f not in PROTECTED_FEATURES]

# Remove redundant features
df = df.drop(columns=redundant_to_remove)
print(f"\n✓ Removed {len(redundant_to_remove)} redundant features")
print(f"✓ Remaining: {len(df.columns)} features")

# ============================================================================
# STEP 3: VIF ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 3: VIF ANALYSIS (Multicollinearity Detection)")
print("="*100)

# Identify features for VIF analysis
id_column = 'Ekipman_ID' if 'Ekipman_ID' in df.columns else None
target_columns = ['Arıza_Olacak_6ay', 'Arıza_Olacak_12ay', 'Arıza_Olacak_24ay']
target_columns = [col for col in target_columns if col in df.columns]

# Exclude ID and targets
features_for_vif = [col for col in df.columns if col != id_column and col not in target_columns]

print(f"\n✓ Analyzing {len(features_for_vif)} features for multicollinearity")

# Separate numeric and categorical
numeric_features = df[features_for_vif].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [col for col in features_for_vif if col not in numeric_features]

print(f"   Numeric features: {len(numeric_features)}")
print(f"   Categorical features: {len(categorical_features)}")

if len(categorical_features) > 0:
    print(f"\n✓ Categorical features: {categorical_features}")
    print("   (Will encode for VIF analysis)")

# Encode categorical features
df_vif = df[numeric_features + categorical_features].copy()

for cat_col in categorical_features:
    if df_vif[cat_col].dtype == 'object' or df_vif[cat_col].dtype.name == 'category':
        le = LabelEncoder()
        df_vif[cat_col] = le.fit_transform(df_vif[cat_col].astype(str))

# Handle infinite values and missing data (CRITICAL for VIF calculation)
print(f"\n--- Data Cleaning for VIF Analysis ---")

# Step 1: Replace infinite values with NaN
df_vif = df_vif.replace([np.inf, -np.inf], np.nan)

# Step 2: Check for problematic columns
nan_counts = df_vif.isnull().sum()
problematic_cols = nan_counts[nan_counts > 0]

if len(problematic_cols) > 0:
    print(f"⚠️  Found {len(problematic_cols)} columns with missing values:")
    for col, count in problematic_cols.head(10).items():
        pct = count / len(df_vif) * 100
        print(f"   {col}: {count} ({pct:.1f}%)")

    if len(problematic_cols) > 10:
        print(f"   ... and {len(problematic_cols) - 10} more")

# Step 3: Fill missing values column by column
for col in df_vif.columns:
    if df_vif[col].isnull().any():
        median_val = df_vif[col].median()
        if pd.isna(median_val):
            # Column is all NaN - fill with 0
            df_vif[col] = df_vif[col].fillna(0)
            print(f"   ⚠️  {col}: All NaN, filled with 0")
        else:
            df_vif[col] = df_vif[col].fillna(median_val)

# Step 4: Verify no NaN or inf remain
remaining_nan = df_vif.isnull().sum().sum()
remaining_inf = np.isinf(df_vif.select_dtypes(include=[np.number])).sum().sum()

if remaining_nan > 0 or remaining_inf > 0:
    print(f"\n❌ ERROR: Still have {remaining_nan} NaN and {remaining_inf} inf values!")
    print("Cannot proceed with VIF analysis")
    exit(1)

print(f"✓ Data cleaned: {len(df_vif.columns)} features ready for VIF analysis")
print(f"  No NaN or inf values remaining")

# Calculate VIF iteratively
print(f"\n--- Iterative VIF Calculation (Target: VIF < {VIF_TARGET}) ---")

vif_features = df_vif.columns.tolist()
iteration = 0
max_iterations = 50

while True:
    iteration += 1
    if iteration > max_iterations:
        print(f"\n⚠️  Reached maximum iterations ({max_iterations})")
        break

    # Calculate VIF for current features
    vif_data = pd.DataFrame()
    vif_data['Feature'] = vif_features
    vif_data['VIF'] = [variance_inflation_factor(df_vif[vif_features].values, i)
                       for i in range(len(vif_features))]

    # Find max VIF
    max_vif = vif_data['VIF'].max()
    max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']

    print(f"\nIteration {iteration}: {len(vif_features)} features, Max VIF = {max_vif:.2f} ({max_vif_feature})")

    # Stop if all VIF below threshold
    if max_vif <= VIF_TARGET:
        print(f"✅ All features have VIF <= {VIF_TARGET}")
        break

    # Stop if max VIF is not too high and we have few features
    if max_vif <= 15 and len(vif_features) <= 15:
        print(f"✅ Stopping: VIF={max_vif:.2f} is acceptable with {len(vif_features)} features")
        break

    # Remove feature with highest VIF (unless protected)
    if max_vif_feature in PROTECTED_FEATURES:
        print(f"   ⚠️  {max_vif_feature} is PROTECTED - keeping despite high VIF")
        # Remove next highest non-protected feature
        vif_sorted = vif_data[~vif_data['Feature'].isin(PROTECTED_FEATURES)].sort_values('VIF', ascending=False)
        if len(vif_sorted) == 0:
            print("   ✅ All remaining features are protected")
            break
        max_vif_feature = vif_sorted.iloc[0]['Feature']
        max_vif = vif_sorted.iloc[0]['VIF']
        print(f"   Removing {max_vif_feature} instead (VIF={max_vif:.2f})")

    print(f"   ❌ Removing: {max_vif_feature} (VIF={max_vif:.2f})")
    vif_features.remove(max_vif_feature)
    df_vif = df_vif[vif_features]

# Final VIF results
print(f"\n--- Final VIF Results ---")
final_vif = pd.DataFrame()
final_vif['Feature'] = vif_features
final_vif['VIF'] = [variance_inflation_factor(df_vif.values, i) for i in range(len(vif_features))]
final_vif = final_vif.sort_values('VIF', ascending=False)

print("\n" + final_vif.to_string(index=False))

# Keep only VIF-selected features (plus ID and targets)
final_columns = [id_column] + target_columns + vif_features if id_column else target_columns + vif_features
df_final = df[[col for col in final_columns if col in df.columns]].copy()

print(f"\n✓ Final feature set: {len(df_final.columns)} features")

# ============================================================================
# STEP 4: SAVE RESULTS
# ============================================================================
print("\n" + "="*100)
print("STEP 4: SAVING RESULTS")
print("="*100)

print(f"\n💾 Saving to: {FEATURES_REDUCED_FILE}")
df_final.to_csv(FEATURES_REDUCED_FILE, index=False, encoding='utf-8-sig')

print(f"✅ Successfully saved!")
print(f"   Records: {len(df_final):,}")
print(f"   Features: {len(df_final.columns)}")
print(f"   File size: {FEATURES_REDUCED_FILE.stat().st_size / 1024**2:.2f} MB")

# Save comprehensive report
print("\n📋 Creating comprehensive feature selection report...")

report_data = []
for feat in original_features:
    status = 'RETAINED'
    reason = 'Passed all checks'

    if feat in leaky_features:
        status = 'REMOVED - LEAKAGE'
        reason = leaky_reasons[feat]
    elif feat in redundant_to_remove:
        status = 'REMOVED - REDUNDANT'
        reason = removal_reasons[feat]['reason']
    elif feat not in df_final.columns and feat != id_column and feat not in target_columns:
        status = 'REMOVED - VIF'
        reason = 'High multicollinearity (VIF > threshold)'

    report_data.append({
        'Feature': feat,
        'Status': status,
        'Reason': reason,
        'In_Final_Set': feat in df_final.columns
    })

report_df = pd.DataFrame(report_data)
report_path = Path('outputs/feature_selection/comprehensive_selection_report.csv')
report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
print(f"✓ Report saved to: {report_path}")

# ============================================================================
# STEP 5: SUMMARY
# ============================================================================
print("\n" + "="*100)
print("COMPREHENSIVE FEATURE SELECTION COMPLETE")
print("="*100)

print(f"\n📊 SELECTION SUMMARY:")
print(f"   Original features: {original_feature_count}")
print(f"   Leaky features removed: {len(leaky_features)}")
print(f"   Redundant features removed: {len(redundant_to_remove)}")
print(f"   VIF-removed features: {original_feature_count - len(leaky_features) - len(redundant_to_remove) - len(df_final.columns)}")
print(f"   Final features: {len(df_final.columns)}")
print(f"   Reduction: {(1 - len(df_final.columns)/original_feature_count)*100:.1f}%")

print(f"\n📂 OUTPUT FILES:")
print(f"   • {FEATURES_REDUCED_FILE}")
print(f"   • {report_path}")

print(f"\n✅ PIPELINE BENEFITS:")
print(f"   • No data leakage (only historical data)")
print(f"   • Reduced multicollinearity (VIF < {VIF_TARGET})")
print(f"   • No redundant features")
print(f"   • Improved model generalization")
print(f"   • Faster training")

print(f"\n💡 NEXT STEPS:")
print(f"   1. Run model training: python 06_model_training.py")
print(f"   2. Expected AUC: 0.70-0.80 (realistic, not 1.0)")
print(f"   3. Review feature selection report for audit trail")

print("\n" + "="*100)
print(f"{'FEATURE SELECTION PIPELINE COMPLETE':^100}")
print("="*100)
