"""
MODEL PERFORMANCE DIAGNOSTIC AUDIT
==================================
Investigates suspiciously high AUC (0.94-0.97) to identify:
1. Data leakage in features
2. Overfitting due to small sample size
3. Feature importance for each horizon
4. Calibration curves
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load data and models
print("="*100)
print("MODEL PERFORMANCE DIAGNOSTIC AUDIT")
print("="*100)

# Load reduced features (the features used for training)
df = pd.read_csv('data/features_reduced.csv')
print(f"\n✓ Loaded features: {len(df)} equipment × {len(df.columns)} columns")

# Load original equipment-level data (with ALL features for leakage check)
df_all = pd.read_csv('data/equipment_level_data.csv')
print(f"✓ Loaded full data: {len(df_all)} equipment × {len(df_all.columns)} columns")

# Merge to get removed features
df_merged = df.merge(df_all[['Ekipman_ID', 'Toplam_Arıza_Sayisi_Lifetime',
                               'Arıza_Sayısı_3ay', 'Arıza_Sayısı_6ay', 'Arıza_Sayısı_12ay']],
                     on='Ekipman_ID', how='left')

horizons = ['3M', '6M', '12M', '24M']

# ============================================================================
# DIAGNOSTIC 1: CHECK FOR LEAKAGE IN TOP FEATURES
# ============================================================================
print("\n" + "="*100)
print("DIAGNOSTIC 1: FEATURE IMPORTANCE & LEAKAGE CHECK")
print("="*100)

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"{horizon} HORIZON FEATURE IMPORTANCE")
    print(f"{'='*80}")

    # Load model
    model_path = f'models/temporal_pof_{horizon}.pkl'
    if not Path(model_path).exists():
        print(f"⚠️  Model not found: {model_path}")
        print(f"   Expected at: {model_path}")
        print(f"   Run 06_temporal_pof_model.py to train models first")
        continue

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Get feature importance
    importance = model.feature_importances_
    feature_names = [col for col in df.columns if col not in ['Ekipman_ID'] + [f'Target_{h}' for h in horizons]]

    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Top 15 features
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))

    # Check for leakage red flags
    print("\n🔍 Leakage Red Flags:")
    top_10 = importance_df.head(10)['feature'].tolist()

    leaky_indicators = {
        'Son_Arıza_Gun_Sayisi': 'Days since last failure - check if uses post-cutoff data',
        'Time_To_Repair': 'Repair time - check if includes future repairs',
        'MTBF': 'MTBF - check if calculated from ALL faults (not just pre-cutoff)',
        'Arıza_Sayısı': 'Fault count - check time window',
        'Customer': 'Customer impact - check if includes future faults'
    }

    found_concerns = False
    for feature in top_10:
        for indicator, concern in leaky_indicators.items():
            if indicator.lower() in feature.lower():
                print(f"   ⚠️  {feature}: {concern}")
                found_concerns = True

    if not found_concerns:
        print("   ✓ No obvious leakage indicators in top 10 features")

# ============================================================================
# DIAGNOSTIC 2: TARGET CORRELATION WITH REMOVED LEAKY FEATURES
# ============================================================================
print("\n" + "="*100)
print("DIAGNOSTIC 2: CORRELATION WITH REMOVED LEAKY FEATURES")
print("="*100)

print("\nChecking correlation between targets and removed leaky features:")
print("(High correlation suggests removed features were indeed leaky)")

leaky_features = ['Toplam_Arıza_Sayisi_Lifetime', 'Arıza_Sayısı_3ay',
                  'Arıza_Sayısı_6ay', 'Arıza_Sayısı_12ay']

for horizon in horizons:
    target_col = f'Target_{horizon}'
    if target_col not in df_merged.columns:
        continue

    print(f"\n{horizon} Target Correlation:")
    for feat in leaky_features:
        if feat in df_merged.columns:
            corr = df_merged[[target_col, feat]].corr().iloc[0, 1]
            if abs(corr) > 0.5:
                print(f"   ❌ {feat}: {corr:.3f} (HIGH - confirms leakage)")
            elif abs(corr) > 0.3:
                print(f"   ⚠️  {feat}: {corr:.3f} (MODERATE)")
            else:
                print(f"   ✓ {feat}: {corr:.3f} (low)")

# ============================================================================
# DIAGNOSTIC 3: CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("DIAGNOSTIC 3: MODEL CALIBRATION CHECK")
print("="*100)

print("\nModel calibration assessment:")
print("Well-calibrated: Predicted probabilities match actual failure rates")
print("Poorly calibrated: High AUC but probabilities unreliable")

# Note: Would need test set predictions to compute calibration
# This is a placeholder for when test predictions are available
print("\n⚠️  Calibration check requires test set predictions")
print("   Add this to 06_temporal_pof_model.py:")
print("   y_pred_proba = model.predict_proba(X_test)[:, 1]")
print("   fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)")

# ============================================================================
# DIAGNOSTIC 4: SAMPLE SIZE ADEQUACY CHECK
# ============================================================================
print("\n" + "="*100)
print("DIAGNOSTIC 4: SAMPLE SIZE ADEQUACY")
print("="*100)

print("\nRule of thumb: Need 10-20 events per feature for stable models")
print("Current setup:")

n_features = len([col for col in df.columns if col not in ['Ekipman_ID'] + [f'Target_{h}' for h in horizons]])
print(f"   Features: {n_features}")

for horizon in horizons:
    target_col = f'Target_{horizon}'
    if target_col in df.columns:
        n_positive = df[target_col].sum()
        ratio = n_positive / n_features

        if ratio < 5:
            status = "❌ SEVERE OVERFITTING RISK"
        elif ratio < 10:
            status = "⚠️  HIGH OVERFITTING RISK"
        elif ratio < 20:
            status = "⚠️  MODERATE OVERFITTING RISK"
        else:
            status = "✓ ADEQUATE"

        print(f"   {horizon}: {n_positive} positives / {n_features} features = {ratio:.1f} events/feature {status}")

# ============================================================================
# DIAGNOSTIC 5: DATA AVAILABILITY CHECK
# ============================================================================
print("\n" + "="*100)
print("DIAGNOSTIC 5: DATA AVAILABILITY FOR 24M HORIZON")
print("="*100)

print("\nChecking why 12M and 24M targets are identical...")

if 'Target_12M' in df.columns and 'Target_24M' in df.columns:
    target_12M = df['Target_12M'].sum()
    target_24M = df['Target_24M'].sum()

    print(f"   12M positive count: {target_12M}")
    print(f"   24M positive count: {target_24M}")

    if target_12M == target_24M:
        print(f"\n   ❌ CONFIRMED: Targets are identical!")
        print(f"   Likely cause: No fault data exists between 2025-06-25 and 2026-06-25")
        print(f"   Recommendation: Remove 24M horizon or document data limitation")
    else:
        print(f"   ✓ Targets are different (as expected)")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)

print("\n📋 SUMMARY:")
print("   1. Check top features for leakage indicators")
print("   2. Verify removed features had high target correlation")
print("   3. Sample size adequacy: Check events/feature ratio")
print("   4. 12M=24M issue: Verify data availability")
print("\n💡 NEXT STEPS:")
print("   If high AUC persists after ruling out leakage:")
print("   - Use time-based cross-validation (not random split)")
print("   - Reduce feature count to 15-20 (currently too many for sample size)")
print("   - Consider ensemble with simpler models (Logistic Regression baseline)")
