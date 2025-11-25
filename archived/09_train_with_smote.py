"""
TRAIN MODEL WITH SMOTE (Optional Enhancement)
==============================================
Uses SMOTE to oversample minority class before training

⚠️ WARNING: Use with caution!
   - SMOTE creates SYNTHETIC samples (not real data)
   - Can cause overfitting on synthetic patterns
   - Best for horizons with ≥40 positive samples (6M, 12M)
   - NOT recommended for 3M (only 29 samples)

COMPARISON: This script trains models with SMOTE and compares to baseline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix Unicode encoding for Windows console (Turkish cp1254 issue)
if sys.platform == 'win32':
    try:
        # Set console to UTF-8 mode for Unicode symbols
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        # Reconfigure stdout with UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        # If encoding setup fails, continue anyway
        pass

print("="*100)
print("MODEL TRAINING WITH SMOTE")
print("="*100)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING DATA")
print("="*100)

df = pd.read_csv('data/features_reduced.csv')
print(f"\n✓ Loaded: {len(df)} equipment × {len(df.columns)} columns")

# ============================================================================
# STEP 2: PREPARE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: PREPARING FEATURES")
print("="*100)

horizons = ['6M', '12M']  # Skip 3M (too few samples for SMOTE)
id_column = 'Ekipman_ID'
target_columns = [f'Target_{h}' for h in horizons]

# Identify categorical features
categorical_features = []
for col in df.columns:
    if col not in [id_column] + target_columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)

print(f"\n✓ Categorical features: {categorical_features}")

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Get feature columns
feature_columns = [col for col in df.columns if col not in [id_column] + target_columns]
print(f"✓ Feature count: {len(feature_columns)}")

# ============================================================================
# STEP 3: TRAIN WITH AND WITHOUT SMOTE
# ============================================================================
print("\n" + "="*100)
print("STEP 3: TRAINING COMPARISON (Baseline vs SMOTE)")
print("="*100)

results = []

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"{horizon} HORIZON")
    print(f"{'='*80}")

    target_col = f'Target_{horizon}'

    # Prepare data
    X = df[feature_columns]
    y = df[target_col]

    # Fill NaN
    X = X.fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 Original Distribution:")
    print(f"   Train: {len(X_train)} samples, {y_train.sum()} positives ({y_train.mean()*100:.1f}%)")
    print(f"   Test:  {len(X_test)} samples, {y_test.sum()} positives ({y_test.mean()*100:.1f}%)")

    # -------------------------------------------------------------------
    # BASELINE: No SMOTE
    # -------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print(f"BASELINE (No SMOTE)")
    print(f"{'─'*80}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model_baseline = xgb.XGBClassifier(
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

    model_baseline.fit(X_train, y_train)

    y_pred_baseline = model_baseline.predict_proba(X_test)[:, 1]
    auc_baseline = roc_auc_score(y_test, y_pred_baseline)
    ap_baseline = average_precision_score(y_test, y_pred_baseline)

    print(f"\n   Baseline Results:")
    print(f"      AUC: {auc_baseline:.4f}")
    print(f"      Average Precision: {ap_baseline:.4f}")

    # -------------------------------------------------------------------
    # WITH SMOTE
    # -------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print(f"WITH SMOTE")
    print(f"{'─'*80}")

    # Check if SMOTE is feasible
    if y_train.sum() < 6:
        print(f"\n   ❌ SKIPPING: Need at least 6 positive samples for SMOTE")
        print(f"      Current: {y_train.sum()} positive samples")
        continue

    # Apply SMOTE with target ratio
    # We'll aim for 1:2 ratio (33% positive)
    target_ratio = 0.33

    try:
        smote = SMOTE(
            sampling_strategy=target_ratio,
            k_neighbors=min(5, y_train.sum() - 1),  # Adjust k based on sample size
            random_state=42
        )

        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        print(f"\n   SMOTE Applied:")
        print(f"      Before: {len(X_train)} samples, {y_train.sum()} positives")
        print(f"      After:  {len(X_train_smote)} samples, {y_train_smote.sum()} positives ({y_train_smote.mean()*100:.1f}%)")
        print(f"      Created: {len(X_train_smote) - len(X_train)} synthetic samples")

        # Train with SMOTE data
        # Note: Don't use scale_pos_weight with SMOTE (data is already balanced)
        model_smote = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            verbosity=0
        )

        model_smote.fit(X_train_smote, y_train_smote)

        y_pred_smote = model_smote.predict_proba(X_test)[:, 1]
        auc_smote = roc_auc_score(y_test, y_pred_smote)
        ap_smote = average_precision_score(y_test, y_pred_smote)

        print(f"\n   SMOTE Results:")
        print(f"      AUC: {auc_smote:.4f}")
        print(f"      Average Precision: {ap_smote:.4f}")

        # Compare
        print(f"\n   📊 COMPARISON:")
        auc_diff = auc_smote - auc_baseline
        ap_diff = ap_smote - ap_baseline

        print(f"      AUC:   {auc_baseline:.4f} → {auc_smote:.4f} ({auc_diff:+.4f})")
        print(f"      AP:    {ap_baseline:.4f} → {ap_smote:.4f} ({ap_diff:+.4f})")

        if auc_smote > auc_baseline + 0.02:
            print(f"      ✅ SMOTE IMPROVED PERFORMANCE (+{auc_diff:.4f})")
            recommendation = "Use SMOTE"
        elif auc_smote < auc_baseline - 0.02:
            print(f"      ❌ SMOTE DEGRADED PERFORMANCE ({auc_diff:.4f})")
            recommendation = "Don't use SMOTE"
        else:
            print(f"      ⚠️  MINIMAL DIFFERENCE (±0.02)")
            recommendation = "SMOTE not necessary"

        # Store results
        results.append({
            'Horizon': horizon,
            'Method': 'Baseline',
            'Train_Samples': len(X_train),
            'Positive_Rate_%': y_train.mean() * 100,
            'AUC': auc_baseline,
            'Avg_Precision': ap_baseline
        })

        results.append({
            'Horizon': horizon,
            'Method': 'SMOTE',
            'Train_Samples': len(X_train_smote),
            'Positive_Rate_%': y_train_smote.mean() * 100,
            'AUC': auc_smote,
            'Avg_Precision': ap_smote
        })

        results.append({
            'Horizon': horizon,
            'Method': 'Recommendation',
            'Train_Samples': '-',
            'Positive_Rate_%': '-',
            'AUC': recommendation,
            'Avg_Precision': '-'
        })

    except Exception as e:
        print(f"\n   ❌ SMOTE FAILED: {str(e)}")

# ============================================================================
# STEP 4: FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("SMOTE EXPERIMENT SUMMARY")
print("="*100)

if len(results) > 0:
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # Save results
    Path('results').mkdir(exist_ok=True)
    results_df.to_csv('results/smote_comparison.csv', index=False)
    print("\n💾 Results saved to: results/smote_comparison.csv")

print("\n" + "="*100)
print("RECOMMENDATIONS")
print("="*100)

print("\n💡 WHEN TO USE SMOTE:")
print("   ✅ Use if SMOTE improves AUC by >0.02")
print("   ✅ Best for horizons with 40+ positive samples (6M, 12M)")
print("   ❌ Don't use for 3M (only 29 samples - high overfitting risk)")
print("   ❌ Don't use if baseline AUC already >0.90 (likely overfitting)")

print("\n⚠️  IMPORTANT CAVEATS:")
print("   - SMOTE creates SYNTHETIC data (not real failures)")
print("   - Model may learn synthetic patterns that don't exist in reality")
print("   - Use cautiously in production (validate with domain experts)")
print("   - Better solution: Collect more real data")

print("\n🎯 RECOMMENDED APPROACH:")
print("   1. Start with class weights (already implemented)")
print("   2. Use stratified sampling (maintains natural distribution)")
print("   3. Try SMOTE as experiment (use this script)")
print("   4. Compare performance on hold-out test set")
print("   5. If SMOTE helps, document clearly that model uses synthetic data")

print("\n" + "="*100)
print("SMOTE EXPERIMENT COMPLETE")
print("="*100)
