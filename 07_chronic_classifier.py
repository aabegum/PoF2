"""
CHRONIC REPEATER CLASSIFICATION
Turkish EDAŞ PoF Prediction Project (v4.0)

Purpose:
- Identify chronic repeater equipment (failure-prone assets)
- Binary classification: Chronic (1) vs Non-Chronic (0)
- Complement temporal PoF predictions (script 06_model_training.py)
- Support Replace vs Repair decisions

Target Definition:
- Chronic Repeater = Equipment with multiple recurring failures
- Criteria: Tekrarlayan_Arıza_90gün_Flag = 1 (94 equipment, 12%)
- OR: >= 2 lifetime failures AND high failure rate

Difference from Temporal PoF:
- Temporal PoF (script 06): "WHEN will equipment fail?" (prospective)
- Chronic Repeater: "WHICH equipment are failure-prone?" (classification)

Strategy:
- Balanced class weights (handle 12% positive class)
- 70/30 train/test split with stratification
- GridSearchCV for hyperparameter optimization
- Feature importance to understand chronic repeater patterns

Input:  data/features_selected_clean.csv (26 features)
Output: models/chronic_repeater_*.pkl, predictions/chronic_repeaters.csv

Author: Data Analytics Team
Date: 2025
Version: 4.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
import sys

# Import centralized configuration
from config import (
    FEATURES_REDUCED_FILE,
    MODEL_DIR,
    PREDICTION_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    N_FOLDS
)

# Fix Unicode encoding for Windows console (Turkish cp1254 issue)
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# Model libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*100)
print(" "*25 + "CHRONIC REPEATER CLASSIFICATION")
print(" "*20 + "Identify Failure-Prone Equipment | Replace vs Repair")
print("="*100)

# ============================================================================
# CONFIGURATION (Imported from config.py)
# ============================================================================

# Model parameters (from config.py):
# RANDOM_STATE, TEST_SIZE, N_FOLDS

# GridSearchCV settings
USE_GRIDSEARCH = True
GRIDSEARCH_VERBOSE = 1
GRIDSEARCH_N_JOBS = -1

# XGBoost base parameters
XGBOOST_BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'scale_pos_weight': 1.0  # Will be calculated based on class balance
}

# XGBoost GridSearchCV parameter grid
XGBOOST_PARAM_GRID = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
    'min_child_weight': [3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 0.1],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [1.0, 2.0]
}  # 48 combinations

# CatBoost base parameters
CATBOOST_BASE_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'auto_class_weights': 'Balanced',
    'task_type': 'CPU',
    'thread_count': -1
}

# CatBoost GridSearchCV parameter grid
CATBOOST_PARAM_GRID = {
    'iterations': [100, 150],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 5, 6],
    'l2_leaf_reg': [1, 3],
    'border_count': [64]
}  # 24 combinations

# Create output directories
MODEL_DIR.mkdir(exist_ok=True)
PREDICTION_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'chronic_repeater').mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("\n📋 Configuration:")
print(f"   Random State: {RANDOM_STATE}")
print(f"   Train/Test Split: {100-TEST_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"   Cross-Validation Folds: {N_FOLDS}")
print(f"   Class Weight Strategy: Balanced")
print(f"   Hyperparameter Tuning: {'GridSearchCV (ENABLED)' if USE_GRIDSEARCH else 'DISABLED'}")
if USE_GRIDSEARCH:
    xgb_combinations = np.prod([len(v) for v in XGBOOST_PARAM_GRID.values()])
    cat_combinations = np.prod([len(v) for v in CATBOOST_PARAM_GRID.values()])
    print(f"   XGBoost Grid Size: {xgb_combinations:,} combinations")
    print(f"   CatBoost Grid Size: {cat_combinations:,} combinations")

print(f"\n🎯 CHRONIC REPEATER CLASSIFICATION:")
print(f"   • Target = Equipment with recurring failures (Tekrarlayan_Arıza_90gün_Flag)")
print(f"   • Expected positive class: ~12% (94 out of 789 equipment)")
print(f"   • Expected AUC: 0.85-0.92 (realistic classification)")
print(f"   • Use: Replace vs Repair decisions")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING SELECTED FEATURES")
print("="*100)

# Use reduced features (comprehensive feature selection)
data_path = FEATURES_REDUCED_FILE

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("Please run 05b_remove_leaky_features.py first!")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# ============================================================================
# STEP 2: CREATE CHRONIC REPEATER TARGET
# ============================================================================
print("\n" + "="*100)
print("STEP 2: CREATING CHRONIC REPEATER TARGET")
print("="*100)

print("\n🎯 CHRONIC REPEATER DEFINITION:")
print("   Equipment with recurring failures within 90-day window")
print("   Using Tekrarlayan_Arıza_90gün_Flag feature")

# Verify required column exists
if 'Tekrarlayan_Arıza_90gün_Flag' not in df.columns:
    print(f"\n❌ ERROR: 'Tekrarlayan_Arıza_90gün_Flag' column not found!")
    print("This column should be in features_selected_clean.csv")
    print("Available columns:", list(df.columns[:10]), "...")
    exit(1)

# Target = Chronic repeater flag
df['Target_Chronic_Repeater'] = df['Tekrarlayan_Arıza_90gün_Flag'].astype(int)

# Print distribution
target_dist = df['Target_Chronic_Repeater'].value_counts()
pos_rate = target_dist.get(1, 0) / len(df) * 100

print(f"\n--- Chronic Repeater Target Distribution ---")
print(f"   Chronic Repeater (1):     {target_dist.get(1, 0):3d} ({pos_rate:5.1f}%)")
print(f"   Non-Chronic (0):          {target_dist.get(0, 0):3d} ({100-pos_rate:5.1f}%)")

# Validation
expected = 94
if abs(target_dist.get(1, 0) - expected) <= 5:
    print(f"   ✅ Status: CORRECT (expected ~{expected}, got {target_dist.get(1, 0)})")
else:
    print(f"   ⚠️  Status: CHECK (expected ~{expected}, got {target_dist.get(1, 0)})")

# Warning if class imbalance is severe
if pos_rate < 5:
    print(f"\n   ⚠️  WARNING: Severe class imbalance ({pos_rate:.1f}% positive)")
    print(f"       Using balanced class weights to address imbalance")

# ============================================================================
# STEP 3: PREPARE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 3: PREPARING FEATURES FOR MODELING")
print("="*100)

# Identify feature types
id_col = 'Ekipman_ID'
target_col = 'Target_Chronic_Repeater'

# Get feature columns (exclude ID, target, and the flag itself to avoid leakage)
exclude_cols = [id_col, target_col, 'Tekrarlayan_Arıza_90gün_Flag']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Identify categorical vs numeric features
categorical_features = []
numeric_features = []

for col in feature_cols:
    if df[col].dtype == 'object' or df[col].nunique() < 10:
        categorical_features.append(col)
    else:
        numeric_features.append(col)

print(f"\n✓ Feature Preparation:")
print(f"   ID column: {id_col}")
print(f"   Numeric features: {len(numeric_features)}")
print(f"   Categorical features: {len(categorical_features)}")
print(f"   Target variable: {target_col}")

print(f"\nNumeric Features:")
for i, feat in enumerate(numeric_features, 1):
    print(f"  {i:2d}. {feat}")

if categorical_features:
    print(f"\nCategorical Features:")
    for i, feat in enumerate(categorical_features, 1):
        print(f"  {i:2d}. {feat}")

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"\n✓ Encoded {col}: {df[col].nunique()} unique values")

# Prepare feature matrix
X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"\n✓ Feature matrix prepared: {X.shape}")

# ============================================================================
# STEP 4: CREATE TRAIN/TEST SPLITS
# ============================================================================
print("\n" + "="*100)
print("STEP 4: CREATING TRAIN/TEST SPLITS")
print("="*100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\n✓ Data Split:")
print(f"   Training set: {len(X_train):,} equipment ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test):,} equipment ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n  Chronic Repeater Target Split:")
print(f"    Train positive rate: {y_train.mean()*100:.1f}%")
print(f"    Test positive rate: {y_test.mean()*100:.1f}%")

# ============================================================================
# STEP 5: TRAIN XGBOOST MODEL
# ============================================================================
print("\n" + "="*100)
print("STEP 5: TRAINING XGBOOST CHRONIC REPEATER MODEL")
print("="*100)

# Calculate scale_pos_weight for class imbalance
n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()
scale_pos_weight = n_negative / n_positive

print(f"\n📊 Class Balance:")
print(f"   Negative samples: {n_negative}")
print(f"   Positive samples: {n_positive}")
print(f"   Scale pos weight: {scale_pos_weight:.2f}")

if USE_GRIDSEARCH:
    print(f"\n⏳ Running GridSearchCV for hyperparameter tuning...")
    print(f"   Grid size: {np.prod([len(v) for v in XGBOOST_PARAM_GRID.values()]):,} combinations")
    print(f"   CV folds: {N_FOLDS}")
    print(f"   This may take several minutes...")

    # Update base params with scale_pos_weight
    xgb_params = XGBOOST_BASE_PARAMS.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight

    # Create model
    xgb_model = xgb.XGBClassifier(**xgb_params)

    # GridSearchCV
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        xgb_model,
        XGBOOST_PARAM_GRID,
        cv=cv,
        scoring='roc_auc',
        n_jobs=GRIDSEARCH_N_JOBS,
        verbose=GRIDSEARCH_VERBOSE
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ GridSearchCV Complete!")
    print(f"   Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"\n   Best Hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")

    # Best model
    xgb_best = grid_search.best_estimator_
else:
    print(f"\n⏳ Training with default parameters...")
    xgb_params = XGBOOST_BASE_PARAMS.copy()
    xgb_params.update({
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': scale_pos_weight
    })
    xgb_best = xgb.XGBClassifier(**xgb_params)
    xgb_best.fit(X_train, y_train)

# Evaluate on test set
y_pred_proba = xgb_best.predict_proba(X_test)[:, 1]
y_pred = xgb_best.predict(X_test)

xgb_auc = roc_auc_score(y_test, y_pred_proba)
xgb_ap = average_precision_score(y_test, y_pred_proba)
xgb_precision = precision_score(y_test, y_pred)
xgb_recall = recall_score(y_test, y_pred)
xgb_f1 = f1_score(y_test, y_pred)

print(f"\n✅ XGBoost Chronic Repeater Test Set Results:")
print(f"   AUC: {xgb_auc:.4f}")
print(f"   Average Precision: {xgb_ap:.4f}")
print(f"   Precision: {xgb_precision:.4f}")
print(f"   Recall: {xgb_recall:.4f}")
print(f"   F1-Score: {xgb_f1:.4f}")

if xgb_auc >= 0.95:
    print(f"\n   ⚠️  WARNING: Very high AUC ({xgb_auc:.4f}) - potential overfitting!")
    print(f"       Expected AUC: 0.85-0.92 for chronic repeater classification")
elif xgb_auc >= 0.85:
    print(f"\n   ✅ Realistic AUC ({xgb_auc:.4f}) - model generalizes well")
else:
    print(f"\n   ⚠️  Low AUC ({xgb_auc:.4f}) - model may need more features or tuning")

# Save model
model_path = MODEL_DIR / 'chronic_repeater_xgboost.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(xgb_best, f)
print(f"\n💾 Model saved: {model_path}")

# ============================================================================
# STEP 6: TRAIN CATBOOST MODEL
# ============================================================================
print("\n" + "="*100)
print("STEP 6: TRAINING CATBOOST CHRONIC REPEATER MODEL")
print("="*100)

if USE_GRIDSEARCH:
    print(f"\n⏳ Running GridSearchCV for hyperparameter tuning...")
    print(f"   Grid size: {np.prod([len(v) for v in CATBOOST_PARAM_GRID.values()]):,} combinations")
    print(f"   CV folds: {N_FOLDS}")
    print(f"   This may take several minutes...")

    # Create model
    cat_model = CatBoostClassifier(**CATBOOST_BASE_PARAMS)

    # GridSearchCV
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        cat_model,
        CATBOOST_PARAM_GRID,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1,  # CatBoost handles parallelization internally
        verbose=GRIDSEARCH_VERBOSE
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ GridSearchCV Complete!")
    print(f"   Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"\n   Best Hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")

    cat_best = grid_search.best_estimator_
else:
    print(f"\n⏳ Training with default parameters...")
    cat_params = CATBOOST_BASE_PARAMS.copy()
    cat_params.update({
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 5
    })
    cat_best = CatBoostClassifier(**cat_params)
    cat_best.fit(X_train, y_train)

# Evaluate
y_pred_proba_cat = cat_best.predict_proba(X_test)[:, 1]
y_pred_cat = cat_best.predict(X_test)

cat_auc = roc_auc_score(y_test, y_pred_proba_cat)
cat_ap = average_precision_score(y_test, y_pred_proba_cat)
cat_precision = precision_score(y_test, y_pred_cat)
cat_recall = recall_score(y_test, y_pred_cat)
cat_f1 = f1_score(y_test, y_pred_cat)

print(f"\n✅ CatBoost Chronic Repeater Test Set Results:")
print(f"   AUC: {cat_auc:.4f}")
print(f"   Average Precision: {cat_ap:.4f}")
print(f"   Precision: {cat_precision:.4f}")
print(f"   Recall: {cat_recall:.4f}")
print(f"   F1-Score: {cat_f1:.4f}")

# Save model
model_path = MODEL_DIR / 'chronic_repeater_catboost.pkl'
cat_best.save_model(str(model_path))
print(f"\n💾 Model saved: {model_path}")

# ============================================================================
# STEP 7: MODEL PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "="*100)
print("STEP 7: MODEL PERFORMANCE COMPARISON")
print("="*100)

comparison = pd.DataFrame({
    'Model': ['XGBoost', 'CatBoost'],
    'AUC': [xgb_auc, cat_auc],
    'AP': [xgb_ap, cat_ap],
    'Precision': [xgb_precision, cat_precision],
    'Recall': [xgb_recall, cat_recall],
    'F1': [xgb_f1, cat_f1]
})

print(f"\n📊 Model Performance Comparison:")
print(comparison.to_string(index=False))

# Save comparison
comparison_path = RESULTS_DIR / 'chronic_repeater_model_comparison.csv'
comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
print(f"\n✓ Comparison saved: {comparison_path}")

# Best model
best_model_name = 'XGBoost' if xgb_auc >= cat_auc else 'CatBoost'
best_auc = max(xgb_auc, cat_auc)
print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("="*100)

print(f"\n--- XGBoost Feature Importance ---")

# Get feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_best.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Features for Chronic Repeater Classification:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['Feature']:<35} {row['Importance']:.4f}")

# Save feature importance
importance_path = RESULTS_DIR / 'chronic_repeater_feature_importance.csv'
importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
print(f"\n✓ Feature importance saved: {importance_path}")

# ============================================================================
# STEP 9: GENERATE PREDICTIONS FOR ALL EQUIPMENT
# ============================================================================
print("\n" + "="*100)
print("STEP 9: GENERATING PREDICTIONS FOR ALL EQUIPMENT")
print("="*100)

print(f"\n--- Generating Chronic Repeater Predictions ---")

# Use best model
best_model = xgb_best if xgb_auc >= cat_auc else cat_best

# Predict on all data
chronic_proba = best_model.predict_proba(X)[:, 1]

# Create predictions dataframe
predictions = pd.DataFrame({
    'Ekipman_ID': df[id_col],
    'Chronic_Probability': chronic_proba,
    'Chronic_Repeater_Flag_Actual': df['Tekrarlayan_Arıza_90gün_Flag'],
    'Chronic_Class': pd.cut(chronic_proba,
                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High', 'Critical'])
})

# Save predictions
pred_path = PREDICTION_DIR / 'chronic_repeaters.csv'
predictions.to_csv(pred_path, index=False, encoding='utf-8-sig')

# Distribution
risk_dist = predictions['Chronic_Class'].value_counts()
print(f"\n✓ Chronic Repeater Predictions:")
print(f"  Saved to: {pred_path}")
print(f"  Total equipment: {len(predictions):,}")
print(f"  🔴 Critical (>70%): {risk_dist.get('Critical', 0):,} ({risk_dist.get('Critical', 0)/len(predictions)*100:.1f}%)")
print(f"  🟠 High (50-70%): {risk_dist.get('High', 0):,} ({risk_dist.get('High', 0)/len(predictions)*100:.1f}%)")
print(f"  🟡 Medium (30-50%): {risk_dist.get('Medium', 0):,} ({risk_dist.get('Medium', 0)/len(predictions)*100:.1f}%)")
print(f"  🟢 Low (<30%): {risk_dist.get('Low', 0):,} ({risk_dist.get('Low', 0)/len(predictions)*100:.1f}%)")

# ============================================================================
# STEP 10: HIGH-RISK CHRONIC REPEATER REPORT
# ============================================================================
print("\n" + "="*100)
print("STEP 10: GENERATING HIGH-RISK CHRONIC REPEATER REPORT")
print("="*100)

print(f"\n--- Identifying High-Risk Chronic Repeaters ---")

# High-risk threshold
high_risk = predictions[predictions['Chronic_Probability'] > 0.50]

print(f"\n🚨 High-Risk Chronic Repeaters Identified: {len(high_risk)}")
print(f"   Threshold: Probability > 50%")
print(f"   Percentage of total: {len(high_risk)/len(predictions)*100:.1f}%")

# Merge with original features for context
high_risk_full = high_risk.merge(
    df[['Ekipman_ID', 'Equipment_Class_Primary'] +
       ([col for col in df.columns if 'Composite' in col or 'Risk_Score' in col][:1] if any('Composite' in col or 'Risk_Score' in col for col in df.columns) else [])],
    on='Ekipman_ID',
    how='left'
)

# Decode equipment class if encoded
if 'Equipment_Class_Primary' in label_encoders:
    high_risk_full['Equipment_Class_Primary'] = label_encoders['Equipment_Class_Primary'].inverse_transform(
        high_risk_full['Equipment_Class_Primary'].astype(int)
    )

# Sort by probability
high_risk_full = high_risk_full.sort_values('Chronic_Probability', ascending=False)

print(f"\n--- Top 10 Highest Risk Chronic Repeaters ---")
for idx, (i, row) in enumerate(high_risk_full.head(10).iterrows(), 1):
    print(f"  {idx:2d}. ID: {int(row['Ekipman_ID'])} | Class: {row.get('Equipment_Class_Primary', 'N/A')} | "
          f"Probability: {row['Chronic_Probability']*100:.1f}%")

# Save high-risk report
report_path = RESULTS_DIR / 'high_risk_chronic_repeaters.csv'
high_risk_full.to_csv(report_path, index=False, encoding='utf-8-sig')
print(f"\n✓ High-risk chronic repeater report saved: {report_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("CHRONIC REPEATER CLASSIFICATION COMPLETE")
print("="*100)

print(f"\n🎯 MODELS TRAINED:")
print(f"   XGBoost Chronic Repeater: AUC={xgb_auc:.4f}")
print(f"   CatBoost Chronic Repeater: AUC={cat_auc:.4f}")
print(f"   Best Model: {best_model_name}")

print(f"\n📊 CHRONIC REPEATER SUMMARY:")
print(f"   Total Equipment: {len(predictions):,}")
print(f"   High-Risk (>50%): {len(high_risk):,} ({len(high_risk)/len(predictions)*100:.1f}%)")
print(f"   Actual Chronic Flags: {predictions['Chronic_Repeater_Flag_Actual'].sum():,}")

print(f"\n📂 OUTPUT FILES:")
print(f"   Models: models/chronic_repeater_*.pkl")
print(f"   Predictions: {pred_path}")
print(f"   High-Risk Report: {report_path}")
print(f"   Feature Importance: {importance_path}")

print(f"\n✅ READY FOR DEPLOYMENT:")
print(f"   • Use predictions for Replace vs Repair decisions")
print(f"   • Combine with Temporal PoF (script 06) for complete risk assessment")
print(f"   • High-risk chronic repeaters: Priority for replacement")
print(f"   • Medium-risk: Enhanced monitoring and preventive maintenance")

print(f"\n💡 NEXT STEPS:")
print(f"   1. Review high-risk chronic repeater list")
print(f"   2. Run Temporal PoF (python 06_model_training.py)")
print(f"   3. Combine both predictions for integrated risk assessment")
print(f"   4. Run Survival Analysis (python 09_survival_analysis.py)")
print(f"   5. Integrate with CoF (python 11_consequence_of_failure.py)")

print("\n" + "="*100)
print(f"{'CHRONIC REPEATER CLASSIFICATION PIPELINE COMPLETE':^100}")
print("="*100)
