"""
MODEL TRAINING - TEMPORAL POF PREDICTION
Turkish EDAŞ PoF Prediction Project (v4.0 - Temporal Targets)

Purpose:
- Train XGBoost and CatBoost models with GridSearchCV hyperparameter tuning
- Predict TEMPORAL failure probability (equipment that WILL fail in next 6/12 months)
- Evaluate performance with multiple metrics
- Generate predictions and identify high-risk equipment

Changes in v4.0 (TEMPORAL TARGETS):
- MAJOR FIX: Target now based on ACTUAL future failures (after 2024-06-25)
- TEMPORAL: Predicts which equipment WILL fail in next 6M/12M (prospective)
- IMPROVED: Realistic AUC (0.75-0.85) instead of overfitted 1.0
- VALIDATED: Can compare predictions vs actual outcomes

Changes in v3.1:
- FIXED: Removed 3M horizon (all equipment has >= 1 lifetime failure)
- FIXED: Adjusted thresholds for better class balance
- IMPROVED: Reduced verbosity for cleaner console output

Strategy:
- Single model for all equipment classes (using Equipment_Class_Primary feature)
- Balanced class weights (optimize recall)
- 70/30 train/test split with stratification
- GridSearchCV with 3-fold stratified CV for hyperparameter optimization

Input:  data/features_selected_clean.csv (non-leaky features)
Output: models/*.pkl, predictions/*.csv, evaluation reports

Author: Data Analytics Team
Date: 2025
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
print(" "*30 + "POF MODEL TRAINING PIPELINE")
print(" "*30 + "XGBoost + CatBoost | 6/12 Months")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_FOLDS = 3  # For GridSearchCV (reduced from 5 for speed)

# GridSearchCV settings
USE_GRIDSEARCH = True  # Set to False to skip hyperparameter tuning
GRIDSEARCH_VERBOSE = 1  # 0=silent, 1=progress bar, 2=detailed (REDUCED for cleaner output)
GRIDSEARCH_N_JOBS = -1

# Prediction horizons - TEMPORAL (future failure windows)
# Cutoff date: 2024-06-25 (all features calculated using data BEFORE this date)
CUTOFF_DATE = pd.Timestamp('2024-06-25')

HORIZONS = {
    '6M': 180,   # Predict failures between 2024-06-25 and 2024-12-25 (164 equipment)
    '12M': 365   # Predict failures between 2024-06-25 and 2025-06-25 (266 equipment)
}

# Expected positive class rates (from check_future_data.py)
# 6M: ~20.8% (164 out of 789 equipment)
# 12M: ~33.7% (266 out of 789 equipment)

# XGBoost base parameters (fixed across all searches)
XGBOOST_BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'scale_pos_weight': 1.0  # Will be calculated per target
}

# XGBoost GridSearchCV parameter grid (REDUCED for stability)
XGBOOST_PARAM_GRID = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1.0]
}  # 64 combinations (much more manageable)

# CatBoost base parameters (fixed across all searches)
CATBOOST_BASE_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'auto_class_weights': 'Balanced',
    'task_type': 'CPU',
    'thread_count': -1
}

# CatBoost GridSearchCV parameter grid (REDUCED for stability)
CATBOOST_PARAM_GRID = {
    'iterations': [100, 200],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 6],
    'l2_leaf_reg': [1, 3],
    'border_count': [64]
}  # 16 combinations (much more manageable)

# Create output directories
Path('models').mkdir(exist_ok=True)
Path('predictions').mkdir(exist_ok=True)
Path('outputs/model_evaluation').mkdir(parents=True, exist_ok=True)
Path('results').mkdir(exist_ok=True)

print("\n📋 Configuration:")
print(f"   Random State: {RANDOM_STATE}")
print(f"   Train/Test Split: {100-TEST_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"   Cross-Validation Folds: {N_FOLDS}")
print(f"   Cutoff Date: {CUTOFF_DATE.date()}")
print(f"   Prediction Horizons: {list(HORIZONS.keys())}")
print(f"   Class Weight Strategy: Balanced")
print(f"   Hyperparameter Tuning: {'GridSearchCV (ENABLED)' if USE_GRIDSEARCH else 'DISABLED (using defaults)'}")
if USE_GRIDSEARCH:
    xgb_combinations = np.prod([len(v) for v in XGBOOST_PARAM_GRID.values()])
    cat_combinations = np.prod([len(v) for v in CATBOOST_PARAM_GRID.values()])
    print(f"   XGBoost Grid Size: {xgb_combinations:,} combinations")
    print(f"   CatBoost Grid Size: {cat_combinations:,} combinations")
    print(f"   Verbosity Level: {GRIDSEARCH_VERBOSE} (1=progress bar, cleaner output)")
print(f"\n🎯 TEMPORAL POF PREDICTION (v4.0):")
print(f"   • Target = Equipment that WILL fail in future windows")
print(f"   • 6M window: {CUTOFF_DATE.date()} → {(CUTOFF_DATE + pd.DateOffset(months=6)).date()}")
print(f"   • 12M window: {CUTOFF_DATE.date()} → {(CUTOFF_DATE + pd.DateOffset(months=12)).date()}")
print(f"   • Expected positive class: 6M=20.8%, 12M=33.7%")
print(f"   • Expected AUC: 0.75-0.85 (realistic temporal prediction)")
print(f"\n✓  Target Creation: Using ACTUAL future failures (prospective, not retrospective)")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING SELECTED FEATURES")
print("="*100)

# Try reduced features first (fixes data leakage), fall back to clean features
data_path_reduced = Path('data/features_reduced.csv')
data_path_clean = Path('data/features_selected_clean.csv')

if data_path_reduced.exists():
    data_path = data_path_reduced
    print(f"\n✓ Using REDUCED features (data leakage fixed)")
elif data_path_clean.exists():
    data_path = data_path_clean
    print(f"\n⚠️  Using CLEAN features (may have data leakage - run 05c_reduce_feature_redundancy.py)")
else:
    print(f"\n❌ ERROR: No feature files found!")
    print("Please run: python 05c_reduce_feature_redundancy.py")
    print("Or: python 05b_remove_leaky_features.py")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# ============================================================================
# STEP 2: CREATE TEMPORAL TARGET VARIABLES (v4.0)
# ============================================================================
print("\n" + "="*100)
print("STEP 2: CREATING TEMPORAL TARGET VARIABLES")
print("="*100)

print("\n🎯 TEMPORAL POF APPROACH: Using ACTUAL future failures (v4.0)")
print("   Target = Equipment that WILL fail in the future window")
print("   Based on actual fault occurrences AFTER cutoff date (2024-06-25)")

# Load ALL faults (including future) from original data
print("\n✓ Loading original fault data for temporal target creation...")
all_faults = pd.read_excel('data/combined_data.xlsx')
all_faults['started at'] = pd.to_datetime(all_faults['started at'],
                                           dayfirst=True,  # Turkish DD-MM-YYYY format
                                           errors='coerce')

# Define future prediction windows
FUTURE_6M_END = CUTOFF_DATE + pd.DateOffset(months=6)   # 2024-12-25
FUTURE_12M_END = CUTOFF_DATE + pd.DateOffset(months=12)  # 2025-06-25

print(f"\n--- Temporal Prediction Windows ---")
print(f"   Cutoff date:   {CUTOFF_DATE.date()}")
print(f"   6M window:     {CUTOFF_DATE.date()} → {FUTURE_6M_END.date()}")
print(f"   12M window:    {CUTOFF_DATE.date()} → {FUTURE_12M_END.date()}")

# Identify equipment that WILL FAIL in future windows
future_faults_6M = all_faults[
    (all_faults['started at'] > CUTOFF_DATE) &
    (all_faults['started at'] <= FUTURE_6M_END)
]['cbs_id'].dropna().unique()

future_faults_12M = all_faults[
    (all_faults['started at'] > CUTOFF_DATE) &
    (all_faults['started at'] <= FUTURE_12M_END)
]['cbs_id'].dropna().unique()

print(f"\n   Equipment that WILL fail in future:")
print(f"      6M window:  {len(future_faults_6M):,} equipment")
print(f"      12M window: {len(future_faults_12M):,} equipment")

# Create binary targets
print("\n--- Creating Binary Temporal Targets ---")

targets = {}

for horizon_name, horizon_days in HORIZONS.items():
    # Get equipment IDs that will fail in this window
    if horizon_name == '6M':
        failed_equipment = future_faults_6M
    else:  # 12M
        failed_equipment = future_faults_12M

    # Target = 1 if equipment WILL fail in future window
    targets[horizon_name] = df['Ekipman_ID'].isin(failed_equipment).astype(int)

    # Add to main dataframe
    df[f'Target_{horizon_name}'] = targets[horizon_name].values

    # Print distribution
    target_dist = df[f'Target_{horizon_name}'].value_counts()
    pos_rate = target_dist.get(1, 0) / len(df) * 100

    print(f"\n{horizon_name} Target (will fail in next {horizon_days} days):")
    print(f"   Will fail (1):     {target_dist.get(1, 0):3d} ({pos_rate:5.1f}%)")
    print(f"   Won't fail (0):    {target_dist.get(0, 0):3d} ({100-pos_rate:5.1f}%)")
    print(f"   ✓ Positive Rate: {pos_rate:.1f}%")

    # Validation - check against expected values
    expected_6M = 164
    expected_12M = 266
    expected = expected_6M if horizon_name == '6M' else expected_12M

    if abs(target_dist.get(1, 0) - expected) <= 5:
        print(f"   ✅ Status: CORRECT (expected ~{expected}, got {target_dist.get(1, 0)})")
    else:
        print(f"   ⚠️  Status: CHECK (expected ~{expected}, got {target_dist.get(1, 0)})")
        print(f"       Verify date parsing and equipment ID matching")

# ============================================================================
# STEP 3: PREPARE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 3: PREPARING FEATURES FOR MODELING")
print("="*100)

# Identify feature types
id_column = 'Ekipman_ID'
target_columns = [f'Target_{h}' for h in HORIZONS.keys()]
categorical_features = ['Equipment_Class_Primary', 'Risk_Category']

# Numeric features (all except ID, targets, and categoricals)
feature_columns = [col for col in df.columns 
                   if col != id_column 
                   and col not in target_columns 
                   and col not in categorical_features]

print(f"\n✓ Feature Preparation:")
print(f"   ID column: {id_column}")
print(f"   Numeric features: {len(feature_columns)}")
print(f"   Categorical features: {len(categorical_features)}")
print(f"   Target variables: {len(target_columns)}")

print(f"\nNumeric Features:")
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {feat}")

print(f"\nCategorical Features:")
for i, feat in enumerate(categorical_features, 1):
    print(f"  {i:2d}. {feat}")

# Encode categorical features for XGBoost
df_encoded = df.copy()
label_encoders = {}

for cat_feat in categorical_features:
    le = LabelEncoder()
    df_encoded[cat_feat] = le.fit_transform(df_encoded[cat_feat].astype(str))
    label_encoders[cat_feat] = le
    print(f"\n✓ Encoded {cat_feat}: {len(le.classes_)} unique values")

# All features for modeling (numeric + encoded categorical)
all_features = feature_columns + categorical_features

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*100)
print("STEP 4: CREATING TRAIN/TEST SPLITS")
print("="*100)

# We'll use the same train/test split for all horizons (stratified on 12M target)
X = df_encoded[all_features].copy()
y_12m = df_encoded['Target_12M'].copy()

# Stratified split to maintain class balance
X_train, X_test, _, _ = train_test_split(
    X, y_12m, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y_12m
)

train_idx = X_train.index
test_idx = X_test.index

print(f"\n✓ Data Split:")
print(f"   Training set: {len(train_idx):,} equipment ({len(train_idx)/len(df)*100:.1f}%)")
print(f"   Test set: {len(test_idx):,} equipment ({len(test_idx)/len(df)*100:.1f}%)")

# Split targets for each horizon
targets_train = {}
targets_test = {}

for horizon in HORIZONS.keys():
    target_col = f'Target_{horizon}'
    targets_train[horizon] = df_encoded.loc[train_idx, target_col]
    targets_test[horizon] = df_encoded.loc[test_idx, target_col]
    
    pos_rate_train = targets_train[horizon].sum() / len(targets_train[horizon]) * 100
    pos_rate_test = targets_test[horizon].sum() / len(targets_test[horizon]) * 100
    
    print(f"\n  {horizon} Target Split:")
    print(f"    Train positive rate: {pos_rate_train:.1f}%")
    print(f"    Test positive rate: {pos_rate_test:.1f}%")

# ============================================================================
# STEP 5: TRAIN MODELS - XGBOOST WITH GRIDSEARCHCV
# ============================================================================
print("\n" + "="*100)
print("STEP 5: TRAINING XGBOOST MODELS" + (" WITH GRIDSEARCHCV" if USE_GRIDSEARCH else ""))
print("="*100)

xgb_models = {}
xgb_results = {}
xgb_best_params = {}

for horizon in HORIZONS.keys():
    print(f"\n{'='*80}")
    print(f"Training XGBoost for {horizon} Horizon")
    print(f"{'='*80}")

    y_train = targets_train[horizon]
    y_test = targets_test[horizon]

    # Calculate class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\n📊 Class Balance:")
    print(f"   Negative samples: {n_neg:,}")
    print(f"   Positive samples: {n_pos:,}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    # Update base parameters with calculated class weight
    base_params = XGBOOST_BASE_PARAMS.copy()
    base_params['scale_pos_weight'] = scale_pos_weight

    if USE_GRIDSEARCH:
        # ===== GRIDSEARCHCV HYPERPARAMETER TUNING =====
        print(f"\n⏳ Running GridSearchCV for hyperparameter tuning...")
        print(f"   Grid size: {np.prod([len(v) for v in XGBOOST_PARAM_GRID.values()]):,} combinations")
        print(f"   CV folds: {N_FOLDS}")
        print(f"   This may take several minutes...")

        # Create base estimator
        xgb_estimator = xgb.XGBClassifier(**base_params)

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_estimator,
            param_grid=XGBOOST_PARAM_GRID,
            scoring='roc_auc',
            cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=GRIDSEARCH_N_JOBS,
            verbose=GRIDSEARCH_VERBOSE,
            refit=True
        )

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print(f"\n✅ GridSearchCV Complete!")
        print(f"   Best CV AUC: {best_cv_score:.4f}")
        print(f"\n   Best Hyperparameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")

        xgb_best_params[horizon] = best_params

    else:
        # ===== TRAINING WITH DEFAULT PARAMETERS =====
        print(f"\n⏳ Training XGBoost with default parameters...")

        # Use default parameters from grid (middle values)
        default_params = base_params.copy()
        default_params.update({
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        })

        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ===== EVALUATION =====
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ap = average_precision_score(y_test, y_pred_proba)

    print(f"\n✅ XGBoost {horizon} Test Set Results:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Average Precision: {ap:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Warning for suspiciously high performance
    if auc >= 0.98:
        print(f"\n   ⚠️  WARNING: Very high AUC ({auc:.4f}) may indicate data leakage!")
        print(f"   Check if features contain information from the target.")

    # Save model
    model_path = Path(f'models/xgboost_{horizon.lower()}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved: {model_path}")

    # Store results
    xgb_models[horizon] = model
    xgb_results[horizon] = {
        'auc': auc,
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

# Save best parameters if GridSearch was used
if USE_GRIDSEARCH:
    best_params_df = pd.DataFrame(xgb_best_params).T
    best_params_df.to_csv('results/xgboost_best_params.csv')
    print(f"\n💾 Best parameters saved: results/xgboost_best_params.csv")

# ============================================================================
# STEP 6: TRAIN MODELS - CATBOOST WITH GRIDSEARCHCV
# ============================================================================
print("\n" + "="*100)
print("STEP 6: TRAINING CATBOOST MODELS" + (" WITH GRIDSEARCHCV" if USE_GRIDSEARCH else ""))
print("="*100)

catboost_models = {}
catboost_results = {}
catboost_best_params = {}

# Get categorical feature indices
cat_features_idx = [all_features.index(f) for f in categorical_features if f in all_features]

for horizon in HORIZONS.keys():
    print(f"\n{'='*80}")
    print(f"Training CatBoost for {horizon} Horizon")
    print(f"{'='*80}")

    y_train = targets_train[horizon]
    y_test = targets_test[horizon]

    # Prepare data (use original non-encoded categoricals for CatBoost)
    X_train_cat = df.loc[train_idx, all_features].copy()
    X_test_cat = df.loc[test_idx, all_features].copy()

    # Handle NaN values in categorical features
    for cat_feat in categorical_features:
        X_train_cat[cat_feat] = X_train_cat[cat_feat].fillna('Unknown').astype(str)
        X_test_cat[cat_feat] = X_test_cat[cat_feat].fillna('Unknown').astype(str)

    # Handle NaN values in numeric features (fill with median)
    numeric_features_in_data = [f for f in all_features if f not in categorical_features]
    for num_feat in numeric_features_in_data:
        if X_train_cat[num_feat].isna().any():
            median_val = X_train_cat[num_feat].median()
            X_train_cat[num_feat] = X_train_cat[num_feat].fillna(median_val)
            X_test_cat[num_feat] = X_test_cat[num_feat].fillna(median_val)

    # Final NaN check
    if X_train_cat.isna().any().any():
        print("\n⚠️  WARNING: Filling remaining NaN values with 0")
        X_train_cat = X_train_cat.fillna(0)
        X_test_cat = X_test_cat.fillna(0)

    if USE_GRIDSEARCH:
        # ===== GRIDSEARCHCV HYPERPARAMETER TUNING =====
        print(f"\n⏳ Running GridSearchCV for hyperparameter tuning...")
        print(f"   Grid size: {np.prod([len(v) for v in CATBOOST_PARAM_GRID.values()]):,} combinations")
        print(f"   CV folds: {N_FOLDS}")
        print(f"   This may take several minutes...")

        # Create base estimator with cat_features
        cat_estimator = CatBoostClassifier(**CATBOOST_BASE_PARAMS, cat_features=cat_features_idx)

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=cat_estimator,
            param_grid=CATBOOST_PARAM_GRID,
            scoring='roc_auc',
            cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=1,  # CatBoost handles parallelization internally
            verbose=GRIDSEARCH_VERBOSE,
            refit=True
        )

        # Fit GridSearchCV
        grid_search.fit(X_train_cat, y_train)

        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print(f"\n✅ GridSearchCV Complete!")
        print(f"   Best CV AUC: {best_cv_score:.4f}")
        print(f"\n   Best Hyperparameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")

        catboost_best_params[horizon] = best_params

    else:
        # ===== TRAINING WITH DEFAULT PARAMETERS =====
        print(f"\n⏳ Training CatBoost with default parameters...")

        # Use default parameters from grid (middle values)
        default_params = CATBOOST_BASE_PARAMS.copy()
        default_params.update({
            'iterations': 200,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'border_count': 64
        })

        # Create pools for CatBoost
        train_pool = Pool(X_train_cat, y_train, cat_features=categorical_features)
        test_pool = Pool(X_test_cat, y_test, cat_features=categorical_features)

        model = CatBoostClassifier(**default_params, cat_features=cat_features_idx)
        model.fit(train_pool, eval_set=test_pool, verbose=False)

    # ===== EVALUATION =====
    # Predictions
    y_pred_proba = model.predict_proba(X_test_cat)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ap = average_precision_score(y_test, y_pred_proba)

    print(f"\n✅ CatBoost {horizon} Test Set Results:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Average Precision: {ap:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Save model
    model_path = Path(f'models/catboost_{horizon.lower()}.pkl')
    model.save_model(str(model_path))
    print(f"\n💾 Model saved: {model_path}")

    # Store results
    catboost_models[horizon] = model
    catboost_results[horizon] = {
        'auc': auc,
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

# Save best parameters if GridSearch was used
if USE_GRIDSEARCH:
    best_params_df = pd.DataFrame(catboost_best_params).T
    best_params_df.to_csv('results/catboost_best_params.csv')
    print(f"\n💾 Best parameters saved: results/catboost_best_params.csv")
# ============================================================================
# STEP 7: MODEL COMPARISON
# ============================================================================
print("\n" + "="*100)
print("STEP 7: MODEL PERFORMANCE COMPARISON")
print("="*100)

# Create comparison dataframe
comparison_data = []

for horizon in HORIZONS.keys():
    comparison_data.append({
        'Horizon': horizon,
        'Model': 'XGBoost',
        'AUC': xgb_results[horizon]['auc'],
        'AP': xgb_results[horizon]['ap'],
        'Precision': xgb_results[horizon]['precision'],
        'Recall': xgb_results[horizon]['recall'],
        'F1': xgb_results[horizon]['f1']
    })
    
    comparison_data.append({
        'Horizon': horizon,
        'Model': 'CatBoost',
        'AUC': catboost_results[horizon]['auc'],
        'AP': catboost_results[horizon]['ap'],
        'Precision': catboost_results[horizon]['precision'],
        'Recall': catboost_results[horizon]['recall'],
        'F1': catboost_results[horizon]['f1']
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n📊 Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('results/model_performance_comparison.csv', index=False)
print(f"\n✓ Comparison saved: results/model_performance_comparison.csv")

# Determine best model per horizon
print("\n🏆 Best Model by Horizon (based on AUC):")
for horizon in HORIZONS.keys():
    xgb_auc = xgb_results[horizon]['auc']
    cat_auc = catboost_results[horizon]['auc']
    
    if xgb_auc > cat_auc:
        print(f"  {horizon}: XGBoost (AUC: {xgb_auc:.4f}) vs CatBoost (AUC: {cat_auc:.4f})")
    else:
        print(f"  {horizon}: CatBoost (AUC: {cat_auc:.4f}) vs XGBoost (AUC: {xgb_auc:.4f})")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*100)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("="*100)

# XGBoost feature importance
print("\n--- XGBoost Feature Importance by Horizon ---")

importance_data = []

for horizon in HORIZONS.keys():
    model = xgb_models[horizon]
    importance = model.feature_importances_
    
    for feat, imp in zip(all_features, importance):
        importance_data.append({
            'Horizon': horizon,
            'Feature': feat,
            'Importance': imp,
            'Model': 'XGBoost'
        })

importance_df = pd.DataFrame(importance_data)

# Top 10 features per horizon
for horizon in HORIZONS.keys():
    print(f"\n{horizon} Horizon - Top 10 Features:")
    top_features = importance_df[
        (importance_df['Horizon'] == horizon) & 
        (importance_df['Model'] == 'XGBoost')
    ].nlargest(10, 'Importance')
    
    for i, row in enumerate(top_features.itertuples(), 1):
        print(f"  {i:2d}. {row.Feature:<35} {row.Importance:.4f}")

# Save feature importance
importance_df.to_csv('results/feature_importance_by_horizon.csv', index=False)
print(f"\n✓ Feature importance saved: results/feature_importance_by_horizon.csv")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 9: CREATING EVALUATION VISUALIZATIONS")
print("="*100)

# CHANGED: Adjusted subplot layout for 3 horizons instead of 4
# 1. ROC Curves
print("\n--- Creating ROC Curves ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # CHANGED: 1x3 instead of 2x2

for idx, horizon in enumerate(HORIZONS.keys()):
    ax = axes[idx]
    
    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(
        xgb_results[horizon]['y_test'], 
        xgb_results[horizon]['y_pred_proba']
    )
    ax.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={xgb_results[horizon]['auc']:.3f})", 
            linewidth=2, color='steelblue')
    
    # CatBoost ROC
    fpr_cat, tpr_cat, _ = roc_curve(
        catboost_results[horizon]['y_test'], 
        catboost_results[horizon]['y_pred_proba']
    )
    ax.plot(fpr_cat, tpr_cat, label=f"CatBoost (AUC={catboost_results[horizon]['auc']:.3f})", 
            linewidth=2, color='coral')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{horizon} Horizon ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_evaluation/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC curves saved: outputs/model_evaluation/roc_curves.png")

# 2. Precision-Recall Curves
print("\n--- Creating Precision-Recall Curves ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # CHANGED: 1x3 instead of 2x2

for idx, horizon in enumerate(HORIZONS.keys()):
    ax = axes[idx]
    
    # XGBoost PR
    precision_xgb, recall_xgb, _ = precision_recall_curve(
        xgb_results[horizon]['y_test'], 
        xgb_results[horizon]['y_pred_proba']
    )
    ax.plot(recall_xgb, precision_xgb, 
            label=f"XGBoost (AP={xgb_results[horizon]['ap']:.3f})", 
            linewidth=2, color='steelblue')
    
    # CatBoost PR
    precision_cat, recall_cat, _ = precision_recall_curve(
        catboost_results[horizon]['y_test'], 
        catboost_results[horizon]['y_pred_proba']
    )
    ax.plot(recall_cat, precision_cat, 
            label=f"CatBoost (AP={catboost_results[horizon]['ap']:.3f})", 
            linewidth=2, color='coral')
    
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'{horizon} Horizon Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_evaluation/precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ PR curves saved: outputs/model_evaluation/precision_recall_curves.png")

# 3. Confusion Matrices
print("\n--- Creating Confusion Matrices ---")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))  # CHANGED: 2x3 instead of 2x4

for idx, horizon in enumerate(HORIZONS.keys()):
    # XGBoost confusion matrix
    ax_xgb = axes[0, idx]
    cm_xgb = confusion_matrix(
        xgb_results[horizon]['y_test'], 
        xgb_results[horizon]['y_pred']
    )
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax_xgb, 
                cbar=False, square=True)
    ax_xgb.set_title(f'XGBoost {horizon}', fontsize=12, fontweight='bold')
    ax_xgb.set_ylabel('True Label', fontsize=10)
    ax_xgb.set_xlabel('Predicted Label', fontsize=10)
    
    # CatBoost confusion matrix
    ax_cat = axes[1, idx]
    cm_cat = confusion_matrix(
        catboost_results[horizon]['y_test'], 
        catboost_results[horizon]['y_pred']
    )
    sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Oranges', ax=ax_cat, 
                cbar=False, square=True)
    ax_cat.set_title(f'CatBoost {horizon}', fontsize=12, fontweight='bold')
    ax_cat.set_ylabel('True Label', fontsize=10)
    ax_cat.set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/model_evaluation/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrices saved: outputs/model_evaluation/confusion_matrices.png")

# 4. Feature Importance Comparison
print("\n--- Creating Feature Importance Comparison ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # CHANGED: 1x3 instead of 2x2

for idx, horizon in enumerate(HORIZONS.keys()):
    ax = axes[idx]
    
    # Get top 15 features for this horizon
    top_imp = importance_df[
        (importance_df['Horizon'] == horizon) & 
        (importance_df['Model'] == 'XGBoost')
    ].nlargest(15, 'Importance')
    
    # Plot
    ax.barh(range(len(top_imp)), top_imp['Importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp['Feature'].values, fontsize=8)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'{horizon} Horizon - Top 15 Features', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_evaluation/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature importance comparison saved")

# ============================================================================
# STEP 10: GENERATE PREDICTIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 10: GENERATING PREDICTIONS FOR ALL EQUIPMENT")
print("="*100)

# Generate predictions using best model (XGBoost by default, or choose based on AUC)
print("\n--- Generating Predictions ---")

for horizon in HORIZONS.keys():
    # Use XGBoost model (can switch to CatBoost if preferred)
    model = xgb_models[horizon]
    
    # Predict on ALL equipment
    X_all = df_encoded[all_features]
    predictions = model.predict_proba(X_all)[:, 1]
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'Ekipman_ID': df['Ekipman_ID'],
        'Equipment_Class': df['Equipment_Class_Primary'],
        f'Failure_Probability_{horizon}': predictions,
        f'Actual_Target_{horizon}': df[f'Target_{horizon}'],
        'Risk_Score': predictions * 100  # Convert to 0-100 score
    })
    
    # Add risk category
    pred_df['Risk_Level'] = pd.cut(
        pred_df['Risk_Score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Sort by risk
    pred_df = pred_df.sort_values('Risk_Score', ascending=False)
    
    # Save predictions
    pred_path = f'predictions/predictions_{horizon.lower()}.csv'
    pred_df.to_csv(pred_path, index=False)
    
    print(f"\n✓ {horizon} Predictions:")
    print(f"  Saved to: {pred_path}")
    print(f"  Total equipment: {len(pred_df):,}")
    
    # Risk distribution
    risk_dist = pred_df['Risk_Level'].value_counts()
    for risk_level in ['Critical', 'High', 'Medium', 'Low']:
        count = risk_dist.get(risk_level, 0)
        pct = count / len(pred_df) * 100
        icon = "🔴" if risk_level == 'Critical' else ("🟠" if risk_level == 'High' else ("🟡" if risk_level == 'Medium' else "🟢"))
        print(f"  {icon} {risk_level}: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 11: HIGH-RISK EQUIPMENT REPORT
# ============================================================================
print("\n" + "="*100)
print("STEP 11: GENERATING HIGH-RISK EQUIPMENT REPORT")
print("="*100)

print("\n--- Identifying High-Risk Equipment ---")

# Load predictions and merge
high_risk_data = df[['Ekipman_ID', 'Equipment_Class_Primary']].copy()

for horizon in HORIZONS.keys():
    pred_df = pd.read_csv(f'predictions/predictions_{horizon.lower()}.csv')
    high_risk_data[f'Risk_Score_{horizon}'] = pred_df['Risk_Score'].values
    high_risk_data[f'Risk_Level_{horizon}'] = pred_df['Risk_Level'].values

# Calculate average risk score across all horizons
risk_cols = [f'Risk_Score_{h}' for h in HORIZONS.keys()]
high_risk_data['Avg_Risk_Score'] = high_risk_data[risk_cols].mean(axis=1)

# Identify high-risk (average risk > 50)
high_risk = high_risk_data[high_risk_data['Avg_Risk_Score'] > 50].copy()
high_risk = high_risk.sort_values('Avg_Risk_Score', ascending=False)

print(f"\n🚨 High-Risk Equipment Identified: {len(high_risk):,}")
print(f"   Threshold: Average Risk Score > 50")
print(f"   Percentage of total: {len(high_risk)/len(df)*100:.1f}%")

if len(high_risk) > 0:
    print(f"\n--- Top 10 Highest Risk Equipment ---")
    for i, row in enumerate(high_risk.head(10).itertuples(), 1):
        print(f"  {i:2d}. ID: {row.Ekipman_ID} | Class: {row.Equipment_Class_Primary} | Risk: {row.Avg_Risk_Score:.1f}")
    
    # Save high-risk report
    high_risk_path = 'results/high_risk_equipment_report.csv'
    high_risk.to_csv(high_risk_path, index=False)
    print(f"\n✓ High-risk report saved: {high_risk_path}")
else:
    print("\n✓ No equipment with average risk > 50")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("MODEL TRAINING COMPLETE - SUMMARY")
print("="*100)

print(f"\n🎯 MODELS TRAINED:")
print(f"   XGBoost: 3 models (3M, 6M, 12M)")  # CHANGED: 3 instead of 4
print(f"   CatBoost: 3 models (3M, 6M, 12M)")  # CHANGED: 3 instead of 4
print(f"   Total: 6 models")  # CHANGED: 6 instead of 8

print(f"\n📊 PERFORMANCE SUMMARY (AUC):")
for horizon in HORIZONS.keys():
    xgb_auc = xgb_results[horizon]['auc']
    cat_auc = catboost_results[horizon]['auc']
    print(f"   {horizon}: XGBoost={xgb_auc:.4f} | CatBoost={cat_auc:.4f}")

print(f"\n📂 OUTPUT FILES:")
print(f"   Models: models/ (6 .pkl files)")  # CHANGED: 6 instead of 8
print(f"   Predictions: predictions/ (3 CSV files)")  # CHANGED: 3 instead of 4
print(f"   Visualizations: outputs/model_evaluation/ (4 PNG files)")
print(f"   Results: results/ (3 CSV files)")

print(f"\n🚨 HIGH-RISK EQUIPMENT:")
if len(high_risk) > 0:
    print(f"   Identified: {len(high_risk):,} equipment")
    print(f"   Report: results/high_risk_equipment_report.csv")
else:
    print(f"   None identified (all equipment < 50 risk score)")

print(f"\n✅ READY FOR DEPLOYMENT:")
print(f"   • Load models with pickle/catboost")
print(f"   • Make predictions on new equipment")
print(f"   • Monitor high-risk equipment")
print(f"   • Schedule maintenance based on predictions")

print("\n" + "="*100)
print(f"{'POF MODEL TRAINING PIPELINE COMPLETE':^100}")
print("="*100)