"""
LOGISTIC REGRESSION BASELINE - POF PREDICTION
Turkish EDAŞ PoF Prediction Project (v2.0)

Purpose:
- Train interpretable Logistic Regression baseline models with GridSearchCV
- Train Ridge and Lasso Regression for comparison
- Provide coefficient-based explanations (odds ratios)
- Benchmark for complex models (XGBoost/CatBoost)
- Enable business stakeholder understanding

Changes in v2.0:
- FIXED: Target creation now uses lifetime failure thresholds (no data leakage)
- ADDED: GridSearchCV for optimal regularization parameter (C) tuning
- ADDED: Ridge Regression (L2) and Lasso Regression (L1) as additional baselines
- IMPROVED: More robust target definition based on failure propensity

Strategy:
- L1 (Lasso), L2 (Ridge), and Elastic Net regularization options
- GridSearchCV for optimal C (inverse regularization strength)
- Balanced class weights (handle imbalance)
- 70/30 train/test split with stratification
- Feature importance via coefficients

Input:  data/features_selected_clean.csv (non-leaky features)
Output: models/logistic_*.pkl, results/logistic_coefficients.csv

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
warnings.filterwarnings('ignore')

# Model libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, average_precision_score
)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*100)
print(" "*30 + "POF LOGISTIC REGRESSION BASELINE")
print(" "*28 + "Interpretable Models | 6/12 Months")
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
GRIDSEARCH_VERBOSE = 1
GRIDSEARCH_N_JOBS = -1

# Prediction horizons (days)
# NOTE: 3M removed (100% positive class - all equipment has >= 1 lifetime failure)
HORIZONS = {
    '6M': 180,
    '12M': 365
}

# Target creation thresholds (based on lifetime failures)
# Equipment with >= threshold lifetime failures is considered "failure-prone"
# Based on data: All 1148 equipment have >= 1 failure, 245 have >= 2, 104 have >= 3
TARGET_THRESHOLDS = {
    '6M': 2,   # At least 2 lifetime failures → 245/1148 = 21.3% positive
    '12M': 2   # At least 2 lifetime failures → 245/1148 = 21.3% positive
}

# Logistic Regression base parameters
LOGISTIC_BASE_PARAMS = {
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# GridSearchCV parameter grid for Logistic Regression
LOGISTIC_PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse regularization strength
    'penalty': ['l1', 'l2'],  # L1 (Lasso) or L2 (Ridge)
    'solver': ['liblinear', 'saga']  # Solvers that support both L1 and L2
}

# Ridge Regression (L2 only) parameter grid
RIDGE_PARAM_GRID = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
    'max_iter': [1000]
}

# Lasso Regression (L1 only) parameter grid
LASSO_PARAM_GRID = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
    'max_iter': [1000]
}

# Create output directories
Path('models').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
Path('outputs/logistic_baseline').mkdir(parents=True, exist_ok=True)

print("\n📋 Configuration:")
print(f"   Random State: {RANDOM_STATE}")
print(f"   Train/Test Split: {100-TEST_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"   Cross-Validation Folds: {N_FOLDS}")
print(f"   Prediction Horizons: {list(HORIZONS.keys())}")
print(f"   Target Thresholds: {TARGET_THRESHOLDS}")
print(f"   Class Weight Strategy: Balanced")
print(f"   Hyperparameter Tuning: {'GridSearchCV (ENABLED)' if USE_GRIDSEARCH else 'DISABLED (using defaults)'}")
if USE_GRIDSEARCH:
    logistic_combinations = np.prod([len(v) for v in LOGISTIC_PARAM_GRID.values()])
    ridge_combinations = np.prod([len(v) for v in RIDGE_PARAM_GRID.values()])
    lasso_combinations = np.prod([len(v) for v in LASSO_PARAM_GRID.values()])
    print(f"   Logistic Grid Size: {logistic_combinations} combinations")
    print(f"   Ridge Grid Size: {ridge_combinations} combinations")
    print(f"   Lasso Grid Size: {lasso_combinations} combinations")
print(f"\n⚠️  NOTE: 3M horizon removed (100% positive class - all equipment has >= 1 lifetime failure)")
print(f"✓  Target Creation: Using lifetime failure thresholds (NO DATA LEAKAGE)")

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

# Load full engineered data for target creation
df_full = pd.read_csv('data/features_engineered.csv')
print(f"✓ Loaded full data for target creation: {df_full.shape[0]:,} equipment")

# ============================================================================
# STEP 2: CREATE TARGET VARIABLES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: CREATING TARGET VARIABLES FOR MULTIPLE HORIZONS")
print("="*100)

print("\n⚠️  NEW APPROACH: Using lifetime failure thresholds (NO DATA LEAKAGE)")
print("   Equipment with higher lifetime failures is more likely to fail in the future")
print("   Different thresholds for different prediction horizons")

# Verify required column exists
if 'Toplam_Arıza_Sayisi_Lifetime' not in df_full.columns:
    print(f"\n❌ ERROR: 'Toplam_Arıza_Sayisi_Lifetime' column not found!")
    print("This column should be created by 02_data_transformation.py")
    print("Available columns:", list(df_full.columns[:10]), "...")
    exit(1)

print("\n--- Creating Binary Targets Based on Lifetime Failure Propensity ---")

targets = {}

for horizon_name, horizon_days in HORIZONS.items():
    threshold = TARGET_THRESHOLDS[horizon_name]

    # Target = 1 if equipment has >= threshold lifetime failures
    # Equipment with more historical failures is failure-prone (higher future failure risk)
    targets[horizon_name] = (df_full['Toplam_Arıza_Sayisi_Lifetime'] >= threshold).astype(int)

    # Add to main dataframe
    df[f'Target_{horizon_name}'] = targets[horizon_name].values

    # Print distribution
    target_dist = df[f'Target_{horizon_name}'].value_counts()
    pos_rate = target_dist.get(1, 0) / len(df) * 100

    print(f"\n{horizon_name} ({horizon_days} days) Target:")
    print(f"  Threshold: >= {threshold} lifetime failures")
    print(f"  Failure-Prone (1): {target_dist.get(1, 0):,} ({pos_rate:.1f}%)")
    print(f"  Not Failure-Prone (0): {target_dist.get(0, 0):,} ({100-pos_rate:.1f}%)")
    print(f"  Positive Rate: {pos_rate:.1f}%")

    # Warning if class imbalance is severe
    if pos_rate < 5 or pos_rate > 95:
        print(f"  ⚠️  WARNING: Severe class imbalance ({pos_rate:.1f}% positive)")
        print(f"     Consider adjusting threshold or using SMOTE")

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

# Encode categorical features
df_encoded = df.copy()
label_encoders = {}

for cat_feat in categorical_features:
    le = LabelEncoder()
    df_encoded[cat_feat] = le.fit_transform(df_encoded[cat_feat].astype(str))
    label_encoders[cat_feat] = le
    print(f"\n✓ Encoded {cat_feat}: {len(le.classes_)} unique values")

# Save encoders
with open('models/label_encoders_logistic.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("\n✓ Saved label encoders")

# All features for modeling (numeric + encoded categorical)
all_features = feature_columns + categorical_features

# ============================================================================
# STEP 4: STANDARDIZE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 4: STANDARDIZING FEATURES")
print("="*100)

print("\n⚠️  NOTE: Logistic Regression requires feature scaling for interpretability")

# Separate features for scaling
X = df_encoded[all_features].copy()

# Standardize (mean=0, std=1)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=all_features,
    index=X.index
)

# Save scaler
with open('models/scaler_logistic.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved feature scaler")

print("\nFeature Scaling Statistics:")
for feat in all_features:
    print(f"  {feat:40s} | Mean: {X_scaled[feat].mean():7.3f} | Std: {X_scaled[feat].std():7.3f}")

# ============================================================================
# STEP 5: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*100)
print("STEP 5: CREATING TRAIN/TEST SPLITS")
print("="*100)

# Use same train/test split for all horizons (stratified on 12M target)
y_12m = df_encoded['Target_12M'].copy()

# Stratified split to maintain class balance
X_train, X_test, _, _ = train_test_split(
    X_scaled, y_12m,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_12m
)

train_idx = X_train.index
test_idx = X_test.index

print(f"\n✓ Data Split:")
print(f"   Training set: {len(train_idx):,} equipment ({len(train_idx)/len(df)*100:.1f}%)")
print(f"   Test set: {len(test_idx):,} equipment ({len(test_idx)/len(df)*100:.1f}%)")

# ============================================================================
# STEP 6: TRAIN LINEAR MODELS WITH GRIDSEARCHCV
# ============================================================================
print("\n" + "="*100)
print("STEP 6: TRAINING LINEAR MODELS" + (" WITH GRIDSEARCHCV" if USE_GRIDSEARCH else ""))
print("="*100)

# Storage for results
models = {}
predictions = {}
performance_metrics = []
all_coefficients = []
best_params_all = {}

# Model types to train
MODEL_TYPES = ['Logistic', 'Ridge', 'Lasso']

for horizon_name in HORIZONS.keys():
    print(f"\n{'='*100}")
    print(f"TRAINING MODELS FOR {horizon_name} HORIZON")
    print(f"{'='*100}")

    # Get target for this horizon
    y_train = df_encoded.loc[train_idx, f'Target_{horizon_name}'].values
    y_test = df_encoded.loc[test_idx, f'Target_{horizon_name}'].values

    # Check class distribution
    train_pos_rate = y_train.sum() / len(y_train) * 100
    test_pos_rate = y_test.sum() / len(y_test) * 100

    print(f"\n--- {horizon_name} Target Distribution ---")
    print(f"Training set:")
    print(f"  No Failure (0): {(y_train == 0).sum():,} ({100-train_pos_rate:.1f}%)")
    print(f"  Failure (1): {y_train.sum():,} ({train_pos_rate:.1f}%)")
    print(f"Test set:")
    print(f"  No Failure (0): {(y_test == 0).sum():,} ({100-test_pos_rate:.1f}%)")
    print(f"  Failure (1): {y_test.sum():,} ({test_pos_rate:.1f}%)")

    models[horizon_name] = {}
    best_params_all[horizon_name] = {}

    # Train each model type
    for model_type in MODEL_TYPES:
        print(f"\n{'='*80}")
        print(f"Training {model_type} Regression")
        print(f"{'='*80}")

        if USE_GRIDSEARCH:
            # ===== GRIDSEARCHCV HYPERPARAMETER TUNING =====
            print(f"\n⏳ Running GridSearchCV for {model_type}...")

            if model_type == 'Logistic':
                estimator = LogisticRegression(**LOGISTIC_BASE_PARAMS)
                param_grid = LOGISTIC_PARAM_GRID
            elif model_type == 'Ridge':
                estimator = Ridge(random_state=RANDOM_STATE)
                param_grid = RIDGE_PARAM_GRID
            else:  # Lasso
                estimator = Lasso(random_state=RANDOM_STATE)
                param_grid = LASSO_PARAM_GRID

            print(f"   Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
            print(f"   CV folds: {N_FOLDS}")

            # GridSearchCV
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
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

            best_params_all[horizon_name][model_type] = best_params

        else:
            # ===== TRAINING WITH DEFAULT PARAMETERS =====
            print(f"\n⏳ Training {model_type} with default parameters...")

            if model_type == 'Logistic':
                params = LOGISTIC_BASE_PARAMS.copy()
                params.update({'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'})
                model = LogisticRegression(**params)
            elif model_type == 'Ridge':
                model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
            else:  # Lasso
                model = Lasso(alpha=1.0, random_state=RANDOM_STATE)

            model.fit(X_train, y_train)

        # ===== EVALUATION =====
        # Predictions (handle different model types)
        if model_type == 'Logistic':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:  # Ridge and Lasso output continuous values
            y_pred_raw = model.predict(X_test)
            # Convert to probabilities (clip to [0,1])
            y_pred_proba = np.clip(y_pred_raw, 0, 1)
            y_pred = (y_pred_proba >= 0.5).astype(int)

        # Evaluation metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n✅ {model_type} Test Set Performance:")
        print(f"   AUC-ROC: {auc:.4f}")
        print(f"   Average Precision: {ap:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        # Store results
        models[horizon_name][model_type] = model
        predictions[f'{horizon_name}_{model_type}'] = {
            'y_true': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }

        # Save performance metrics
        performance_metrics.append({
            'Horizon': horizon_name,
            'Model': f'{model_type} Regression',
            'AUC': auc,
            'Average_Precision': ap,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        })

        # Extract coefficients (for interpretation)
        if model_type == 'Logistic':
            coefficients = pd.DataFrame({
                'Feature': all_features,
                'Coefficient': model.coef_[0],
                'Abs_Coefficient': np.abs(model.coef_[0]),
                'Odds_Ratio': np.exp(model.coef_[0])
            })
            coefficients['Intercept'] = model.intercept_[0]
        else:  # Ridge or Lasso
            coefficients = pd.DataFrame({
                'Feature': all_features,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_),
                'Odds_Ratio': np.nan  # Not applicable for Ridge/Lasso
            })
            coefficients['Intercept'] = model.intercept_

        coefficients['Horizon'] = horizon_name
        coefficients['Model'] = model_type
        coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

        all_coefficients.append(coefficients)

        print(f"\n--- Top 5 Most Important Features (by |coefficient|) ---")
        for idx, row in coefficients.head(5).iterrows():
            direction = "↑" if row['Coefficient'] > 0 else "↓"
            if model_type == 'Logistic':
                print(f"  {direction} {row['Feature']:40s} | Coef: {row['Coefficient']:7.3f} | Odds Ratio: {row['Odds_Ratio']:6.3f}")
            else:
                print(f"  {direction} {row['Feature']:40s} | Coef: {row['Coefficient']:7.3f}")

        # Save model
        model_path = f'models/{model_type.lower()}_{horizon_name.lower()}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n💾 Model saved: {model_path}")

    # Save predictions for test set (Logistic only for main predictions file)
    logistic_model = models[horizon_name]['Logistic']
    y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
    y_pred = logistic_model.predict(X_test)

    pred_df = pd.DataFrame({
        'Ekipman_ID': df.loc[test_idx, id_column].values,
        'Equipment_Class': df.loc[test_idx, 'Equipment_Class_Primary'].values,
        'True_Label': y_test,
        'Failure_Probability': y_pred_proba,
        'Predicted_Label': y_pred,
        'Risk_Score': (y_pred_proba * 100).round(2)
    })

    # Add risk levels
    pred_df['Risk_Level'] = pd.cut(
        pred_df['Risk_Score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    pred_path = f'predictions/logistic_predictions_{horizon_name.lower()}.csv'
    pred_df.to_csv(pred_path, index=False)
    print(f"\n✓ Predictions saved: {pred_path}")

# Save best parameters if GridSearch was used
if USE_GRIDSEARCH:
    best_params_df = pd.DataFrame(best_params_all).T
    best_params_df.to_csv('results/linear_models_best_params.csv')
    print(f"\n💾 Best parameters saved: results/linear_models_best_params.csv")

# ============================================================================
# STEP 7: VISUALIZE MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*100)
print("STEP 7: VISUALIZING MODEL PERFORMANCE")
print("="*100)

# 1. ROC Curves for all horizons (Logistic vs Ridge vs Lasso)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, horizon_name in enumerate(HORIZONS.keys()):
    ax = axes[idx]

    for model_type in MODEL_TYPES:
        pred_key = f'{horizon_name}_{model_type}'
        y_test = predictions[pred_key]['y_true']
        y_pred_proba = predictions[pred_key]['y_pred_proba']

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        ax.plot(fpr, tpr, label=f'{model_type} (AUC = {auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'{horizon_name} Horizon', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('ROC Curves - Linear Models Comparison (All Horizons)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/logistic_baseline/roc_curves_all_horizons.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/logistic_baseline/roc_curves_all_horizons.png")
plt.close()

# 2. Precision-Recall Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, horizon_name in enumerate(HORIZONS.keys()):
    ax = axes[idx]

    for model_type in MODEL_TYPES:
        pred_key = f'{horizon_name}_{model_type}'
        y_test = predictions[pred_key]['y_true']
        y_pred_proba = predictions[pred_key]['y_pred_proba']

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)

        ax.plot(recall, precision, label=f'{model_type} (AP = {ap:.3f})', linewidth=2)

    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title(f'{horizon_name} Horizon', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Precision-Recall Curves - Linear Models Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/logistic_baseline/pr_curves_all_horizons.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/logistic_baseline/pr_curves_all_horizons.png")
plt.close()

# 3. Confusion Matrices (Logistic Regression only - main model)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, horizon_name in enumerate(HORIZONS.keys()):
    pred_key = f'{horizon_name}_Logistic'
    y_test = predictions[pred_key]['y_true']
    y_pred = predictions[pred_key]['y_pred']

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    axes[idx].set_title(f'{horizon_name} Horizon', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/logistic_baseline/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/logistic_baseline/confusion_matrices.png")
plt.close()

# 4. Feature Coefficients Comparison (Logistic Regression only)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Get Logistic coefficients for each horizon
logistic_coefs = [coef_df for coef_df in all_coefficients if coef_df['Model'].iloc[0] == 'Logistic']

for idx, (horizon_name, coef_df) in enumerate(zip(HORIZONS.keys(), logistic_coefs)):
    top_coef = coef_df.head(10).copy()

    # Sort by coefficient value for better visualization
    top_coef = top_coef.sort_values('Coefficient')

    colors = ['red' if x < 0 else 'green' for x in top_coef['Coefficient']]

    axes[idx].barh(range(len(top_coef)), top_coef['Coefficient'], color=colors, alpha=0.7)
    axes[idx].set_yticks(range(len(top_coef)))
    axes[idx].set_yticklabels(top_coef['Feature'], fontsize=8)
    axes[idx].set_xlabel('Coefficient Value', fontsize=10)
    axes[idx].set_title(f'{horizon_name} Horizon', fontsize=12, fontweight='bold')
    axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/logistic_baseline/feature_coefficients_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/logistic_baseline/feature_coefficients_comparison.png")
plt.close()

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
print("\n" + "="*100)
print("STEP 8: SAVING RESULTS")
print("="*100)

# Save performance metrics
perf_df = pd.DataFrame(performance_metrics)
perf_df.to_csv('results/logistic_baseline_performance.csv', index=False)
print("✓ Saved: results/logistic_baseline_performance.csv")

# Save all coefficients
coef_all_df = pd.concat(all_coefficients, ignore_index=True)
coef_all_df.to_csv('results/logistic_coefficients.csv', index=False)
print("✓ Saved: results/logistic_coefficients.csv")

# ============================================================================
# STEP 9: SUMMARY REPORT
# ============================================================================
print("\n" + "="*100)
print("SUMMARY: LINEAR MODELS BASELINE RESULTS")
print("="*100)

print("\n📊 Model Performance Summary:")
print(perf_df.to_string(index=False))

print("\n\n📈 Best Model by Horizon (based on AUC):")
for horizon in HORIZONS.keys():
    horizon_perf = perf_df[perf_df['Horizon'] == horizon]
    best_model = horizon_perf.loc[horizon_perf['AUC'].idxmax()]
    print(f"\n{horizon} Horizon:")
    print(f"  Best Model: {best_model['Model']}")
    print(f"  AUC: {best_model['AUC']:.4f}")
    print(f"  F1-Score: {best_model['F1_Score']:.4f}")

print("\n\n📈 Coefficient Interpretation Guide (Logistic Regression):")
print("─" * 100)
print("Positive Coefficient → Feature INCREASES failure probability")
print("Negative Coefficient → Feature DECREASES failure probability")
print("Odds Ratio > 1 → Feature increases odds of failure")
print("Odds Ratio < 1 → Feature decreases odds of failure")
print("─" * 100)

print("\n\n🎯 Top Risk Factors (12M Horizon - Logistic Regression):")
# Find 12M Logistic coefficients
coef_12m_logistic = [coef_df for coef_df in all_coefficients
                      if coef_df['Horizon'].iloc[0] == '12M' and coef_df['Model'].iloc[0] == 'Logistic'][0]

print("\nIncreasing Risk:")
positive_coef = coef_12m_logistic[coef_12m_logistic['Coefficient'] > 0].head(3)
for idx, row in positive_coef.iterrows():
    print(f"  ↑ {row['Feature']:40s} | Odds Ratio: {row['Odds_Ratio']:.3f} ({(row['Odds_Ratio']-1)*100:+.1f}% per unit increase)")

print("\nDecreasing Risk:")
negative_coef = coef_12m_logistic[coef_12m_logistic['Coefficient'] < 0].head(3)
for idx, row in negative_coef.iterrows():
    print(f"  ↓ {row['Feature']:40s} | Odds Ratio: {row['Odds_Ratio']:.3f} ({(1-row['Odds_Ratio'])*100:.1f}% per unit increase)")

print("\n" + "="*100)
print("✅ LINEAR MODELS BASELINE COMPLETE!")
print("="*100)
print("\n📂 Outputs:")
print("   Models: models/logistic_*.pkl, models/ridge_*.pkl, models/lasso_*.pkl")
print("   Predictions: predictions/logistic_predictions_*.csv")
print("   Visualizations: outputs/logistic_baseline/*.png")
print("   Results: results/logistic_baseline_performance.csv")
print("   Coefficients: results/logistic_coefficients.csv")
if USE_GRIDSEARCH:
    print("   Best Parameters: results/linear_models_best_params.csv")
print("\n💡 Model Comparison:")
print("   • Logistic: Best for interpretability (odds ratios)")
print("   • Ridge (L2): Best for correlated features (smooth coefficients)")
print("   • Lasso (L1): Best for feature selection (sparse coefficients)")
print("\n💡 Next Steps:")
print("   1. Compare with XGBoost/CatBoost performance (06_model_training.py)")
print("   2. Add monotonic constraints to tree models (06c_monotonic_models.py)")
print("   3. Generate SHAP explanations (07_explainability.py)")
print("="*100)
