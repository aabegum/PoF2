"""
LOGISTIC REGRESSION BASELINE - POF PREDICTION
Turkish EDAŞ PoF Prediction Project

Purpose:
- Train interpretable Logistic Regression baseline models
- Provide coefficient-based explanations (odds ratios)
- Benchmark for complex models (XGBoost/CatBoost)
- Enable business stakeholder understanding

Strategy:
- L2 regularization to prevent overfitting
- Balanced class weights (handle imbalance)
- 70/30 train/test split with stratification
- Feature importance via coefficients

Input:  data/features_selected_clean.csv (11 features)
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
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
print(" "*28 + "Interpretable Models | 3/6/12 Months")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_FOLDS = 5

# Prediction horizons (days)
HORIZONS = {
    '3M': 90,
    '6M': 180,
    '12M': 365
}

# Logistic Regression parameters
LOGISTIC_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,  # Inverse regularization strength
    'solver': 'lbfgs',
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
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
print(f"   Regularization: L2 (Ridge), C={LOGISTIC_PARAMS['C']}")
print(f"   Class Weight Strategy: Balanced")

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

print("\n--- Creating Binary Targets ---")

targets = {}

for horizon_name, horizon_days in HORIZONS.items():
    # Target = 1 if equipment had ANY failure in time period
    if horizon_name == '3M' and 'Arıza_Sayısı_3ay' in df_full.columns:
        targets[horizon_name] = (df_full['Arıza_Sayısı_3ay'] > 0).astype(int)
    elif horizon_name == '6M' and 'Arıza_Sayısı_6ay' in df_full.columns:
        targets[horizon_name] = (df_full['Arıza_Sayısı_6ay'] > 0).astype(int)
    elif horizon_name == '12M' and 'Arıza_Sayısı_12ay' in df_full.columns:
        targets[horizon_name] = (df_full['Arıza_Sayısı_12ay'] > 0).astype(int)
    else:
        # Fallback: use 12M target
        targets[horizon_name] = (df_full['Arıza_Sayısı_12ay'] > 0).astype(int)

    # Add to main dataframe
    df[f'Target_{horizon_name}'] = targets[horizon_name].values

    # Print distribution
    target_dist = df[f'Target_{horizon_name}'].value_counts()
    pos_rate = target_dist.get(1, 0) / len(df) * 100

    print(f"\n{horizon_name} ({horizon_days} days) Target:")
    print(f"  No Failure (0): {target_dist.get(0, 0):,} ({100-pos_rate:.1f}%)")
    print(f"  Failure (1): {target_dist.get(1, 0):,} ({pos_rate:.1f}%)")
    print(f"  Positive Rate: {pos_rate:.1f}%")

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
# STEP 6: TRAIN LOGISTIC REGRESSION MODELS
# ============================================================================
print("\n" + "="*100)
print("STEP 6: TRAINING LOGISTIC REGRESSION MODELS")
print("="*100)

# Storage for results
models = {}
predictions = {}
performance_metrics = []
all_coefficients = []

for horizon_name in HORIZONS.keys():
    print(f"\n{'='*100}")
    print(f"TRAINING MODEL FOR {horizon_name} HORIZON")
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

    # Train model
    print(f"\n--- Training Logistic Regression ---")

    model = LogisticRegression(**LOGISTIC_PARAMS)
    model.fit(X_train, y_train)

    print(f"✓ Model trained successfully")

    # Cross-validation on training set
    print(f"\n--- Cross-Validation (Training Set) ---")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    print(f"5-Fold CV AUC Scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Evaluation metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n--- Test Set Performance ---")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

    # Store results
    models[horizon_name] = model
    predictions[horizon_name] = {
        'y_true': y_test,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

    # Save performance metrics
    performance_metrics.append({
        'Horizon': horizon_name,
        'Model': 'Logistic Regression',
        'AUC': auc,
        'Average_Precision': ap,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'CV_AUC_Mean': cv_scores.mean(),
        'CV_AUC_Std': cv_scores.std()
    })

    # Extract coefficients and calculate odds ratios
    coefficients = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0]),
        'Odds_Ratio': np.exp(model.coef_[0])
    })
    coefficients['Horizon'] = horizon_name
    coefficients['Intercept'] = model.intercept_[0]
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

    all_coefficients.append(coefficients)

    print(f"\n--- Top 5 Most Important Features (by |coefficient|) ---")
    for idx, row in coefficients.head(5).iterrows():
        direction = "↑" if row['Coefficient'] > 0 else "↓"
        print(f"  {direction} {row['Feature']:40s} | Coef: {row['Coefficient']:7.3f} | Odds Ratio: {row['Odds_Ratio']:6.3f}")

    # Save model
    model_path = f'models/logistic_{horizon_name.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved: {model_path}")

    # Save predictions for test set
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
    print(f"✓ Predictions saved: {pred_path}")

# ============================================================================
# STEP 7: VISUALIZE MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*100)
print("STEP 7: VISUALIZING MODEL PERFORMANCE")
print("="*100)

# 1. ROC Curves for all horizons
plt.figure(figsize=(10, 8))

for horizon_name in HORIZONS.keys():
    y_test = predictions[horizon_name]['y_true']
    y_pred_proba = predictions[horizon_name]['y_pred_proba']

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr, tpr, label=f'{horizon_name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Logistic Regression (All Horizons)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/logistic_baseline/roc_curves_all_horizons.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/logistic_baseline/roc_curves_all_horizons.png")
plt.close()

# 2. Precision-Recall Curves
plt.figure(figsize=(10, 8))

for horizon_name in HORIZONS.keys():
    y_test = predictions[horizon_name]['y_true']
    y_pred_proba = predictions[horizon_name]['y_pred_proba']

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)

    plt.plot(recall, precision, label=f'{horizon_name} (AP = {ap:.3f})', linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Logistic Regression (All Horizons)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/logistic_baseline/pr_curves_all_horizons.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/logistic_baseline/pr_curves_all_horizons.png")
plt.close()

# 3. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, horizon_name in enumerate(HORIZONS.keys()):
    y_test = predictions[horizon_name]['y_true']
    y_pred = predictions[horizon_name]['y_pred']

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

# 4. Feature Coefficients Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, horizon_name in enumerate(HORIZONS.keys()):
    coef_df = all_coefficients[idx].head(10).copy()

    # Sort by coefficient value for better visualization
    coef_df = coef_df.sort_values('Coefficient')

    colors = ['red' if x < 0 else 'green' for x in coef_df['Coefficient']]

    axes[idx].barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, alpha=0.7)
    axes[idx].set_yticks(range(len(coef_df)))
    axes[idx].set_yticklabels(coef_df['Feature'], fontsize=8)
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
print("SUMMARY: LOGISTIC REGRESSION BASELINE RESULTS")
print("="*100)

print("\n📊 Model Performance Summary:")
print(perf_df.to_string(index=False))

print("\n\n📈 Coefficient Interpretation Guide:")
print("─" * 100)
print("Positive Coefficient → Feature INCREASES failure probability")
print("Negative Coefficient → Feature DECREASES failure probability")
print("Odds Ratio > 1 → Feature increases odds of failure")
print("Odds Ratio < 1 → Feature decreases odds of failure")
print("─" * 100)

print("\n\n🎯 Top Risk Factors (12M Horizon):")
coef_12m = all_coefficients[2]  # 12M is index 2
print("\nIncreasing Risk:")
positive_coef = coef_12m[coef_12m['Coefficient'] > 0].head(3)
for idx, row in positive_coef.iterrows():
    print(f"  ↑ {row['Feature']:40s} | Odds Ratio: {row['Odds_Ratio']:.3f} ({(row['Odds_Ratio']-1)*100:+.1f}% per unit increase)")

print("\nDecreasing Risk:")
negative_coef = coef_12m[coef_12m['Coefficient'] < 0].head(3)
for idx, row in negative_coef.iterrows():
    print(f"  ↓ {row['Feature']:40s} | Odds Ratio: {row['Odds_Ratio']:.3f} ({(1-row['Odds_Ratio'])*100:.1f}% per unit increase)")

print("\n" + "="*100)
print("✅ LOGISTIC REGRESSION BASELINE COMPLETE!")
print("="*100)
print("\n📂 Outputs:")
print("   Models: models/logistic_*.pkl")
print("   Predictions: predictions/logistic_predictions_*.csv")
print("   Visualizations: outputs/logistic_baseline/*.png")
print("   Results: results/logistic_baseline_performance.csv")
print("   Coefficients: results/logistic_coefficients.csv")
print("\n💡 Next Steps:")
print("   1. Compare with XGBoost/CatBoost performance (06_model_training.py)")
print("   2. Add monotonic constraints to tree models (06c_monotonic_models.py)")
print("   3. Generate SHAP explanations (07_explainability.py)")
print("="*100)
