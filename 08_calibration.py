"""
MODEL CALIBRATION - POF PREDICTION
Turkish EDAŞ PoF Prediction Project

Purpose:
- Assess probability calibration of trained models
- Apply Isotonic or Platt scaling to improve calibration
- Ensure predicted probabilities match actual failure rates
- Critical for accurate risk scoring and maintenance budgeting

Why Calibration Matters:
- Model says "70% risk" → Should see ~70% actual failure rate
- Poor calibration → Wrong maintenance priorities
- Good calibration → Accurate budget planning

Strategy:
- Test calibration with reliability diagrams
- Apply Isotonic scaling (non-parametric, flexible)
- Apply Platt scaling (parametric, logistic)
- Compare calibrated vs uncalibrated performance

Input:  models/monotonic_*.pkl, data/features_selected_clean.csv
Output: models/calibrated_*.pkl, outputs/calibration/*.png

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import centralized configuration
from config import (
    FEATURES_REDUCED_FILE,
    FEATURES_ENGINEERED_FILE,
    MODEL_DIR,
    PREDICTION_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    HORIZONS
)

# Calibration libraries
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*100)
print(" "*30 + "POF MODEL CALIBRATION")
print(" "*25 + "Isotonic & Platt Scaling | Accurate Probabilities")
print("="*100)

# ============================================================================
# CONFIGURATION (Imported from config.py)
# ============================================================================

# Parameters (from config.py): RANDOM_STATE, TEST_SIZE, HORIZONS

N_BINS = 10  # Number of bins for calibration curve

# Target thresholds based on lifetime failure count
# Based on data: All 1148 equipment have >= 1 failure, 245 have >= 2, 104 have >= 3
TARGET_THRESHOLDS = {
    '6M': 2,   # At least 2 lifetime failures → 245/1148 = 21.3% positive
    '12M': 2   # At least 2 lifetime failures → 245/1148 = 21.3% positive
}

# Calibration methods
CALIBRATION_METHODS = ['isotonic', 'sigmoid']  # isotonic = Isotonic, sigmoid = Platt

# Create output directories
MODEL_DIR.mkdir(exist_ok=True)
calibration_dir = OUTPUT_DIR / 'calibration'
calibration_dir.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("\n📋 Configuration:")
print(f"   Random State: {RANDOM_STATE}")
print(f"   Train/Test Split: {100-TEST_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"   Calibration Bins: {N_BINS}")
print(f"   Calibration Methods: Isotonic + Platt (Sigmoid)")
print(f"   Horizons: {HORIZONS}")
print(f"   Target Thresholds: {TARGET_THRESHOLDS}")
print(f"\n⚠️  NOTE: 3M horizon removed (100% positive class - all equipment has >= 1 lifetime failure)")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING DATA")
print("="*100)

data_path = FEATURES_REDUCED_FILE

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("Please run 05b_remove_leaky_features.py first!")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# Load full data for target creation
df_full = pd.read_csv(FEATURES_ENGINEERED_FILE)
print(f"✓ Loaded full data: {df_full.shape[0]:,} equipment")

# ============================================================================
# STEP 2: CREATE TARGETS & PREPARE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: CREATING TARGETS & PREPARING FEATURES")
print("="*100)

# Verify required column exists
if 'Toplam_Arıza_Sayisi_Lifetime' not in df_full.columns:
    print("\n❌ ERROR: 'Toplam_Arıza_Sayisi_Lifetime' not found in features_engineered.csv")
    print("Please run 02_data_transformation.py first!")
    exit(1)

# Create targets (lifetime-based to prevent data leakage)
print("\n--- Creating Binary Targets (Lifetime-Based) ---")
print("Strategy: Equipment with X+ lifetime failures → high risk")

for horizon in HORIZONS:
    threshold = TARGET_THRESHOLDS[horizon]

    # Target = 1 if equipment has threshold or more lifetime failures
    df[f'Target_{horizon}'] = (df_full['Toplam_Arıza_Sayisi_Lifetime'] >= threshold).astype(int)

    # Print distribution
    target_dist = df[f'Target_{horizon}'].value_counts()
    pos_rate = target_dist.get(1, 0) / len(df) * 100

    print(f"\n{horizon} - Threshold: {threshold}+ lifetime failures")
    print(f"  Low Risk (0): {target_dist.get(0, 0):,} ({100-pos_rate:.1f}%)")
    print(f"  High Risk (1): {target_dist.get(1, 0):,} ({pos_rate:.1f}%)")
    print(f"  Positive Rate: {pos_rate:.1f}%")

# Prepare features
id_column = 'Ekipman_ID'

# Dynamically detect categorical features (don't hardcode!)
categorical_features = []
for col in df.columns:
    if col != id_column and not col.startswith('Target_'):
        # Check if column is categorical/object type
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)

# If no categorical features detected, add known ones that exist
known_categoricals = ['Equipment_Class_Primary', 'Risk_Category', 'Voltage_Class', 'Bölge_Tipi']
for cat in known_categoricals:
    if cat in df.columns and cat not in categorical_features:
        categorical_features.append(cat)

print(f"\n✓ Detected categorical features: {categorical_features}")

feature_columns = [col for col in df.columns
                   if col != id_column
                   and not col.startswith('Target_')
                   and col not in categorical_features]

# Encode categorical features
df_encoded = df.copy()
label_encoders = {}

for cat_feat in categorical_features:
    le = LabelEncoder()
    df_encoded[cat_feat] = le.fit_transform(df_encoded[cat_feat].astype(str))
    label_encoders[cat_feat] = le

all_features = feature_columns + categorical_features
X = df_encoded[all_features].copy()

print(f"\n✓ Total features: {len(all_features)}")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*100)
print("STEP 3: CREATING TRAIN/TEST SPLITS")
print("="*100)

# Use same train/test split for all horizons
y_12m = df_encoded['Target_12M'].copy()

X_train, X_test, _, _ = train_test_split(
    X, y_12m,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_12m
)

train_idx = X_train.index
test_idx = X_test.index

print(f"\n✓ Data Split:")
print(f"   Training set: {len(train_idx):,} equipment")
print(f"   Test set: {len(test_idx):,} equipment")

# ============================================================================
# STEP 4: LOAD UNCALIBRATED MODELS
# ============================================================================
print("\n" + "="*100)
print("STEP 4: LOADING UNCALIBRATED MODELS")
print("="*100)

uncalibrated_models = {}

for horizon in HORIZONS:
    model_path = MODEL_DIR / f'monotonic_xgboost_{horizon.lower()}.pkl'

    if Path(model_path).exists():
        with open(model_path, 'rb') as f:
            uncalibrated_models[horizon] = pickle.load(f)
        print(f"✓ Loaded XGBoost model: {horizon}")
    else:
        print(f"⚠️  Model not found: {model_path}")
        print(f"   Run 06c_monotonic_models.py first!")
        exit(1)

# ============================================================================
# STEP 5: ASSESS CALIBRATION (BEFORE)
# ============================================================================
print("\n" + "="*100)
print("STEP 5: ASSESSING CALIBRATION (UNCALIBRATED MODELS)")
print("="*100)

uncalibrated_metrics = []

for horizon in HORIZONS:
    print(f"\n--- {horizon} Horizon ---")

    model = uncalibrated_models[horizon]
    y_train = df_encoded.loc[train_idx, f'Target_{horizon}'].values
    y_test = df_encoded.loc[test_idx, f'Target_{horizon}'].values

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"AUC-ROC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f} (lower is better)")
    print(f"Log Loss: {logloss:.4f} (lower is better)")

    # Calibration curve
    fraction_positives, mean_predicted = calibration_curve(
        y_test, y_pred_proba, n_bins=N_BINS, strategy='uniform'
    )

    # Calculate calibration error (mean absolute deviation from diagonal)
    calibration_error = np.mean(np.abs(fraction_positives - mean_predicted))
    print(f"Calibration Error: {calibration_error:.4f} (lower is better)")

    uncalibrated_metrics.append({
        'Horizon': horizon,
        'Model': 'Uncalibrated',
        'AUC': auc,
        'Brier_Score': brier,
        'Log_Loss': logloss,
        'Calibration_Error': calibration_error
    })

# ============================================================================
# STEP 6: CALIBRATE MODELS
# ============================================================================
print("\n" + "="*100)
print("STEP 6: CALIBRATING MODELS")
print("="*100)

calibrated_models = {}
calibrated_metrics = []

for horizon in HORIZONS:
    print(f"\n{'='*100}")
    print(f"CALIBRATING MODEL FOR {horizon} HORIZON")
    print(f"{'='*100}")

    base_model = uncalibrated_models[horizon]
    y_train = df_encoded.loc[train_idx, f'Target_{horizon}'].values
    y_test = df_encoded.loc[test_idx, f'Target_{horizon}'].values

    calibrated_models[horizon] = {}

    for method in CALIBRATION_METHODS:
        method_name = 'Isotonic' if method == 'isotonic' else 'Platt'
        print(f"\n--- {method_name} Calibration ---")

        # Calibrate using validation set from training data
        # Use prefit to avoid retraining the base model
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method=method,
            cv='prefit'
        )

        # Fit calibrator on training data
        calibrated_model.fit(X_train, y_train)

        print(f"✓ {method_name} calibration applied")

        # Predictions
        y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba_cal)
        brier = brier_score_loss(y_test, y_pred_proba_cal)
        logloss = log_loss(y_test, y_pred_proba_cal)

        # Calibration curve
        fraction_positives, mean_predicted = calibration_curve(
            y_test, y_pred_proba_cal, n_bins=N_BINS, strategy='uniform'
        )

        calibration_error = np.mean(np.abs(fraction_positives - mean_predicted))

        print(f"AUC-ROC: {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"Calibration Error: {calibration_error:.4f}")

        # Store calibrated model
        calibrated_models[horizon][method] = calibrated_model

        # Save calibrated model
        model_path = MODEL_DIR / f'calibrated_{method}_{horizon.lower()}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(calibrated_model, f)
        print(f"✓ Saved: {model_path}")

        calibrated_metrics.append({
            'Horizon': horizon,
            'Model': f'{method_name} Calibrated',
            'AUC': auc,
            'Brier_Score': brier,
            'Log_Loss': logloss,
            'Calibration_Error': calibration_error
        })

# ============================================================================
# STEP 7: VISUALIZE CALIBRATION CURVES
# ============================================================================
print("\n" + "="*100)
print("STEP 7: VISUALIZING CALIBRATION CURVES")
print("="*100)

# 1. Calibration curves for all horizons (Before vs After)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, horizon in enumerate(HORIZONS):
    y_test = df_encoded.loc[test_idx, f'Target_{horizon}'].values

    # Uncalibrated
    uncal_model = uncalibrated_models[horizon]
    uncal_proba = uncal_model.predict_proba(X_test)[:, 1]
    frac_pos_uncal, mean_pred_uncal = calibration_curve(
        y_test, uncal_proba, n_bins=N_BINS, strategy='uniform'
    )

    # Isotonic calibrated
    iso_model = calibrated_models[horizon]['isotonic']
    iso_proba = iso_model.predict_proba(X_test)[:, 1]
    frac_pos_iso, mean_pred_iso = calibration_curve(
        y_test, iso_proba, n_bins=N_BINS, strategy='uniform'
    )

    # Platt calibrated
    platt_model = calibrated_models[horizon]['sigmoid']
    platt_proba = platt_model.predict_proba(X_test)[:, 1]
    frac_pos_platt, mean_pred_platt = calibration_curve(
        y_test, platt_proba, n_bins=N_BINS, strategy='uniform'
    )

    # Plot
    axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    axes[idx].plot(mean_pred_uncal, frac_pos_uncal, 'o-', label='Uncalibrated', linewidth=2, markersize=6)
    axes[idx].plot(mean_pred_iso, frac_pos_iso, 's-', label='Isotonic', linewidth=2, markersize=6)
    axes[idx].plot(mean_pred_platt, frac_pos_platt, '^-', label='Platt', linewidth=2, markersize=6)

    axes[idx].set_xlabel('Mean Predicted Probability', fontsize=10)
    axes[idx].set_ylabel('Fraction of Positives', fontsize=10)
    axes[idx].set_title(f'{horizon} Horizon', fontsize=12, fontweight='bold')
    axes[idx].legend(loc='upper left', fontsize=9)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim([0, 1])
    axes[idx].set_ylim([0, 1])

plt.suptitle('Calibration Curves: Uncalibrated vs Calibrated Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(calibration_dir / 'calibration_curves_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/calibration/calibration_curves_comparison.png")
plt.close()

# 2. Calibration Error Comparison (Bar Chart)
all_metrics = uncalibrated_metrics + calibrated_metrics
metrics_df = pd.DataFrame(all_metrics)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calibration Error
cal_error_pivot = metrics_df.pivot(index='Horizon', columns='Model', values='Calibration_Error')
cal_error_pivot.plot(kind='bar', ax=axes[0], width=0.7)
axes[0].set_title('Calibration Error Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Calibration Error (lower is better)', fontsize=10)
axes[0].set_xlabel('Horizon', fontsize=10)
axes[0].legend(title='Model', fontsize=8, loc='upper right')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Brier Score
brier_pivot = metrics_df.pivot(index='Horizon', columns='Model', values='Brier_Score')
brier_pivot.plot(kind='bar', ax=axes[1], width=0.7)
axes[1].set_title('Brier Score Comparison', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Brier Score (lower is better)', fontsize=10)
axes[1].set_xlabel('Horizon', fontsize=10)
axes[1].legend(title='Model', fontsize=8, loc='upper right')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(calibration_dir / 'calibration_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/calibration/calibration_metrics_comparison.png")
plt.close()

# 3. Reliability Diagrams (Detailed)
for horizon in HORIZONS:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    y_test = df_encoded.loc[test_idx, f'Target_{horizon}'].values

    models_to_plot = [
        ('Uncalibrated', uncalibrated_models[horizon]),
        ('Isotonic', calibrated_models[horizon]['isotonic']),
        ('Platt', calibrated_models[horizon]['sigmoid'])
    ]

    for idx, (name, model) in enumerate(models_to_plot):
        proba = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=N_BINS, strategy='uniform')

        # Reliability diagram with histogram
        ax = axes[idx]
        ax2 = ax.twinx()

        # Calibration curve
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
        ax.plot(mean_pred, frac_pos, 'o-', label='Model', linewidth=2, markersize=8, color='steelblue')

        # Histogram of predictions
        ax2.hist(proba, bins=N_BINS, alpha=0.3, color='orange', label='Distribution')

        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax.set_ylabel('Fraction of Positives', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.suptitle(f'Reliability Diagrams - {horizon} Horizon', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(calibration_dir / f'reliability_diagram_{horizon.lower()}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: outputs/calibration/reliability_diagram_{horizon.lower()}.png")
    plt.close()

# ============================================================================
# STEP 8: GENERATE CALIBRATED PREDICTIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 8: GENERATING CALIBRATED PREDICTIONS")
print("="*100)

for horizon in HORIZONS:
    print(f"\n--- {horizon} Horizon ---")

    # Use Isotonic calibration (typically best for tree models)
    calibrated_model = calibrated_models[horizon]['isotonic']

    # Generate predictions for ALL equipment
    y_pred_proba = calibrated_model.predict_proba(X)[:, 1]

    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'Ekipman_ID': df[id_column].values,
        'Equipment_Class': df['Equipment_Class_Primary'].values,
        'Calibrated_Failure_Probability': y_pred_proba,
        'Risk_Score': (y_pred_proba * 100).round(2)
    })

    # Add risk levels
    pred_df['Risk_Level'] = pd.cut(
        pred_df['Risk_Score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    # Sort by risk score
    pred_df = pred_df.sort_values('Risk_Score', ascending=False)

    # Save
    pred_path = PREDICTION_DIR / f'calibrated_predictions_{horizon.lower()}.csv'
    pred_df.to_csv(pred_path, index=False)
    print(f"✓ Saved: {pred_path}")

    # Show top 10 high-risk
    print(f"\nTop 10 High-Risk Equipment ({horizon}):")
    print(pred_df.head(10)[['Ekipman_ID', 'Equipment_Class', 'Risk_Score', 'Risk_Level']].to_string(index=False))

# ============================================================================
# STEP 9: SAVE CALIBRATION RESULTS
# ============================================================================
print("\n" + "="*100)
print("STEP 9: SAVING CALIBRATION RESULTS")
print("="*100)

# Save metrics
metrics_df.to_csv(RESULTS_DIR / 'calibration_metrics.csv', index=False)
print("✓ Saved: results/calibration_metrics.csv")

# Calculate improvements
improvements = []

for horizon in HORIZONS:
    uncal = metrics_df[(metrics_df['Horizon'] == horizon) & (metrics_df['Model'] == 'Uncalibrated')].iloc[0]
    iso = metrics_df[(metrics_df['Horizon'] == horizon) & (metrics_df['Model'] == 'Isotonic Calibrated')].iloc[0]
    platt = metrics_df[(metrics_df['Horizon'] == horizon) & (metrics_df['Model'] == 'Platt Calibrated')].iloc[0]

    improvements.append({
        'Horizon': horizon,
        'Uncalibrated_Cal_Error': uncal['Calibration_Error'],
        'Isotonic_Cal_Error': iso['Calibration_Error'],
        'Platt_Cal_Error': platt['Calibration_Error'],
        'Isotonic_Improvement': ((uncal['Calibration_Error'] - iso['Calibration_Error']) / uncal['Calibration_Error'] * 100),
        'Platt_Improvement': ((uncal['Calibration_Error'] - platt['Calibration_Error']) / uncal['Calibration_Error'] * 100)
    })

improvements_df = pd.DataFrame(improvements)
improvements_df.to_csv(RESULTS_DIR / 'calibration_improvements.csv', index=False)
print("✓ Saved: results/calibration_improvements.csv")

# ============================================================================
# STEP 10: SUMMARY REPORT
# ============================================================================
print("\n" + "="*100)
print("SUMMARY: MODEL CALIBRATION RESULTS")
print("="*100)

print("\n📊 Calibration Metrics Comparison:")
print(metrics_df.to_string(index=False))

print("\n\n📈 Calibration Improvements:")
print(improvements_df.to_string(index=False))

print("\n\n🎯 Recommended Calibrated Models:")
for horizon in HORIZONS:
    horizon_metrics = metrics_df[metrics_df['Horizon'] == horizon]
    best_model = horizon_metrics.loc[horizon_metrics['Calibration_Error'].idxmin()]
    print(f"\n{horizon} Horizon:")
    print(f"  Best Model: {best_model['Model']}")
    print(f"  Calibration Error: {best_model['Calibration_Error']:.4f}")
    print(f"  Brier Score: {best_model['Brier_Score']:.4f}")
    print(f"  AUC: {best_model['AUC']:.4f}")

print("\n\n💡 Key Insights:")
print("─" * 100)
print("1. Calibration improves probability accuracy without changing rankings (AUC stays same)")
print("2. Isotonic calibration typically best for tree models with sufficient data")
print("3. Platt scaling better for small samples or parametric approaches")
print("4. Lower calibration error = predictions closer to actual failure rates")
print("5. Calibrated probabilities essential for accurate risk scoring and budgeting")
print("─" * 100)

print("\n\n📋 Calibration Interpretation:")
print("─" * 100)
print("Before Calibration:")
print("  - Model predicts 70% risk")
print("  - Actual failure rate might be 50% or 90% (miscalibrated)")
print("\nAfter Calibration:")
print("  - Model predicts 70% risk")
print("  - Actual failure rate ≈ 70% (well-calibrated)")
print("  - Can confidently budget for expected failures")
print("─" * 100)

print("\n" + "="*100)
print("✅ MODEL CALIBRATION COMPLETE!")
print("="*100)
print("\n📂 Outputs:")
print("   Calibrated Models: models/calibrated_*.pkl")
print("   Calibrated Predictions: predictions/calibrated_predictions_*.csv")
print("   Calibration Curves: outputs/calibration/calibration_curves_comparison.png")
print("   Reliability Diagrams: outputs/calibration/reliability_diagram_*.png")
print("   Metrics: results/calibration_metrics.csv")
print("   Improvements: results/calibration_improvements.csv")
print("\n💡 Benefits:")
print("   ✓ Accurate risk probabilities (70% prediction = 70% actual)")
print("   ✓ Better maintenance budget planning")
print("   ✓ Increased confidence in model predictions")
print("   ✓ Improved decision-making for prioritization")
print("\n💡 Deployment:")
print("   - Use calibrated_isotonic_*.pkl models for production")
print("   - Risk scores now reflect true failure probabilities")
print("   - Share calibration_curves with stakeholders for transparency")
print("="*100)
