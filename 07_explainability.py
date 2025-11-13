"""
MODEL EXPLAINABILITY - SHAP ANALYSIS
Turkish EDAŞ PoF Prediction Project

Purpose:
- Generate SHAP explanations for trained models
- Create global feature importance plots
- Generate individual equipment explanations
- Build trust with field engineers and management

Outputs:
- SHAP summary plots (global feature importance)
- SHAP waterfall plots (individual predictions)
- SHAP dependence plots (feature effects)
- Risk explanation reports for high-risk equipment

Strategy:
- Use TreeExplainer for XGBoost/CatBoost (fast, exact)
- Generate explanations for top 100 high-risk equipment
- Create visual reports for stakeholders

Input:  models/monotonic_*.pkl, data/features_selected_clean.csv
Output: outputs/explainability/*.png, reports/risk_explanations.csv

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

# SHAP library
import shap

# Model libraries
from sklearn.preprocessing import LabelEncoder

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*100)
print(" "*30 + "POF MODEL EXPLAINABILITY")
print(" "*25 + "SHAP Analysis | Understanding Risk Factors")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
N_BACKGROUND_SAMPLES = 100  # For SHAP explainer (use subset for speed)
N_HIGH_RISK_EXAMPLES = 20   # Number of individual explanations

# Prediction horizons
# NOTE: 3M removed (100% positive class - all equipment has >= 1 lifetime failure)
HORIZONS = ['6M', '12M']

# Create output directories
Path('outputs/explainability').mkdir(parents=True, exist_ok=True)
Path('outputs/explainability/waterfall').mkdir(parents=True, exist_ok=True)
Path('outputs/explainability/dependence').mkdir(parents=True, exist_ok=True)
Path('reports').mkdir(exist_ok=True)

print("\n📋 Configuration:")
print(f"   SHAP Background Samples: {N_BACKGROUND_SAMPLES}")
print(f"   High-Risk Examples: {N_HIGH_RISK_EXAMPLES}")
print(f"   Horizons: {HORIZONS}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING DATA & MODELS")
print("="*100)

# Load data
data_path = Path('data/features_selected_clean.csv')

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("Please run 05b_remove_leaky_features.py first!")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# ============================================================================
# STEP 2: PREPARE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: PREPARING FEATURES")
print("="*100)

# Identify feature types
id_column = 'Ekipman_ID'
categorical_features = ['Equipment_Class_Primary', 'Risk_Category']

# Numeric features
feature_columns = [col for col in df.columns
                   if col != id_column
                   and col not in categorical_features]

# Encode categorical features
df_encoded = df.copy()
label_encoders = {}

for cat_feat in categorical_features:
    le = LabelEncoder()
    df_encoded[cat_feat] = le.fit_transform(df_encoded[cat_feat].astype(str))
    label_encoders[cat_feat] = le
    print(f"✓ Encoded {cat_feat}: {len(le.classes_)} unique values")

# All features for modeling
all_features = feature_columns + categorical_features
X = df_encoded[all_features].copy()

print(f"\n✓ Total features: {len(all_features)}")

# ============================================================================
# STEP 3: LOAD MODELS & GENERATE PREDICTIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 3: LOADING MODELS & GENERATING PREDICTIONS")
print("="*100)

models = {}
predictions = {}

for horizon in HORIZONS:
    model_path = f'models/monotonic_xgboost_{horizon.lower()}.pkl'

    if Path(model_path).exists():
        with open(model_path, 'rb') as f:
            models[horizon] = pickle.load(f)
        print(f"✓ Loaded XGBoost model: {horizon}")

        # Generate predictions
        pred_proba = models[horizon].predict_proba(X)[:, 1]
        predictions[horizon] = pred_proba

    else:
        print(f"⚠️  Model not found: {model_path}")
        print(f"   Run 06c_monotonic_models.py first!")
        exit(1)

# ============================================================================
# STEP 4: CREATE SHAP EXPLAINERS
# ============================================================================
print("\n" + "="*100)
print("STEP 4: CREATING SHAP EXPLAINERS")
print("="*100)

print("\n⚠️  NOTE: Using TreeExplainer (exact, fast for tree models)")

explainers = {}
shap_values_dict = {}

for horizon in HORIZONS:
    print(f"\n--- Creating explainer for {horizon} ---")

    # Create explainer with background data (use subset for speed)
    background_data = shap.sample(X, N_BACKGROUND_SAMPLES, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(models[horizon], background_data)
    explainers[horizon] = explainer

    print(f"✓ TreeExplainer created")

    # Calculate SHAP values for all data
    print(f"Calculating SHAP values for {len(X)} equipment...")
    shap_values = explainer.shap_values(X)
    shap_values_dict[horizon] = shap_values

    print(f"✓ SHAP values calculated")

# ============================================================================
# STEP 5: GLOBAL FEATURE IMPORTANCE (SUMMARY PLOTS)
# ============================================================================
print("\n" + "="*100)
print("STEP 5: GENERATING GLOBAL FEATURE IMPORTANCE")
print("="*100)

# 1. Summary Plot (Bee Swarm) for each horizon
for horizon in HORIZONS:
    print(f"\n--- {horizon} Summary Plot ---")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_dict[horizon],
        X,
        feature_names=all_features,
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Feature Importance - {horizon} Horizon', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'outputs/explainability/shap_summary_{horizon.lower()}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: outputs/explainability/shap_summary_{horizon.lower()}.png")
    plt.close()

# 2. Mean Absolute SHAP Values (Bar Plot)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, horizon in enumerate(HORIZONS):
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values_dict[horizon]).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Mean_SHAP': mean_shap
    }).sort_values('Mean_SHAP', ascending=False).head(10)

    # Plot
    axes[idx].barh(range(len(feature_importance)), feature_importance['Mean_SHAP'], color='steelblue', alpha=0.7)
    axes[idx].set_yticks(range(len(feature_importance)))
    axes[idx].set_yticklabels(feature_importance['Feature'], fontsize=9)
    axes[idx].set_xlabel('Mean |SHAP Value|', fontsize=10)
    axes[idx].set_title(f'{horizon} Horizon', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/explainability/shap_importance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/explainability/shap_importance_comparison.png")
plt.close()

# ============================================================================
# STEP 6: DEPENDENCE PLOTS (TOP FEATURES)
# ============================================================================
print("\n" + "="*100)
print("STEP 6: GENERATING DEPENDENCE PLOTS")
print("="*100)

print("\n⚠️  NOTE: Showing how individual features affect predictions")

for horizon in HORIZONS:
    print(f"\n--- {horizon} Dependence Plots ---")

    # Get top 4 most important features
    mean_shap = np.abs(shap_values_dict[horizon]).mean(axis=0)
    top_features_idx = np.argsort(mean_shap)[-4:][::-1]
    top_features = [all_features[i] for i in top_features_idx]

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, feat_idx in enumerate(top_features_idx):
        feat_name = all_features[feat_idx]

        # Create dependence plot
        shap.dependence_plot(
            feat_idx,
            shap_values_dict[horizon],
            X,
            feature_names=all_features,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(f'{feat_name}', fontsize=11, fontweight='bold')

    plt.suptitle(f'SHAP Dependence Plots - {horizon} Horizon', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'outputs/explainability/dependence/shap_dependence_{horizon.lower()}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: outputs/explainability/dependence/shap_dependence_{horizon.lower()}.png")
    plt.close()

# ============================================================================
# STEP 7: INDIVIDUAL EXPLANATIONS (HIGH-RISK EQUIPMENT)
# ============================================================================
print("\n" + "="*100)
print("STEP 7: GENERATING INDIVIDUAL EXPLANATIONS")
print("="*100)

# Focus on 12M predictions (most comprehensive)
horizon = '12M'
print(f"\n--- Analyzing Top {N_HIGH_RISK_EXAMPLES} High-Risk Equipment ({horizon}) ---")

# Get high-risk equipment indices
risk_scores = predictions[horizon] * 100
high_risk_idx = np.argsort(risk_scores)[-N_HIGH_RISK_EXAMPLES:][::-1]

# Create waterfall plots for top high-risk equipment
for rank, idx in enumerate(high_risk_idx[:5], 1):  # Show top 5 in plots
    equipment_id = df.loc[idx, id_column]
    risk_score = risk_scores[idx]

    print(f"\n{rank}. Equipment {equipment_id} | Risk Score: {risk_score:.1f}")

    # Create waterfall plot
    plt.figure(figsize=(10, 6))

    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values_dict[horizon][idx],
        base_values=explainers[horizon].expected_value,
        data=X.iloc[idx].values,
        feature_names=all_features
    )

    shap.waterfall_plot(explanation, show=False, max_display=10)
    plt.title(f'Risk Explanation: Equipment {equipment_id}\nRisk Score: {risk_score:.1f}/100 ({horizon} Horizon)',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'outputs/explainability/waterfall/waterfall_{equipment_id}_{horizon.lower()}.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved waterfall plot for Equipment {equipment_id}")
    plt.close()

# ============================================================================
# STEP 8: CREATE RISK EXPLANATION REPORT
# ============================================================================
print("\n" + "="*100)
print("STEP 8: CREATING RISK EXPLANATION REPORT")
print("="*100)

# Create detailed explanation report for high-risk equipment
explanation_records = []

for horizon in HORIZONS:
    risk_scores = predictions[horizon] * 100
    high_risk_idx = np.argsort(risk_scores)[-100:][::-1]  # Top 100

    for idx in high_risk_idx:
        equipment_id = df.loc[idx, id_column]
        equipment_class = df.loc[idx, 'Equipment_Class_Primary']
        risk_score = risk_scores[idx]

        # Get SHAP values for this equipment
        shap_vals = shap_values_dict[horizon][idx]

        # Get top 5 contributing features (positive SHAP = increases risk)
        top_contrib_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

        # Build explanation text
        explanations = []
        for feat_idx in top_contrib_idx:
            feat_name = all_features[feat_idx]
            feat_value = X.iloc[idx, feat_idx]
            shap_value = shap_vals[feat_idx]
            contribution = shap_value * 100  # Convert to risk score contribution

            direction = "↑" if shap_value > 0 else "↓"
            explanations.append(f"{direction} {feat_name}={feat_value:.2f} ({contribution:+.1f} pts)")

        explanation_text = " | ".join(explanations)

        explanation_records.append({
            'Horizon': horizon,
            'Ekipman_ID': equipment_id,
            'Equipment_Class': equipment_class,
            'Risk_Score': round(risk_score, 2),
            'Risk_Level': 'Critical' if risk_score >= 75 else 'High',
            'Top_Risk_Factor': all_features[top_contrib_idx[0]],
            'Top_Risk_Factor_Contribution': round(shap_vals[top_contrib_idx[0]] * 100, 1),
            'Explanation': explanation_text
        })

# Create DataFrame
explanation_df = pd.DataFrame(explanation_records)

# Save report
explanation_df.to_csv('reports/risk_explanations.csv', index=False)
print(f"✓ Saved: reports/risk_explanations.csv")
print(f"   Total explanations: {len(explanation_df)}")

# ============================================================================
# STEP 9: CREATE FEATURE CONTRIBUTION HEATMAP
# ============================================================================
print("\n" + "="*100)
print("STEP 9: CREATING FEATURE CONTRIBUTION HEATMAP")
print("="*100)

# Show how features contribute across all equipment (12M horizon)
horizon = '12M'
shap_values = shap_values_dict[horizon]

# Get top 10 most important features
mean_shap = np.abs(shap_values).mean(axis=0)
top_10_idx = np.argsort(mean_shap)[-10:][::-1]
top_10_features = [all_features[i] for i in top_10_idx]

# Sample 50 random equipment for visualization
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=min(50, len(X)), replace=False)

# Create heatmap data
heatmap_data = shap_values[sample_idx][:, top_10_idx]

plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data.T,
    cmap='RdBu_r',
    center=0,
    xticklabels=[df.loc[i, id_column] for i in sample_idx],
    yticklabels=top_10_features,
    cbar_kws={'label': 'SHAP Value\n(Red=↑Risk, Blue=↓Risk)'}
)
plt.title(f'Feature Contributions to Risk (Sample Equipment) - {horizon} Horizon',
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Equipment ID', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/explainability/feature_contribution_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/explainability/feature_contribution_heatmap.png")
plt.close()

# ============================================================================
# STEP 10: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*100)
print("STEP 10: SUMMARY STATISTICS")
print("="*100)

# Calculate global feature importance across all horizons
global_importance = {}

for horizon in HORIZONS:
    mean_shap = np.abs(shap_values_dict[horizon]).mean(axis=0)

    for feat_idx, feat_name in enumerate(all_features):
        if feat_name not in global_importance:
            global_importance[feat_name] = []
        global_importance[feat_name].append(mean_shap[feat_idx])

# Average across horizons
avg_importance = {feat: np.mean(vals) for feat, vals in global_importance.items()}
sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

print("\n📊 Global Feature Importance (Average across all horizons):")
print("─" * 80)
for rank, (feat, importance) in enumerate(sorted_features, 1):
    print(f"{rank:2d}. {feat:40s} | Mean |SHAP|: {importance:.4f}")
print("─" * 80)

# Save to CSV
importance_df = pd.DataFrame({
    'Rank': range(1, len(sorted_features) + 1),
    'Feature': [f[0] for f in sorted_features],
    'Mean_SHAP_Importance': [f[1] for f in sorted_features]
})
importance_df.to_csv('results/shap_global_importance.csv', index=False)
print("\n✓ Saved: results/shap_global_importance.csv")

# ============================================================================
# STEP 11: SUMMARY REPORT
# ============================================================================
print("\n" + "="*100)
print("SUMMARY: MODEL EXPLAINABILITY ANALYSIS")
print("="*100)

print("\n🎯 Top 5 Most Important Risk Factors:")
for rank, (feat, importance) in enumerate(sorted_features[:5], 1):
    # Calculate average direction (positive = increases risk)
    avg_direction = np.mean([shap_values_dict[h][:, all_features.index(feat)] for h in HORIZONS])
    direction = "↑ Increases" if avg_direction > 0 else "↓ Decreases"
    print(f"{rank}. {feat:40s} | {direction} risk | Importance: {importance:.4f}")

print("\n\n📈 Explanation Coverage:")
print(f"   Total equipment analyzed: {len(df)}")
print(f"   High-risk equipment with detailed explanations: {len(explanation_df)}")
print(f"   Waterfall plots generated: {min(5, N_HIGH_RISK_EXAMPLES)}")

print("\n\n💡 Key Insights:")
print("─" * 100)
print("1. SHAP values show HOW MUCH each feature contributes to risk prediction")
print("2. Positive SHAP values → Feature INCREASES failure risk for that equipment")
print("3. Negative SHAP values → Feature DECREASES failure risk for that equipment")
print("4. Summary plots show global importance across all equipment")
print("5. Waterfall plots explain individual equipment risk scores")
print("─" * 100)

print("\n\n📋 Example Risk Explanation (Top High-Risk Equipment):")
print("─" * 100)
example = explanation_df[explanation_df['Horizon'] == '12M'].iloc[0]
print(f"Equipment ID: {example['Ekipman_ID']}")
print(f"Equipment Class: {example['Equipment_Class']}")
print(f"Risk Score: {example['Risk_Score']}/100 ({example['Risk_Level']})")
print(f"\nTop Risk Factor: {example['Top_Risk_Factor']}")
print(f"Contribution: {example['Top_Risk_Factor_Contribution']:+.1f} points")
print(f"\nFull Explanation:\n{example['Explanation']}")
print("─" * 100)

print("\n" + "="*100)
print("✅ MODEL EXPLAINABILITY ANALYSIS COMPLETE!")
print("="*100)
print("\n📂 Outputs:")
print("   Summary Plots: outputs/explainability/shap_summary_*.png")
print("   Importance Comparison: outputs/explainability/shap_importance_comparison.png")
print("   Dependence Plots: outputs/explainability/dependence/shap_dependence_*.png")
print("   Waterfall Plots: outputs/explainability/waterfall/waterfall_*.png")
print("   Heatmap: outputs/explainability/feature_contribution_heatmap.png")
print("   Risk Explanations: reports/risk_explanations.csv")
print("   Global Importance: results/shap_global_importance.csv")
print("\n💡 Use Cases:")
print("   1. Show field engineers WHY equipment is high-risk")
print("   2. Validate model follows domain knowledge")
print("   3. Build trust with management and stakeholders")
print("   4. Prioritize maintenance based on specific risk factors")
print("\n💡 Next Steps:")
print("   1. Calibrate model probabilities (08_calibration.py)")
print("   2. Share risk_explanations.csv with maintenance teams")
print("   3. Use waterfall plots in stakeholder presentations")
print("="*100)
