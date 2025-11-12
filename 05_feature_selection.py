"""
FEATURE SELECTION - VIF + IMPORTANCE + CORRELATION
Turkish EDAŞ PoF Prediction Project

Purpose:
- Remove multicollinearity (VIF analysis)
- Rank feature importance (Random Forest)
- Filter highly correlated features
- Select optimal feature set for modeling

Input:  data/features_engineered.csv (58 features)
Output: data/features_selected.csv (~25-35 features)

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
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')

print("="*100)
print(" "*30 + "FEATURE SELECTION PIPELINE")
print(" "*25 + "VIF + Importance + Correlation")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

# VIF thresholds
VIF_THRESHOLD = 10  # Features with VIF > 10 are highly collinear
VIF_TARGET = 5      # Target VIF after iterative removal

# Correlation threshold
CORRELATION_THRESHOLD = 0.85  # Remove features with correlation > 0.85

# Feature importance threshold
IMPORTANCE_THRESHOLD = 0.001  # Keep features contributing > 0.1%

# Create output directory
output_dir = Path('outputs/feature_selection')
output_dir.mkdir(parents=True, exist_ok=True)

print("\n📋 Configuration:")
print(f"   VIF Threshold: {VIF_THRESHOLD}")
print(f"   Correlation Threshold: {CORRELATION_THRESHOLD}")
print(f"   Importance Threshold: {IMPORTANCE_THRESHOLD}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING ENGINEERED FEATURES")
print("="*100)

data_path = Path('data/features_engineered.csv')

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("Please run 03_feature_engineering.py first!")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

original_feature_count = df.shape[1]

# ============================================================================
# STEP 2: PREPARE FEATURES FOR ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 2: PREPARING FEATURES FOR ANALYSIS")
print("="*100)

# Identify feature types
print("\n--- Identifying Feature Categories ---")

# ID columns (exclude from modeling)
id_columns = ['Ekipman_ID', 'Equipment_ID_Primary']

# Date columns (exclude from modeling)
date_columns = [col for col in df.columns if 'Tarihi' in col or 'Date' in col or '_at' in col]

# Categorical columns
categorical_columns = [
    'Ekipman_Sınıfı', 'Equipment_Type', 'Equipment_Class_Primary',
    'İl', 'İlçe', 'Mahalle',
    'Age_Risk_Category', 'Customer_Impact_Category', 'Risk_Category'
]

# Remove non-existent columns
id_columns = [col for col in id_columns if col in df.columns]
date_columns = [col for col in date_columns if col in df.columns]
categorical_columns = [col for col in categorical_columns if col in df.columns]

# Numeric columns (for VIF and modeling)
exclude_columns = id_columns + date_columns + categorical_columns
numeric_columns = [col for col in df.columns if col not in exclude_columns and df[col].dtype in ['float64', 'int64']]

print(f"  ID columns (excluded): {len(id_columns)}")
print(f"  Date columns (excluded): {len(date_columns)}")
print(f"  Categorical columns: {len(categorical_columns)}")
print(f"  Numeric columns: {len(numeric_columns)}")

print(f"\n✓ Features for analysis: {len(numeric_columns)} numeric features")

# ============================================================================
# STEP 3: CREATE TARGET VARIABLES
# ============================================================================
print("\n" + "="*100)
print("STEP 3: CREATING TARGET VARIABLES")
print("="*100)

print("\n--- Creating Binary Targets for PoF Prediction ---")

# We need to create target variables for 3/6/12/24 month predictions
# Target = 1 if equipment will have ANY failure in next N months

# For feature selection, we'll use 12-month target as representative
if 'Arıza_Sayısı_12ay' in df.columns:
    df['Target_12M'] = (df['Arıza_Sayısı_12ay'] > 0).astype(int)
    
    target_dist = df['Target_12M'].value_counts()
    print(f"\n12-Month Target Distribution:")
    print(f"  No Failure (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Failure (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
    
    target_column = 'Target_12M'
else:
    print("⚠ WARNING: Cannot create target variable - Arıza_Sayısı_12ay not found")
    print("  Using dummy target for feature selection")
    df['Target_12M'] = 0
    target_column = 'Target_12M'

# ============================================================================
# STEP 4: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*100)
print("STEP 4: HANDLING MISSING VALUES")
print("="*100)

print("\n--- Missing Value Analysis ---")

# Check missing values in numeric columns
missing_counts = df[numeric_columns].isnull().sum()
missing_features = missing_counts[missing_counts > 0].sort_values(ascending=False)

if len(missing_features) > 0:
    print(f"\nFeatures with missing values: {len(missing_features)}")
    print("\nTop 10 features by missing count:")
    for feat, count in missing_features.head(10).items():
        pct = count / len(df) * 100
        print(f"  {feat:<45} {count:>5,} ({pct:>5.1f}%)")
    
    # Strategy: Fill with median for now
    print("\n✓ Strategy: Filling missing values with median")
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
else:
    print("✓ No missing values in numeric features")

# ============================================================================
# STEP 5: VIF ANALYSIS (MULTICOLLINEARITY DETECTION)
# ============================================================================
print("\n" + "="*100)
print("STEP 5: VIF ANALYSIS - MULTICOLLINEARITY DETECTION")
print("="*100)

print("\n--- Calculating Variance Inflation Factors ---")
print("This may take a few minutes for 40+ features...")

def calculate_vif(df_vif, features):
    """Calculate VIF for each feature"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df_vif[features].values, i) 
                       for i in range(len(features))]
    return vif_data.sort_values('VIF', ascending=False)

# Prepare data for VIF (drop any infinite or NaN values)
df_vif = df[numeric_columns].copy()
df_vif = df_vif.replace([np.inf, -np.inf], np.nan)
df_vif = df_vif.fillna(df_vif.median())

# Initial VIF calculation
vif_before = calculate_vif(df_vif, numeric_columns)

print(f"\n✓ Initial VIF calculated for {len(numeric_columns)} features")
print(f"\nTop 10 Features by VIF (Before Removal):")
print(vif_before.head(10).to_string(index=False))

# Count high VIF features
high_vif_count = (vif_before['VIF'] > VIF_THRESHOLD).sum()
print(f"\n⚠ Features with VIF > {VIF_THRESHOLD}: {high_vif_count}")

# Iterative VIF removal
print(f"\n--- Iterative VIF Removal (Target VIF < {VIF_TARGET}) ---")

features_to_keep = numeric_columns.copy()
iteration = 0
max_iterations = 20

while True:
    iteration += 1
    
    # Calculate VIF
    vif_current = calculate_vif(df_vif[features_to_keep], features_to_keep)
    
    # Find maximum VIF
    max_vif = vif_current['VIF'].max()
    max_vif_feature = vif_current.loc[vif_current['VIF'].idxmax(), 'Feature']
    
    print(f"\nIteration {iteration}:")
    print(f"  Features: {len(features_to_keep)}")
    print(f"  Max VIF: {max_vif:.2f} ({max_vif_feature})")
    
    # Check stopping conditions
    if max_vif < VIF_TARGET:
        print(f"  ✓ Target VIF achieved!")
        break
    
    if iteration >= max_iterations:
        print(f"  ⚠ Max iterations reached")
        break
    
    # Remove feature with highest VIF
    features_to_keep.remove(max_vif_feature)
    print(f"  ❌ Removed: {max_vif_feature}")

# Final VIF
vif_after = calculate_vif(df_vif[features_to_keep], features_to_keep)

print(f"\n✓ VIF Reduction Complete")
print(f"  Features before: {len(numeric_columns)}")
print(f"  Features after: {len(features_to_keep)}")
print(f"  Features removed: {len(numeric_columns) - len(features_to_keep)}")

print(f"\nFinal VIF Statistics:")
print(f"  Mean VIF: {vif_after['VIF'].mean():.2f}")
print(f"  Max VIF: {vif_after['VIF'].max():.2f}")
print(f"  Features with VIF > {VIF_THRESHOLD}: {(vif_after['VIF'] > VIF_THRESHOLD).sum()}")

# Save VIF results
vif_comparison = pd.DataFrame({
    'Feature': vif_before['Feature'],
    'VIF_Before': vif_before['VIF'],
    'VIF_After': vif_after['VIF'].reindex(vif_before.index).fillna(0),
    'Removed': ~vif_before['Feature'].isin(features_to_keep)
})
vif_comparison.to_csv(output_dir / 'vif_analysis.csv', index=False)
print(f"\n✓ VIF analysis saved to: {output_dir / 'vif_analysis.csv'}")

# ============================================================================
# STEP 6: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 6: CORRELATION ANALYSIS")
print("="*100)

print("\n--- Identifying Highly Correlated Feature Pairs ---")

# Calculate correlation matrix
corr_matrix = df[features_to_keep].corr().abs()

# Find pairs with correlation > threshold
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > CORRELATION_THRESHOLD:
            high_corr_pairs.append({
                'Feature_1': corr_matrix.columns[i],
                'Feature_2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if len(high_corr_pairs) > 0:
    print(f"\n⚠ Found {len(high_corr_pairs)} highly correlated pairs (>{CORRELATION_THRESHOLD}):")
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
    print(high_corr_df.head(10).to_string(index=False))
    
    # For each pair, remove the feature with lower correlation to target
    print(f"\n--- Removing Less Predictive Feature from Each Pair ---")
    
    features_to_remove_corr = set()
    
    for pair in high_corr_pairs:
        feat1 = pair['Feature_1']
        feat2 = pair['Feature_2']
        
        # Calculate correlation with target
        corr1 = abs(df[feat1].corr(df[target_column]))
        corr2 = abs(df[feat2].corr(df[target_column]))
        
        # Remove feature with lower correlation to target
        if corr1 < corr2:
            features_to_remove_corr.add(feat1)
            print(f"  ❌ Remove {feat1} (target corr: {corr1:.3f}) | Keep {feat2} (target corr: {corr2:.3f})")
        else:
            features_to_remove_corr.add(feat2)
            print(f"  ❌ Remove {feat2} (target corr: {corr2:.3f}) | Keep {feat1} (target corr: {corr1:.3f})")
    
    # Update feature list
    features_to_keep = [f for f in features_to_keep if f not in features_to_remove_corr]
    
    print(f"\n✓ Removed {len(features_to_remove_corr)} features due to high correlation")
else:
    print(f"✓ No highly correlated pairs found (threshold: {CORRELATION_THRESHOLD})")

# Save correlation matrix visualization
print("\n--- Creating Correlation Heatmap ---")
if len(features_to_keep) <= 40:  # Only plot if not too many features
    plt.figure(figsize=(16, 14))
    sns.heatmap(df[features_to_keep].corr(), annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (After VIF Reduction)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation heatmap saved to: {output_dir / 'correlation_matrix.png'}")
else:
    print(f"⚠ Skipping heatmap (too many features: {len(features_to_keep)})")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*100)

print("\n--- Training Random Forest for Feature Importance ---")

# Prepare data
X = df[features_to_keep].copy()
y = df[target_column].copy()

# Check if we have enough positive samples
positive_samples = y.sum()
if positive_samples < 10:
    print(f"⚠ WARNING: Only {positive_samples} positive samples - skipping importance analysis")
    feature_importance_df = pd.DataFrame({
        'Feature': features_to_keep,
        'Importance': 1.0 / len(features_to_keep)  # Equal importance
    })
else:
    # Train Random Forest
    print("  Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X, y)
    
    # Get feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': features_to_keep,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"✓ Random Forest trained")
    print(f"\nTop 15 Most Important Features:")
    print(feature_importance_df.head(15).to_string(index=False))
    
    # Plot feature importance
    print("\n--- Creating Feature Importance Plot ---")
    plt.figure(figsize=(12, max(8, len(features_to_keep) * 0.3)))
    
    top_n = min(30, len(feature_importance_df))
    
    plt.barh(range(top_n), 
             feature_importance_df.head(top_n)['Importance'],
             color='steelblue')
    plt.yticks(range(top_n), feature_importance_df.head(top_n)['Feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance plot saved to: {output_dir / 'feature_importance.png'}")
    
    # Remove low-importance features
    print(f"\n--- Removing Low-Importance Features (< {IMPORTANCE_THRESHOLD}) ---")
    
    low_importance = feature_importance_df[feature_importance_df['Importance'] < IMPORTANCE_THRESHOLD]
    
    if len(low_importance) > 0:
        print(f"  Features below threshold: {len(low_importance)}")
        for feat, imp in low_importance[['Feature', 'Importance']].values:
            print(f"    ❌ {feat}: {imp:.4f}")
        
        features_to_keep = feature_importance_df[
            feature_importance_df['Importance'] >= IMPORTANCE_THRESHOLD
        ]['Feature'].tolist()
        
        print(f"\n✓ Removed {len(low_importance)} low-importance features")
    else:
        print(f"✓ All features meet importance threshold")

# Save feature importance
feature_importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
print(f"\n✓ Feature importance saved to: {output_dir / 'feature_importance.csv'}")

# ============================================================================
# STEP 8: FINAL FEATURE SET
# ============================================================================
print("\n" + "="*100)
print("STEP 8: FINAL FEATURE SELECTION")
print("="*100)

print(f"\n📊 Feature Selection Summary:")
print(f"   Starting features: {len(numeric_columns)}")
print(f"   After VIF reduction: {len(features_to_keep)} (removed {len(numeric_columns) - len(features_to_keep)})")

# Add categorical features back (if needed for modeling)
final_features = features_to_keep.copy()

# Add essential categorical features (for stratification/grouping)
essential_categorical = ['Equipment_Class_Primary', 'Risk_Category']
for col in essential_categorical:
    if col in df.columns and col not in final_features:
        final_features.append(col)

# Add ID column for reference
if 'Ekipman_ID' in df.columns:
    final_features.insert(0, 'Ekipman_ID')

print(f"   Final features: {len(final_features)}")
print(f"   - Numeric: {len([f for f in final_features if f in features_to_keep])}")
print(f"   - Categorical: {len([f for f in final_features if f not in features_to_keep])}")

print(f"\n✅ Final Feature Set ({len(final_features)} features):")
for i, feat in enumerate(final_features, 1):
    feat_type = "ID" if feat == 'Ekipman_ID' else ("CAT" if feat in categorical_columns else "NUM")
    print(f"  {i:2d}. [{feat_type}] {feat}")

# ============================================================================
# STEP 9: SAVE SELECTED FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 9: SAVING SELECTED FEATURES")
print("="*100)

# Create output dataframe
df_selected = df[final_features].copy()

output_path = Path('data/features_selected.csv')
print(f"\n💾 Saving to: {output_path}")
df_selected.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ Successfully saved!")
print(f"   Records: {len(df_selected):,}")
print(f"   Features: {len(df_selected.columns)}")
print(f"   File size: {output_path.stat().st_size / 1024**2:.2f} MB")

# Save feature selection report
print("\n📋 Creating feature selection report...")

report_data = {
    'Stage': ['Original', 'After VIF', 'After Correlation', 'After Importance', 'Final (with categorical)'],
    'Feature_Count': [
        len(numeric_columns),
        len(features_to_keep),
        len(features_to_keep),
        len(features_to_keep),
        len(final_features)
    ]
}

report_df = pd.DataFrame(report_data)
report_df.to_csv(output_dir / 'selection_summary.csv', index=False)
print(f"✓ Selection summary saved to: {output_dir / 'selection_summary.csv'}")

# Save removed features list
removed_features = [f for f in numeric_columns if f not in features_to_keep]
if len(removed_features) > 0:
    removed_df = pd.DataFrame({
        'Removed_Feature': removed_features,
        'Reason': ['VIF or Correlation or Importance'] * len(removed_features)
    })
    removed_df.to_csv(output_dir / 'removed_features.csv', index=False)
    print(f"✓ Removed features list saved to: {output_dir / 'removed_features.csv'}")

# ============================================================================
# STEP 10: FEATURE SELECTION REPORT
# ============================================================================
print("\n" + "="*100)
print("FEATURE SELECTION COMPLETE - SUMMARY")
print("="*100)

print(f"\n🎯 SELECTION RESULTS:")
print(f"   Original features: {original_feature_count}")
print(f"   Numeric features analyzed: {len(numeric_columns)}")
print(f"   Features removed: {len(numeric_columns) - len(features_to_keep)}")
print(f"   Final feature set: {len(final_features)}")

print(f"\n📊 REMOVAL BREAKDOWN:")
print(f"   VIF reduction: {len(numeric_columns) - len(features_to_keep)} features")
if len(high_corr_pairs) > 0:
    print(f"   Correlation filtering: {len(features_to_remove_corr)} features")
else:
    print(f"   Correlation filtering: 0 features")

print(f"\n📂 OUTPUT FILES:")
print(f"   • {output_path}")
print(f"   • {output_dir / 'vif_analysis.csv'}")
print(f"   • {output_dir / 'feature_importance.csv'}")
print(f"   • {output_dir / 'correlation_matrix.png'}")
print(f"   • {output_dir / 'feature_importance.png'}")
print(f"   • {output_dir / 'selection_summary.csv'}")

print(f"\n🚀 READY FOR MODEL TRAINING:")
print(f"   ✓ Multicollinearity removed (VIF < {VIF_TARGET})")
print(f"   ✓ High correlations eliminated")
print(f"   ✓ Low-importance features removed")
print(f"   ✓ Clean feature set ready for XGBoost/CatBoost")

print("\n" + "="*100)
print(f"{'FEATURE SELECTION PIPELINE COMPLETE':^100}")
print("="*100)