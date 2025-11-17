"""
MINIMAL FEATURE TEST
Train with only 5 basic features to isolate leakage vs overfitting

Features to use:
1. Son_Arıza_Gun_Sayisi (days since last failure)
2. MTBF_Gün (mean time between failures)
3. Ilk_Arizaya_Kadar_Yil (years to first failure)
4. Ekipman_Yaşı_Yıl_EDBS_first (equipment age)
5. Equipment_Class_Primary (equipment type)

Expected AUC with clean data: 0.70-0.80
If AUC is still > 0.95: Dataset issue or target leakage in creation
If AUC drops to 0.70-0.80: Other features have leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print(" "*30 + "MINIMAL FEATURE TEST")
print(" "*25 + "Train with 5 Basic Features Only")
print("="*100)

# Load reduced features
df = pd.read_csv('data/features_reduced.csv')
print(f"\n✓ Loaded: {df.shape[0]} equipment × {df.shape[1]} features")

# Select ONLY minimal features
MINIMAL_FEATURES = [
    'Ekipman_ID',
    'Son_Arıza_Gun_Sayisi',
    'MTBF_Gün',
    'Ilk_Arizaya_Kadar_Yil',
    'Ekipman_Yaşı_Yıl_EDBS_first',
    'Equipment_Class_Primary'
]

df_minimal = df[MINIMAL_FEATURES].copy()
print(f"\n✓ Selected {len(MINIMAL_FEATURES)-1} minimal features:")
for i, feat in enumerate(MINIMAL_FEATURES[1:], 1):
    print(f"   {i}. {feat}")

# Create temporal targets
CUTOFF_DATE = pd.Timestamp('2024-06-25')
FUTURE_6M_END = CUTOFF_DATE + pd.DateOffset(months=6)
FUTURE_12M_END = CUTOFF_DATE + pd.DateOffset(months=12)

all_faults = pd.read_excel('data/combined_data.xlsx')
all_faults['started at'] = pd.to_datetime(all_faults['started at'], dayfirst=True, errors='coerce')

future_faults_6M = all_faults[
    (all_faults['started at'] > CUTOFF_DATE) &
    (all_faults['started at'] <= FUTURE_6M_END)
]['cbs_id'].dropna().unique()

future_faults_12M = all_faults[
    (all_faults['started at'] > CUTOFF_DATE) &
    (all_faults['started at'] <= FUTURE_12M_END)
]['cbs_id'].dropna().unique()

df_minimal['Target_6M'] = df_minimal['Ekipman_ID'].isin(future_faults_6M).astype(int)
df_minimal['Target_12M'] = df_minimal['Ekipman_ID'].isin(future_faults_12M).astype(int)

print(f"\n✓ Targets created:")
print(f"   6M: {df_minimal['Target_6M'].sum()} equipment ({df_minimal['Target_6M'].mean()*100:.1f}%)")
print(f"   12M: {df_minimal['Target_12M'].sum()} equipment ({df_minimal['Target_12M'].mean()*100:.1f}%)")

# Prepare features
X = df_minimal.drop(['Ekipman_ID', 'Target_6M', 'Target_12M'], axis=1).copy()

# One-hot encode Equipment_Class_Primary
X = pd.get_dummies(X, columns=['Equipment_Class_Primary'], drop_first=True)

print(f"\n✓ Final feature matrix: {X.shape[1]} features (after encoding)")

# Train/test split
X_train, X_test, y_train_6M, y_test_6M, y_train_12M, y_test_12M = train_test_split(
    X,
    df_minimal['Target_6M'],
    df_minimal['Target_12M'],
    test_size=0.3,
    random_state=42,
    stratify=df_minimal['Target_6M']
)

print(f"\n✓ Train/test split: {len(X_train)} / {len(X_test)}")

# Simple XGBoost (no extensive GridSearch)
print("\n" + "="*100)
print("Training XGBoost with Minimal Features")
print("="*100)

for target_name, y_train, y_test in [('6M', y_train_6M, y_test_6M), ('12M', y_train_12M, y_test_12M)]:
    print(f"\n--- {target_name} Horizon ---")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,  # Shallow to prevent overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n✅ Results:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Average Precision: {ap:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    if auc > 0.90:
        print(f"\n   ⚠️  WARNING: Still very high AUC ({auc:.4f})!")
        print(f"   This suggests:")
        print(f"   1. Target creation may have leakage")
        print(f"   2. Dataset is highly structured/separable")
        print(f"   3. These basic features are genuinely very predictive")
    elif auc > 0.75:
        print(f"\n   ✅ GOOD: Realistic AUC ({auc:.4f})")
        print(f"   This suggests other features in full set have leakage")
    else:
        print(f"\n   ℹ️  Low AUC ({auc:.4f}) - features may not be predictive enough")

print("\n" + "="*100)
print(" "*35 + "MINIMAL FEATURE TEST COMPLETE")
print("="*100)

print("\n🔍 INTERPRETATION:")
print("   If AUC > 0.95 with minimal features:")
print("      → Problem is in target creation or dataset structure")
print("      → Check if targets are calculated correctly")
print("")
print("   If AUC 0.70-0.85 with minimal features:")
print("      → The removed features (location/customer) have leakage")
print("      → Need to review how those aggregated features are calculated")
print("")
print("   If AUC < 0.65 with minimal features:")
print("      → Basic features not predictive enough")
print("      → May need more sophisticated feature engineering")
