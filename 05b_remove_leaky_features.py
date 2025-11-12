"""
REMOVE LEAKY FEATURES - DATA LEAKAGE FIX
Remove features that encode target information
"""

import pandas as pd
from pathlib import Path

# Load current features
df = pd.read_csv('data/features_selected.csv')
print(f"Original features: {df.shape[1]}")

# Define safe features (no data leakage)
SAFE_FEATURES = [
    'Ekipman_ID',
    # Truly predictive features
    'Ekipman_Yaşı_Yıl_Class_Avg',
    'Equipment_Class_Primary',
    'Risk_Category',
    'MTBF_Gün',
    'Reliability_Score',
    'Time_To_Repair_Hours_mean',
    'Summer_Peak_Flag_sum',
    'Arıza_Sayısı_12ay_Cluster_Avg',
    'Arıza_Sayısı_12ay_Class_Avg',
    'Toplam_Arıza_Sayisi_Lifetime'
]

# Keep only safe features
df_clean = df[SAFE_FEATURES].copy()

print(f"\nFeatures after removing leakage: {df_clean.shape[1]}")
print(f"Removed: {df.shape[1] - df_clean.shape[1]} leaky features")

# Save
df_clean.to_csv('data/features_selected_clean.csv', index=False)
print("\n✓ Saved: data/features_selected_clean.csv")
print("\nRemoved leaky features:")
removed = set(df.columns) - set(SAFE_FEATURES)
for feat in removed:
    print(f"  ❌ {feat}")