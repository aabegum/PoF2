"""
DATA TRANSFORMATION: FAULT-LEVEL → EQUIPMENT-LEVEL
Turkish EDAŞ PoF Prediction Project

This script transforms your fault records into equipment-level data ready for PoF modeling.

Key Features:
✓ Handles invalid dates (1900-01-01, 00:00:00, nulls)
✓ Smart age calculation (TESIS_TARIHI → EDBS_IDATE fallback)
✓ Failure history aggregation (3/6/12 months)
✓ MTBF calculation
✓ Recurring fault detection (30/90 days)
✓ Temporal feature engineering

Input:  data/combined_data.xlsx (1,629 fault records)
Output: data/equipment_level_data.csv (~1,300 equipment records)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# Constants
CURRENT_YEAR = 2025
MIN_VALID_YEAR = 1950
MAX_VALID_YEAR = 2025
REFERENCE_DATE = pd.Timestamp('2025-06-25')

print("="*100)
print(" "*30 + "DATA TRANSFORMATION PIPELINE")
print("="*100)

# STEP 1: LOAD DATA
print("\nSTEP 1: Loading fault-level data...")
df = pd.read_excel('data/combined_data.xlsx')
print(f"✓ Loaded: {df.shape[0]:,} faults × {df.shape[1]} columns")
original_fault_count = len(df)

# STEP 2: CLEAN INSTALLATION DATES
print("\nSTEP 2: Cleaning installation dates...")

def clean_date(date_val):
    if pd.isna(date_val):
        return None
    try:
        year = date_val.year
        if year < MIN_VALID_YEAR or year > MAX_VALID_YEAR:
            return None
        return date_val
    except:
        return None

df['TESIS_TARIHI'] = pd.to_datetime(df['TESIS_TARIHI'], errors='coerce')
df['EDBS_IDATE'] = pd.to_datetime(df['EDBS_IDATE'], errors='coerce')

df['TESIS_TARIHI_clean'] = df['TESIS_TARIHI'].apply(clean_date)
df['EDBS_IDATE_clean'] = df['EDBS_IDATE'].apply(clean_date)

df['Installation_Date'] = df['TESIS_TARIHI_clean'].fillna(df['EDBS_IDATE_clean'])

print(f"✓ Combined coverage: {df['Installation_Date'].notna().sum()} ({df['Installation_Date'].notna().sum()/len(df)*100:.1f}%)")

# STEP 3: CALCULATE AGE
print("\nSTEP 3: Calculating equipment age...")
df['Installation_Year'] = df['Installation_Date'].dt.year
df['Ekipman_Yaşı_Yıl'] = CURRENT_YEAR - df['Installation_Year']
print(f"✓ Age calculated for {df['Ekipman_Yaşı_Yıl'].notna().sum():,} records")

# STEP 4: PROCESS TIMESTAMPS
print("\nSTEP 4: Processing fault timestamps...")
df['started at'] = pd.to_datetime(df['started at'], errors='coerce')
df['ended at'] = pd.to_datetime(df['ended at'], errors='coerce')

df['Fault_Month'] = df['started at'].dt.month
df['Summer_Peak_Flag'] = df['Fault_Month'].isin([6, 7, 8, 9]).astype(int)
df['Winter_Peak_Flag'] = df['Fault_Month'].isin([12, 1, 2]).astype(int)
df['Time_To_Repair_Hours'] = (df['ended at'] - df['started at']).dt.total_seconds() / 3600

print("✓ Temporal features created")

# STEP 5: CALCULATE FAILURE PERIODS
print("\nSTEP 5: Calculating failure counts...")
reference_date = df['started at'].max()
cutoff_3m = reference_date - pd.Timedelta(days=90)
cutoff_6m = reference_date - pd.Timedelta(days=180)
cutoff_12m = reference_date - pd.Timedelta(days=365)

df['Fault_Last_3M'] = (df['started at'] >= cutoff_3m).astype(int)
df['Fault_Last_6M'] = (df['started at'] >= cutoff_6m).astype(int)
df['Fault_Last_12M'] = (df['started at'] >= cutoff_12m).astype(int)

print("✓ Failure period flags created")
# ============================================================================
# STEP 5: IDENTIFY PRIMARY EQUIPMENT ID
# ============================================================================
print("\n" + "="*100)
print("STEP 5: EQUIPMENT IDENTIFICATION")
print("="*100)

# PRIMARY STRATEGY: cbs_id → HEPSI_ID (domain expert recommendation)
print("\n--- Smart Equipment ID Selection ---")

# Create unified equipment ID with fallback logic
def get_equipment_id(row):
    """
    Get equipment ID with smart fallback
    Priority: cbs_id → HEPSI_ID
    """
    if pd.notna(row.get('cbs_id')):
        return row['cbs_id']
    elif pd.notna(row.get('HEPSI_ID')):
        return row['HEPSI_ID']
    return None

df['Equipment_ID_Primary'] = df.apply(get_equipment_id, axis=1)

# Statistics
primary_coverage = df['Equipment_ID_Primary'].notna().sum()
unique_equipment = df['Equipment_ID_Primary'].nunique()

print(f"✓ Primary Equipment ID Strategy:")
print(f"  Priority 1: cbs_id")
print(f"  Priority 2: HEPSI_ID")
print(f"  Combined coverage: {primary_coverage:,} ({primary_coverage/len(df)*100:.1f}%)")
print(f"  Unique equipment: {unique_equipment:,}")
print(f"  Average faults per equipment: {len(df)/unique_equipment:.1f}")

# Use this as grouping key
equipment_id_col = 'Equipment_ID_Primary'
# STEP 6: AGGREGATE TO EQUIPMENT LEVEL
print("\nSTEP 6: Aggregating to equipment level...")

agg_dict = {
    'Ekipman Sınıfı': 'first',
    'Equipment_Type': 'first',
    'Kesinti Ekipman Sınıfı': 'first',
    'KOORDINAT_X': 'first',
    'KOORDINAT_Y': 'first',
    'İl': 'first',
    'İlçe': 'first',
    'Mahalle': 'first',
    'Installation_Date': 'first',
    'Installation_Year': 'first',
    'Ekipman_Yaşı_Yıl': 'first',
    'started at': ['count', 'min', 'max'],
    'Fault_Last_3M': 'sum',
    'Fault_Last_6M': 'sum',
    'Fault_Last_12M': 'sum',
    'total customer count': ['mean', 'max'],
    'Summer_Peak_Flag': 'sum',
    'Winter_Peak_Flag': 'sum',
    'Time_To_Repair_Hours': ['mean', 'max']
}

equipment_df = df.groupby(equipment_id_col).agg(agg_dict).reset_index()
equipment_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in equipment_df.columns.values]

print(f"✓ Created {len(equipment_df):,} equipment records from {original_fault_count:,} faults")

# STEP 7: RENAME COLUMNS
print("\nSTEP 7: Creating final features...")

rename_dict = {
    'Equipment_ID_Primary': 'Ekipman_ID',  # ← NEW: Using smart ID
    'Ekipman Sınıfı_first': 'Ekipman_Sınıfı',
    'Equipment_Type_first': 'Equipment_Type',
    'Kesinti Ekipman Sınıfı_first': 'Kesinti Ekipman Sınıfı',
    'KOORDINAT_X_first': 'KOORDINAT_X',
    'KOORDINAT_Y_first': 'KOORDINAT_Y',
    'İl_first': 'İl',
    'İlçe_first': 'İlçe',
    'Mahalle_first': 'Mahalle',
    'Installation_Date_first': 'Installation_Date',
    'Installation_Year_first': 'Installation_Year',
    'Ekipman_Yaşı_Yıl_first': 'Ekipman_Yaşı_Yıl',
    'started at_count': 'Toplam_Arıza_Sayisi_Lifetime',
    'started at_min': 'İlk_Arıza_Tarihi',
    'started at_max': 'Son_Arıza_Tarihi',
    'Fault_Last_3M_sum': 'Arıza_Sayısı_3ay',
    'Fault_Last_6M_sum': 'Arıza_Sayısı_6ay',
    'Fault_Last_12M_sum': 'Arıza_Sayısı_12ay',
    'total customer count_mean': 'Avg_Customer_Count',
    'total customer count_max': 'Max_Customer_Count',
}

equipment_df.rename(columns=rename_dict, inplace=True)

# Calculate MTBF
def calculate_mtbf(row):
    if pd.notna(row['İlk_Arıza_Tarihi']) and pd.notna(row['Son_Arıza_Tarihi']):
        total_days = (row['Son_Arıza_Tarihi'] - row['İlk_Arıza_Tarihi']).days
        total_faults = row['Toplam_Arıza_Sayisi_Lifetime']
        if total_faults > 1 and total_days > 0:
            return total_days / (total_faults - 1)
    return None

equipment_df['MTBF_Gün'] = equipment_df.apply(calculate_mtbf, axis=1)

# Days since last fault
equipment_df['Son_Arıza_Gun_Sayisi'] = (REFERENCE_DATE - equipment_df['Son_Arıza_Tarihi']).dt.days

# Recurrence flags
print("\nSTEP 8: Detecting recurring faults...")

def calculate_recurrence(equipment_id):
    equip_faults = df[df[equipment_id_col] == equipment_id]['started at'].dropna().sort_values()
    if len(equip_faults) < 2:
        return 0, 0
    time_diffs = equip_faults.diff().dt.days.dropna()
    return int((time_diffs <= 30).any()), int((time_diffs <= 90).any())

recurrence_results = equipment_df['Ekipman_ID'].apply(calculate_recurrence)
equipment_df['Tekrarlayan_Arıza_30gün_Flag'] = [r[0] for r in recurrence_results]
equipment_df['Tekrarlayan_Arıza_90gün_Flag'] = [r[1] for r in recurrence_results]

print(f"✓ Found {equipment_df['Tekrarlayan_Arıza_90gün_Flag'].sum()} equipment with recurring faults (90 days)")

# STEP 9: SAVE
print("\nSTEP 9: Saving results...")
equipment_df.to_csv('data/equipment_level_data.csv', index=False, encoding='utf-8-sig')

# Feature documentation
feature_docs = pd.DataFrame({
    'Feature_Name': equipment_df.columns,
    'Data_Type': equipment_df.dtypes.astype(str),
    'Completeness_%': (equipment_df.notna().sum() / len(equipment_df) * 100).round(1)
})
feature_docs.to_csv('data/feature_documentation.csv', index=False)

print("\n" + "="*100)
print("TRANSFORMATION COMPLETE!")
print("="*100)
print(f"\n✅ Output: {len(equipment_df):,} equipment records")
print(f"✅ Features: {len(equipment_df.columns)} columns")
print(f"✅ Files saved:")
print(f"   • data/equipment_level_data.csv")
print(f"   • data/feature_documentation.csv")
print("\n🚀 Ready for PoF modeling!")