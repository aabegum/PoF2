"""
DATA TRANSFORMATION: FAULT-LEVEL → EQUIPMENT-LEVEL v2.0
Turkish EDAŞ PoF Prediction Project

This script transforms fault records into equipment-level data ready for PoF modeling.

Key Features:
✓ Smart Equipment ID (cbs_id → Ekipman ID → HEPSI_ID → Ekipman Kodu)
✓ Unified Equipment Classification (Equipment_Type → Ekipman Sınıfı → fallbacks)
✓ Age source tracking (TESIS_TARIHI vs EDBS_IDATE)
✓ Handles invalid dates (1900-01-01, 00:00:00, nulls)
✓ Failure history aggregation (3/6/12 months)
✓ MTBF calculation
✓ Recurring fault detection (30/90 days)
✓ Customer impact columns (all MV/LV categories)
✓ Optional specifications (voltage_level, kVa_rating) - future-proof

Priority Logic (aligned with 01_data_profiling.py):
- Equipment ID: cbs_id → Ekipman ID → HEPSI_ID → Ekipman Kodu
- Equipment Class: Equipment_Type → Ekipman Sınıfı → Kesinti Ekipman Sınıfı
- Installation Date: TESIS_TARIHI → EDBS_IDATE

Input:  data/combined_data.xlsx (fault records)
Output: data/equipment_level_data.csv (equipment records with ~30+ features)
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

# PRIMARY STRATEGY: cbs_id → Ekipman ID → HEPSI_ID → Ekipman Kodu
print("\n--- Smart Equipment ID Selection ---")

# Create unified equipment ID with fallback logic
def get_equipment_id(row):
    """
    Get equipment ID with smart fallback
    Priority: cbs_id → Ekipman ID → HEPSI_ID → Ekipman Kodu
    """
    if pd.notna(row.get('cbs_id')):
        return row['cbs_id']
    elif pd.notna(row.get('Ekipman ID')):
        return row['Ekipman ID']
    elif pd.notna(row.get('HEPSI_ID')):
        return row['HEPSI_ID']
    elif pd.notna(row.get('Ekipman Kodu')):
        return row['Ekipman Kodu']
    return None

df['Equipment_ID_Primary'] = df.apply(get_equipment_id, axis=1)

# Statistics
primary_coverage = df['Equipment_ID_Primary'].notna().sum()
unique_equipment = df['Equipment_ID_Primary'].nunique()

print(f"✓ Primary Equipment ID Strategy:")
print(f"  Priority 1: cbs_id")
print(f"  Priority 2: Ekipman ID")
print(f"  Priority 3: HEPSI_ID")
print(f"  Priority 4: Ekipman Kodu")
print(f"  Combined coverage: {primary_coverage:,} ({primary_coverage/len(df)*100:.1f}%)")
print(f"  Unique equipment: {unique_equipment:,}")
print(f"  Average faults per equipment: {len(df)/unique_equipment:.1f}")

# Use this as grouping key
equipment_id_col = 'Equipment_ID_Primary'

# ============================================================================
# STEP 5b: CREATE UNIFIED EQUIPMENT CLASSIFICATION
# ============================================================================
print("\n--- Smart Equipment Classification Selection ---")

# Create unified equipment class with fallback logic
def get_equipment_class(row):
    """
    Get equipment class with smart fallback
    Priority: Equipment_Type → Ekipman Sınıfı → Kesinti Ekipman Sınıfı → Ekipman Sınıf
    """
    if pd.notna(row.get('Equipment_Type')):
        return row['Equipment_Type']
    elif pd.notna(row.get('Ekipman Sınıfı')):
        return row['Ekipman Sınıfı']
    elif pd.notna(row.get('Kesinti Ekipman Sınıfı')):
        return row['Kesinti Ekipman Sınıfı']
    elif pd.notna(row.get('Ekipman Sınıf')):
        return row['Ekipman Sınıf']
    return None

df['Equipment_Class_Primary'] = df.apply(get_equipment_class, axis=1)

class_coverage = df['Equipment_Class_Primary'].notna().sum()
print(f"✓ Unified Equipment Class created:")
print(f"  Priority: Equipment_Type → Ekipman Sınıfı → Kesinti Ekipman Sınıfı")
print(f"  Coverage: {class_coverage:,} ({class_coverage/len(df)*100:.1f}%)")
print(f"  Unique types (before harmonization): {df['Equipment_Class_Primary'].nunique()}")

# HARMONIZE EQUIPMENT CLASSES (fix synonyms and case sensitivity)
print("\n--- Equipment Class Harmonization ---")
equipment_class_mapping = {
    # Low Voltage Lines
    'aghat': 'AG Hat',
    'AG Hat': 'AG Hat',

    # Reclosers (case sensitivity)
    'REKORTMAN': 'Rekortman',
    'Rekortman': 'Rekortman',

    # Low Voltage Poles
    'agdirek': 'AG Direk',
    'AG Direk': 'AG Direk',

    # Transformers (consolidate variants)
    'OGAGTRF': 'OG/AG Trafo',
    'OG/AG Trafo': 'OG/AG Trafo',
    'Trafo Bina Tip': 'OG/AG Trafo',

    # Distribution Boxes/Panels
    'SDK': 'AG Pano Box',
    'AG Pano': 'AG Pano Box',

    # Disconnectors (standardize)
    'Ayırıcı': 'Ayırıcı',

    # Switches (standardize)
    'anahtar': 'AG Anahtar',
    'AG Anahtar': 'AG Anahtar',

    # Circuit Breakers (case sensitivity)
    'KESİCİ': 'Kesici',
    'Kesici': 'Kesici',

    # Medium Voltage Lines
    'OGHAT': 'OG Hat',

    # Panels
    'PANO': 'Pano',

    # Buildings
    'Bina': 'Bina',

    # Lighting
    'Armatür': 'Armatür',

    # High Voltage Pole
    'ENHDirek': 'ENH Direk',
}

# Apply mapping
df['Equipment_Class_Primary'] = df['Equipment_Class_Primary'].map(
    lambda x: equipment_class_mapping.get(x, x) if pd.notna(x) else x
)

harmonized_classes = df['Equipment_Class_Primary'].nunique()
print(f"✓ Equipment classes harmonized:")
print(f"  Before: 20 types → After: {harmonized_classes} types")
print(f"\n  Consolidated mappings:")
print(f"    • aghat (92) + AG Hat (13) → AG Hat (105)")
print(f"    • REKORTMAN (70) + Rekortman (6) → Rekortman (76)")
print(f"    • agdirek (5) + AG Direk (2) → AG Direk (7)")
print(f"    • OGAGTRF (12) + OG/AG Trafo (2) + Trafo Bina Tip (3) → OG/AG Trafo (17)")
print(f"    • SDK (12) + AG Pano (2) → AG Pano Box (14)")
print(f"    • anahtar (300) + AG Anahtar (19) → AG Anahtar (319)")

# Track age source
def get_age_source(row):
    """Track which column provided installation date"""
    if pd.notna(row.get('TESIS_TARIHI_clean')):
        return 'TESIS_TARIHI'
    elif pd.notna(row.get('EDBS_IDATE_clean')):
        return 'EDBS_IDATE'
    return 'MISSING'

df['Age_Source'] = df.apply(get_age_source, axis=1)
# STEP 6: AGGREGATE TO EQUIPMENT LEVEL
print("\nSTEP 6: Aggregating to equipment level...")

# Build aggregation dictionary dynamically based on available columns
agg_dict = {
    # Equipment identification & classification
    'Equipment_Class_Primary': 'first',  # NEW: Unified classification
    'Ekipman Sınıfı': 'first',          # Keep for reference
    'Equipment_Type': 'first',           # Keep for reference
    'Kesinti Ekipman Sınıfı': 'first',  # Keep for reference

    # Geographic data
    'KOORDINAT_X': 'first',
    'KOORDINAT_Y': 'first',
    'İl': 'first',
    'İlçe': 'first',
    'Mahalle': 'first',

    # Age data
    'Installation_Date': 'first',
    'Installation_Year': 'first',
    'Ekipman_Yaşı_Yıl': 'first',
    'Age_Source': 'first',  # NEW: Track which date column used

    # Fault history
    'started at': ['count', 'min', 'max'],
    'Fault_Last_3M': 'sum',
    'Fault_Last_6M': 'sum',
    'Fault_Last_12M': 'sum',

    # Temporal features
    'Summer_Peak_Flag': 'sum',
    'Winter_Peak_Flag': 'sum',
    'Time_To_Repair_Hours': ['mean', 'max']
}

# Add customer impact columns if available
customer_impact_cols = [
    'urban mv+suburban mv',
    'urban lv+suburban lv',
    'urban mv',
    'urban lv',
    'suburban mv',
    'suburban lv',
    'rural mv',
    'rural lv',
    'total customer count'
]

print("  Checking for customer impact columns...")
for col in customer_impact_cols:
    if col in df.columns:
        agg_dict[col] = ['mean', 'max']
        print(f"  ✓ Found: {col}")

# Add optional specification columns if available
optional_spec_cols = {
    'voltage_level': 'first',
    'kVa_rating': 'first',
    'component voltage': 'first',
    'MARKA': 'first',
    'MARKA_MODEL': 'first',
    'FIRMA': 'first'
}

print("  Checking for optional specification columns...")
for col, agg_func in optional_spec_cols.items():
    if col in df.columns:
        agg_dict[col] = agg_func
        print(f"  ✓ Found: {col}")

equipment_df = df.groupby(equipment_id_col).agg(agg_dict).reset_index()
equipment_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in equipment_df.columns.values]

print(f"✓ Created {len(equipment_df):,} equipment records from {original_fault_count:,} faults")

# STEP 7: RENAME COLUMNS
print("\nSTEP 7: Creating final features...")

# Base rename dictionary
rename_dict = {
    'Equipment_ID_Primary': 'Ekipman_ID',
    'Equipment_Class_Primary_first': 'Equipment_Class_Primary',  # NEW: Unified classification
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
    'Age_Source_first': 'Age_Source',  # NEW: Track date source
    'started at_count': 'Toplam_Arıza_Sayisi_Lifetime',
    'started at_min': 'İlk_Arıza_Tarihi',
    'started at_max': 'Son_Arıza_Tarihi',
    'Fault_Last_3M_sum': 'Arıza_Sayısı_3ay',
    'Fault_Last_6M_sum': 'Arıza_Sayısı_6ay',
    'Fault_Last_12M_sum': 'Arıza_Sayısı_12ay',
}

# Add customer impact columns dynamically
for col in customer_impact_cols:
    if f'{col}_mean' in equipment_df.columns:
        rename_dict[f'{col}_mean'] = f'{col.replace(" ", "_")}_Avg'
    if f'{col}_max' in equipment_df.columns:
        rename_dict[f'{col}_max'] = f'{col.replace(" ", "_")}_Max'

# Add optional specification columns dynamically
for col in optional_spec_cols.keys():
    if f'{col}_first' in equipment_df.columns:
        rename_dict[f'{col}_first'] = col

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

print(f"\n📊 TRANSFORMATION SUMMARY:")
print(f"   • Input: {original_fault_count:,} fault records")
print(f"   • Output: {len(equipment_df):,} equipment records")
print(f"   • Reduction: {original_fault_count/len(equipment_df):.1f}x (faults per equipment)")
print(f"   • Total Features: {len(equipment_df.columns)} columns")

print(f"\n🎯 KEY FEATURES CREATED:")
print(f"   • Equipment ID Strategy: cbs_id → Ekipman ID → HEPSI_ID → Ekipman Kodu")
print(f"   • Equipment Classification: Equipment_Class_Primary (unified)")
print(f"   • Age Source Tracking: {equipment_df['Age_Source'].value_counts().to_dict() if 'Age_Source' in equipment_df.columns else 'N/A'}")
print(f"   • Failure History: 3M, 6M, 12M fault counts")
print(f"   • MTBF: {equipment_df['MTBF_Gün'].notna().sum():,} equipment with valid MTBF")
print(f"   • Recurring Faults: {equipment_df['Tekrarlayan_Arıza_90gün_Flag'].sum():,} equipment flagged")

# Customer impact summary
customer_cols_found = [col for col in customer_impact_cols if any(col.replace(" ", "_") in c for c in equipment_df.columns)]
if customer_cols_found:
    print(f"\n👥 CUSTOMER IMPACT COLUMNS:")
    for col in customer_cols_found:
        print(f"   ✓ {col}")
else:
    print(f"\n👥 CUSTOMER IMPACT COLUMNS: Using 'total customer count' only")

# Optional specifications summary
optional_cols_found = [col for col in optional_spec_cols.keys() if col in equipment_df.columns]
if optional_cols_found:
    print(f"\n🌟 OPTIONAL SPECIFICATIONS INCLUDED:")
    for col in optional_cols_found:
        coverage = equipment_df[col].notna().sum()
        pct = coverage / len(equipment_df) * 100
        print(f"   ✓ {col}: {coverage:,} ({pct:.1f}% coverage)")
else:
    print(f"\n💡 OPTIONAL SPECIFICATIONS: None found")
    print(f"   Can add later: voltage_level, kVa_rating, component voltage")

print(f"\n✅ FILES SAVED:")
print(f"   • data/equipment_level_data.csv ({len(equipment_df):,} records)")
print(f"   • data/feature_documentation.csv ({len(equipment_df.columns)} features)")

print(f"\n🚀 READY FOR NEXT PHASE:")
print(f"   → Run: 03_feature_engineering.py")
print(f"   → Create advanced features (age ratios, reliability scores, etc.)")
print("="*100)