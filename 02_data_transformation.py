"""
DATA TRANSFORMATION: FAULT-LEVEL → EQUIPMENT-LEVEL v3.0 (ENHANCED)
Turkish EDAŞ PoF Prediction Project

ENHANCEMENTS in v3.0:
✓ Day-precision age calculation (not just year)
✓ Improved date validation with diagnostics
✓ Optional first work order fallback for missing ages
✓ Vectorized operations for better performance
✓ Complete audit trail (install date, age source, age in days)

Key Features:
✓ Smart Equipment ID (cbs_id → Ekipman ID → HEPSI_ID → Ekipman Kodu)
✓ Unified Equipment Classification (Equipment_Type → Ekipman Sınıfı → fallbacks)
✓ Age source tracking (TESIS_TARIHI vs EDBS_IDATE vs FIRST_WORKORDER_PROXY)
✓ Handles invalid dates (1900-01-01, 00:00:00, nulls)
✓ Failure history aggregation (3/6/12 months)
✓ MTBF calculation
✓ Recurring fault detection (30/90 days)
✓ Customer impact columns (all MV/LV categories)
✓ Optional specifications (voltage_level, kVa_rating) - future-proof

Priority Logic:
- Equipment ID: cbs_id → Ekipman ID → HEPSI_ID → Ekipman Kodu
- Equipment Class: Equipment_Type → Ekipman Sınıfı → Kesinti Ekipman Sınıfı
- Installation Date: TESIS_TARIHI → EDBS_IDATE → First Work Order (optional)

Input:  data/combined_data.xlsx (fault records)
Output: data/equipment_level_data.csv (equipment records with ~30+ features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix Unicode encoding for Windows console (Turkish cp1254 issue)
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

pd.set_option('display.max_columns', None)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Constants
CURRENT_YEAR = 2025
MIN_VALID_YEAR = 1950
MAX_VALID_YEAR = 2025
REFERENCE_DATE = pd.Timestamp('2025-06-25')

# Feature flags
USE_FIRST_WORKORDER_FALLBACK = True  # Set to True to enable Option 3 (first work order as age proxy)

print("="*100)
print(" "*25 + "DATA TRANSFORMATION PIPELINE v3.0 (ENHANCED)")
print("="*100)
print(f"\n⚙️  Configuration:")
print(f"   Reference Date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
print(f"   Valid Year Range: {MIN_VALID_YEAR}-{MAX_VALID_YEAR}")
print(f"   First Work Order Fallback: {'ENABLED' if USE_FIRST_WORKORDER_FALLBACK else 'DISABLED'}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING FAULT-LEVEL DATA")
print("="*100)

df = pd.read_excel('data/combined_data.xlsx')
print(f"\n✓ Loaded: {df.shape[0]:,} faults × {df.shape[1]} columns")
original_fault_count = len(df)

# ============================================================================
# STEP 2: ENHANCED DATE PARSING & VALIDATION
# ============================================================================
print("\n" + "="*100)
print("STEP 2: PARSING AND VALIDATING DATE COLUMNS (ENHANCED)")
print("="*100)

def parse_and_validate_date(date_series, column_name, min_year=MIN_VALID_YEAR, max_year=MAX_VALID_YEAR, report=True):
    """
    Parse and validate dates with detailed diagnostics

    Args:
        date_series: Series of date values
        column_name: Name for reporting
        min_year: Minimum valid year (default: 1950)
        max_year: Maximum valid year (default: 2025)
        report: Whether to print statistics (default: True)

    Returns:
        Series of validated datetime values (invalid → NaT)
    """
    # Check if data is Excel serial date (integer/float format)
    if pd.api.types.is_numeric_dtype(date_series):
        # Excel serial dates: days since 1900-01-01 (Windows Excel)
        # Valid range: ~18263 (1950) to ~45657 (2025)
        # Origin = 1899-12-30 because Excel incorrectly treats 1900 as leap year
        parsed = pd.to_datetime(date_series, unit='D', origin='1899-12-30', errors='coerce')
    else:
        # Parse dates with Turkish date format support (DD/MM/YYYY)
        parsed = pd.to_datetime(date_series, errors='coerce', dayfirst=True)

    # Validation masks
    valid_mask = (
        parsed.notna() &
        (parsed.dt.year >= min_year) &
        (parsed.dt.year <= max_year)
    )

    # Categorize invalid dates
    invalid_old = (parsed.notna() & (parsed.dt.year < min_year)).sum()
    invalid_future = (parsed.notna() & (parsed.dt.year > max_year)).sum()

    # Set invalid to NaT
    parsed[~valid_mask] = pd.NaT

    # Statistics
    if report:
        total = len(date_series)
        valid = valid_mask.sum()

        print(f"\n  {column_name:30s}:")
        print(f"    Valid dates:       {valid:6,}/{total:6,} ({valid/total*100:5.1f}%)")
        if invalid_old > 0:
            print(f"    Invalid (< {min_year}): {invalid_old:6,} ⚠️  (set to NaT)")
        if invalid_future > 0:
            print(f"    Invalid (> {max_year}): {invalid_future:6,} ⚠️  (set to NaT)")

    return parsed

# Parse and validate all date columns
print("\nParsing installation date columns:")
df['TESIS_TARIHI_parsed'] = parse_and_validate_date(df['TESIS_TARIHI'], 'TESIS_TARIHI')
df['EDBS_IDATE_parsed'] = parse_and_validate_date(df['EDBS_IDATE'], 'EDBS_IDATE')

print("\nParsing fault timestamp columns:")
df['started at'] = parse_and_validate_date(df['started at'], 'started at', min_year=2020, report=True)
df['ended at'] = parse_and_validate_date(df['ended at'], 'ended at', min_year=2020, report=True)

# Parse work order creation date (for fallback option)
if 'Oluşturma Tarihi Sıralama' in df.columns or 'Oluşturulma_Tarihi' in df.columns:
    creation_col = 'Oluşturma Tarihi Sıralama' if 'Oluşturma Tarihi Sıralama' in df.columns else 'Oluşturulma_Tarihi'
    df['Oluşturulma_Tarihi'] = parse_and_validate_date(df[creation_col], 'Work Order Creation Date', min_year=2015, report=True)
else:
    df['Oluşturulma_Tarihi'] = pd.NaT
    print("\n  ⚠️  Work order creation date not found (fallback option disabled)")

# ============================================================================
# STEP 3: ENHANCED EQUIPMENT AGE CALCULATION
# ============================================================================
print("\n" + "="*100)
print("STEP 3: CALCULATING EQUIPMENT AGE (DAY PRECISION)")
print("="*100)

def calculate_equipment_age_improved(row):
    """
    Calculate equipment age with day precision

    Priority:
    1. TESIS_TARIHI (primary installation date)
    2. EDBS_IDATE (fallback installation date)
    3. First work order date (optional proxy - equipment may be older)

    Returns:
        tuple: (age_in_days, source_used, install_date)
    """
    ref_date = REFERENCE_DATE

    # Option 1: TESIS_TARIHI (primary)
    if pd.notna(row['TESIS_TARIHI_parsed']):
        install_date = row['TESIS_TARIHI_parsed']
        if install_date < ref_date:
            age_days = (ref_date - install_date).days
            return age_days, 'TESIS_TARIHI', install_date

    # Option 2: EDBS_IDATE (fallback)
    if pd.notna(row['EDBS_IDATE_parsed']):
        install_date = row['EDBS_IDATE_parsed']
        if install_date < ref_date:
            age_days = (ref_date - install_date).days
            return age_days, 'EDBS_IDATE', install_date

    # No valid installation date found
    return None, 'MISSING', None

print("\nCalculating ages from installation dates...")

# Optimized tuple unpacking (vectorized)
results = df.apply(calculate_equipment_age_improved, axis=1, result_type='expand')
results.columns = ['Ekipman_Yaşı_Gün', 'Yaş_Kaynak', 'Ekipman_Kurulum_Tarihi']

# Assign all at once
df[['Ekipman_Yaşı_Gün', 'Yaş_Kaynak', 'Ekipman_Kurulum_Tarihi']] = results
df['Ekipman_Yaşı_Yıl'] = df['Ekipman_Yaşı_Gün'] / 365.25

# Statistics
print("\n✓ Age Calculation Results:")
print(f"\n  Age Source Distribution:")
source_counts = df['Yaş_Kaynak'].value_counts()
for source, count in source_counts.items():
    pct = count / len(df) * 100
    print(f"    {source:25s}: {count:6,} ({pct:5.1f}%)")

# Age statistics (excluding missing)
valid_ages = df[df['Yaş_Kaynak'] != 'MISSING']['Ekipman_Yaşı_Yıl']
if len(valid_ages) > 0:
    print(f"\n  Age Statistics (valid ages only):")
    print(f"    Mean:   {valid_ages.mean():>6.1f} years")
    print(f"    Median: {valid_ages.median():>6.1f} years")
    print(f"    Min:    {valid_ages.min():>6.1f} years")
    print(f"    Max:    {valid_ages.max():>6.1f} years")

    # Age distribution
    age_bins = [0, 5, 10, 20, 30, 50, 75]
    age_labels = ['0-5 yrs', '5-10 yrs', '10-20 yrs', '20-30 yrs', '30-50 yrs', '50-75 yrs']
    age_dist = pd.cut(valid_ages, bins=age_bins, labels=age_labels).value_counts().sort_index()

    print(f"\n  Age Distribution:")
    for label, count in age_dist.items():
        pct = count / len(valid_ages) * 100
        bar = '█' * int(pct / 2)  # Visual bar
        print(f"    {label}: {count:>4,} ({pct:>5.1f}%) {bar}")

    # Warnings
    if (valid_ages > 75).sum() > 0:
        print(f"\n  ⚠️  WARNING: {(valid_ages > 75).sum()} equipment > 75 years (check data quality!)")
    if valid_ages.median() < 1:
        print(f"  ⚠️  WARNING: Median age is {valid_ages.median():.1f} years - investigate if accurate")

# ============================================================================
# STEP 3B: OPTIONAL FIRST WORK ORDER FALLBACK
# ============================================================================
if USE_FIRST_WORKORDER_FALLBACK:
    print("\n" + "="*100)
    print("STEP 3B: FILLING MISSING AGES WITH FIRST WORK ORDER (VECTORIZED)")
    print("="*100)

    missing_mask = df['Yaş_Kaynak'] == 'MISSING'
    missing_count = missing_mask.sum()

    if missing_count > 0 and 'Oluşturulma_Tarihi' in df.columns:
        print(f"\n  Equipment with MISSING age: {missing_count:,} ({missing_count/len(df)*100:.1f}%)")
        print(f"  Attempting to use first work order date as proxy...\n")

        # Identify equipment ID column
        equip_id_cols = ['cbs_id', 'Ekipman Kodu', 'Ekipman ID', 'HEPSI_ID']
        equip_id_col = None
        for col in equip_id_cols:
            if col in df.columns:
                equip_id_col = col
                break

        if equip_id_col:
            print(f"  Using equipment ID column: {equip_id_col}")

            # Vectorized approach: Get first work order per equipment
            first_wo_dates = df.groupby(equip_id_col)['Oluşturulma_Tarihi'].min()

            # Map first work order dates to all rows
            df['_first_wo'] = df[equip_id_col].map(first_wo_dates)

            # Calculate age from first work order (vectorized)
            age_from_wo = (REFERENCE_DATE - df['_first_wo']).dt.days

            # Only fill where: missing AND first_wo is valid AND age is positive
            fill_mask = (
                missing_mask &
                df['_first_wo'].notna() &
                (age_from_wo > 0)
            )

            # Vectorized assignment
            df.loc[fill_mask, 'Ekipman_Yaşı_Gün'] = age_from_wo[fill_mask]
            df.loc[fill_mask, 'Ekipman_Yaşı_Yıl'] = age_from_wo[fill_mask] / 365.25
            df.loc[fill_mask, 'Yaş_Kaynak'] = 'FIRST_WORKORDER_PROXY'
            df.loc[fill_mask, 'Ekipman_Kurulum_Tarihi'] = df.loc[fill_mask, '_first_wo']

            # Cleanup temporary column
            df.drop(columns=['_first_wo'], inplace=True)

            filled_count = fill_mask.sum()
            remaining_missing = (df['Yaş_Kaynak'] == 'MISSING').sum()

            print(f"  ✓ Filled: {filled_count:,} using first work order proxy")
            print(f"  ✓ Remaining MISSING: {remaining_missing:,} ({remaining_missing/len(df)*100:.1f}%)")

            # Final age statistics
            if filled_count > 0:
                print(f"\n  Updated Age Source Distribution:")
                for source, count in df['Yaş_Kaynak'].value_counts().items():
                    pct = count / len(df) * 100
                    print(f"    {source:25s}: {count:6,} ({pct:5.1f}%)")
        else:
            print(f"  ⚠️  Equipment ID column not found - cannot use first work order fallback")
    elif missing_count == 0:
        print(f"\n  ✓ No missing ages - first work order fallback not needed")
    else:
        print(f"\n  ⚠️  Work order creation date not available - cannot use fallback")

# ============================================================================
# STEP 4: PROCESS FAULT TIMESTAMPS
# ============================================================================
print("\n" + "="*100)
print("STEP 4: PROCESSING FAULT TIMESTAMPS")
print("="*100)

df['Fault_Month'] = df['started at'].dt.month
df['Summer_Peak_Flag'] = df['Fault_Month'].isin([6, 7, 8, 9]).astype(int)
df['Winter_Peak_Flag'] = df['Fault_Month'].isin([12, 1, 2]).astype(int)
df['Time_To_Repair_Hours'] = (df['ended at'] - df['started at']).dt.total_seconds() / 3600

print("\n✓ Temporal features created:")
print(f"  Summer peak faults: {df['Summer_Peak_Flag'].sum():,}")
print(f"  Winter peak faults: {df['Winter_Peak_Flag'].sum():,}")
print(f"  Avg repair time: {df['Time_To_Repair_Hours'].mean():.1f} hours")

# ============================================================================
# STEP 5: CALCULATE FAILURE PERIODS
# ============================================================================
print("\n" + "="*100)
print("STEP 5: CALCULATING FAILURE PERIOD FLAGS")
print("="*100)

reference_date = df['started at'].max()
cutoff_3m = reference_date - pd.Timedelta(days=90)
cutoff_6m = reference_date - pd.Timedelta(days=180)
cutoff_12m = reference_date - pd.Timedelta(days=365)

df['Fault_Last_3M'] = (df['started at'] >= cutoff_3m).astype(int)
df['Fault_Last_6M'] = (df['started at'] >= cutoff_6m).astype(int)
df['Fault_Last_12M'] = (df['started at'] >= cutoff_12m).astype(int)

print(f"\n✓ Failure period flags created:")
print(f"  Reference date: {reference_date.strftime('%Y-%m-%d')}")
print(f"  Faults in last 3M:  {df['Fault_Last_3M'].sum():,}")
print(f"  Faults in last 6M:  {df['Fault_Last_6M'].sum():,}")
print(f"  Faults in last 12M: {df['Fault_Last_12M'].sum():,}")

# ============================================================================
# STEP 6: IDENTIFY PRIMARY EQUIPMENT ID
# ============================================================================
print("\n" + "="*100)
print("STEP 6: EQUIPMENT IDENTIFICATION")
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
# STEP 6B: CREATE UNIFIED EQUIPMENT CLASSIFICATION
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
print(f"  Before: {len(equipment_class_mapping)} types → After: {harmonized_classes} types")
print(f"\n  Consolidated mappings:")
print(f"    • aghat + AG Hat → AG Hat")
print(f"    • REKORTMAN + Rekortman → Rekortman")
print(f"    • agdirek + AG Direk → AG Direk")
print(f"    • OGAGTRF + OG/AG Trafo + Trafo Bina Tip → OG/AG Trafo")
print(f"    • SDK + AG Pano → AG Pano Box")
print(f"    • anahtar + AG Anahtar → AG Anahtar")

# Track age source
def get_age_source(row):
    """Track which column provided installation date"""
    return row['Yaş_Kaynak']  # Already set in step 3

df['Age_Source'] = df['Yaş_Kaynak']

# ============================================================================
# STEP 7: AGGREGATE TO EQUIPMENT LEVEL
# ============================================================================
print("\n" + "="*100)
print("STEP 7: AGGREGATING TO EQUIPMENT LEVEL")
print("="*100)

# Sort by Age_Source to prioritize TESIS_TARIHI when aggregating with 'first'
# This ensures that for equipment with multiple faults, we prefer TESIS_TARIHI over EDBS_IDATE
source_priority = {'TESIS_TARIHI': 0, 'EDBS_IDATE': 1, 'FIRST_WORKORDER_PROXY': 2, 'MISSING': 3}
df['_source_priority'] = df['Age_Source'].map(source_priority).fillna(99)
df = df.sort_values('_source_priority')
df = df.drop(columns=['_source_priority'])

print("\n  ✓ Sorted data to prioritize TESIS_TARIHI as age source during aggregation")

# Build aggregation dictionary dynamically based on available columns
agg_dict = {
    # Equipment identification & classification
    'Equipment_Class_Primary': 'first',
    'Ekipman Sınıfı': 'first',
    'Equipment_Type': 'first',
    'Kesinti Ekipman Sınıfı': 'first',

    # Geographic data
    'KOORDINAT_X': 'first',
    'KOORDINAT_Y': 'first',
    'İl': 'first',
    'İlçe': 'first',
    'Mahalle': 'first',

    # Age data (ENHANCED - TESIS_TARIHI prioritized via pre-sort)
    'Ekipman_Kurulum_Tarihi': 'first',
    'Ekipman_Yaşı_Gün': 'first',
    'Ekipman_Yaşı_Yıl': 'first',
    'Age_Source': 'first',

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

# Add cause code column if available
if 'cause code' in df.columns:
    agg_dict['cause code'] = ['first', 'last', lambda x: x.mode()[0] if len(x.mode()) > 0 else None]
    print("\n  ✓ Found: cause code (will aggregate first, last, and most common)")

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

print(f"\n✓ Aggregating {len(df):,} fault records to equipment level...")
equipment_df = df.groupby(equipment_id_col).agg(agg_dict).reset_index()
equipment_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in equipment_df.columns.values]

print(f"✓ Created {len(equipment_df):,} equipment records from {original_fault_count:,} faults")

# ============================================================================
# STEP 8: RENAME COLUMNS
# ============================================================================
print("\n" + "="*100)
print("STEP 8: CREATING FINAL FEATURES")
print("="*100)

# Base rename dictionary (ENHANCED - includes new age columns)
rename_dict = {
    'Equipment_ID_Primary': 'Ekipman_ID',
    'Equipment_Class_Primary_first': 'Equipment_Class_Primary',
    'Ekipman Sınıfı_first': 'Ekipman_Sınıfı',
    'Equipment_Type_first': 'Equipment_Type',
    'Kesinti Ekipman Sınıfı_first': 'Kesinti Ekipman Sınıfı',
    'KOORDINAT_X_first': 'KOORDINAT_X',
    'KOORDINAT_Y_first': 'KOORDINAT_Y',
    'İl_first': 'İl',
    'İlçe_first': 'İlçe',
    'Mahalle_first': 'Mahalle',
    'Ekipman_Kurulum_Tarihi_first': 'Ekipman_Kurulum_Tarihi',  # NEW
    'Ekipman_Yaşı_Gün_first': 'Ekipman_Yaşı_Gün',  # NEW
    'Ekipman_Yaşı_Yıl_first': 'Ekipman_Yaşı_Yıl',
    'Age_Source_first': 'Age_Source',
    'started at_count': 'Toplam_Arıza_Sayisi_Lifetime',
    'started at_min': 'İlk_Arıza_Tarihi',
    'started at_max': 'Son_Arıza_Tarihi',
    'Fault_Last_3M_sum': 'Arıza_Sayısı_3ay',
    'Fault_Last_6M_sum': 'Arıza_Sayısı_6ay',
    'Fault_Last_12M_sum': 'Arıza_Sayısı_12ay',
}

# Add cause code columns if available
if 'cause code_first' in equipment_df.columns:
    rename_dict['cause code_first'] = 'Arıza_Nedeni_İlk'
    rename_dict['cause code_last'] = 'Arıza_Nedeni_Son'
    rename_dict['cause code_<lambda>'] = 'Arıza_Nedeni_Sık'

# Add customer impact columns dynamically
for col in customer_impact_cols:
    if f'{col}_mean' in equipment_df.columns:
        rename_dict[f'{col}_mean'] = f'{col.replace(" ", "_")}_Avg'
    if f'{col}_max' in equipment_df.columns:
        rename_dict[f'{col}_max'] = f'{col.replace(" ", "_")}_Max'

# Add optional specification columns dynamically
for col in optional_spec_cols.keys():
    if f'{col}_first' in equipment_df.columns:
        clean_col_name = col.replace(' ', '_')
        rename_dict[f'{col}_first'] = clean_col_name

equipment_df.rename(columns=rename_dict, inplace=True)

# ============================================================================
# STEP 9: CALCULATE CAUSE CODE FEATURES
# ============================================================================
has_cause_code = any(col for col in equipment_df.columns if 'cause code' in col.lower() or 'arıza_nedeni' in col.lower())

if has_cause_code and 'cause code' in df.columns:
    print("\nCalculating cause code features...")

    # Create cause code distribution per equipment
    cause_distribution = df.groupby([equipment_id_col, 'cause code']).size().unstack(fill_value=0)

    # Cause diversity: How many different cause types per equipment
    equipment_df['Arıza_Nedeni_Çeşitlilik'] = (cause_distribution > 0).sum(axis=1).reindex(equipment_df['Ekipman_ID']).fillna(0).values

    # Cause consistency: Percentage of faults with most common cause
    total_faults_per_equip = cause_distribution.sum(axis=1)
    max_cause_per_equip = cause_distribution.max(axis=1)
    cause_consistency = (max_cause_per_equip / total_faults_per_equip).reindex(equipment_df['Ekipman_ID']).fillna(0).values
    equipment_df['Arıza_Nedeni_Tutarlılık'] = cause_consistency

    print(f"  ✓ Created Arıza_Nedeni_Çeşitlilik (cause diversity)")
    print(f"  ✓ Created Arıza_Nedeni_Tutarlılık (cause consistency)")
    print(f"  ✓ Avg cause types per equipment: {equipment_df['Arıza_Nedeni_Çeşitlilik'].mean():.2f}")
    print(f"  ✓ Avg cause consistency: {equipment_df['Arıza_Nedeni_Tutarlılık'].mean():.2%}")
else:
    print("\n⚠ Cause code column not found in fault data - skipping cause diversity/consistency features")

# ============================================================================
# STEP 10: CALCULATE MTBF
# ============================================================================
print("\nCalculating MTBF (Mean Time Between Failures)...")

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

print(f"  ✓ MTBF calculable for {equipment_df['MTBF_Gün'].notna().sum():,} equipment")

# ============================================================================
# STEP 11: DETECT RECURRING FAULTS
# ============================================================================
print("\n" + "="*100)
print("STEP 11: DETECTING RECURRING FAULTS")
print("="*100)

def calculate_recurrence(equipment_id):
    equip_faults = df[df[equipment_id_col] == equipment_id]['started at'].dropna().sort_values()
    if len(equip_faults) < 2:
        return 0, 0
    time_diffs = equip_faults.diff().dt.days.dropna()
    return int((time_diffs <= 30).any()), int((time_diffs <= 90).any())

print("\nAnalyzing recurring fault patterns...")
recurrence_results = equipment_df['Ekipman_ID'].apply(calculate_recurrence)
equipment_df['Tekrarlayan_Arıza_30gün_Flag'] = [r[0] for r in recurrence_results]
equipment_df['Tekrarlayan_Arıza_90gün_Flag'] = [r[1] for r in recurrence_results]

print(f"✓ Recurring faults (30 days): {equipment_df['Tekrarlayan_Arıza_30gün_Flag'].sum():,} equipment")
print(f"✓ Recurring faults (90 days): {equipment_df['Tekrarlayan_Arıza_90gün_Flag'].sum():,} equipment")

# ============================================================================
# STEP 12: SAVE RESULTS
# ============================================================================
print("\n" + "="*100)
print("STEP 12: SAVING RESULTS")
print("="*100)

equipment_df.to_csv('data/equipment_level_data.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ Saved: data/equipment_level_data.csv ({len(equipment_df):,} records)")

# Feature documentation
feature_docs = pd.DataFrame({
    'Feature_Name': equipment_df.columns,
    'Data_Type': equipment_df.dtypes.astype(str),
    'Completeness_%': (equipment_df.notna().sum() / len(equipment_df) * 100).round(1)
})
feature_docs.to_csv('data/feature_documentation.csv', index=False)
print(f"✓ Saved: data/feature_documentation.csv ({len(equipment_df.columns)} features)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
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
print(f"   • Age Precision: DAY-LEVEL (not just year) ✨")
print(f"   • Age Sources: {equipment_df['Age_Source'].value_counts().to_dict()}")
print(f"   • Failure History: 3M, 6M, 12M fault counts")
print(f"   • MTBF: {equipment_df['MTBF_Gün'].notna().sum():,} equipment with valid MTBF")
print(f"   • Recurring Faults: {equipment_df['Tekrarlayan_Arıza_90gün_Flag'].sum():,} equipment flagged")

# Customer impact summary
customer_cols_found = [col for col in customer_impact_cols if any(col.replace(" ", "_") in c for c in equipment_df.columns)]
if customer_cols_found:
    print(f"\n👥 CUSTOMER IMPACT COLUMNS:")
    for col in customer_cols_found[:5]:  # Show first 5
        print(f"   ✓ {col}")
    if len(customer_cols_found) > 5:
        print(f"   ... and {len(customer_cols_found)-5} more")

# Optional specifications summary
optional_cols_found = [col for col in optional_spec_cols.keys() if col in equipment_df.columns]
if optional_cols_found:
    print(f"\n🌟 OPTIONAL SPECIFICATIONS INCLUDED:")
    for col in optional_cols_found:
        coverage = equipment_df[col].notna().sum()
        pct = coverage / len(equipment_df) * 100
        print(f"   ✓ {col}: {coverage:,} ({pct:.1f}% coverage)")

print(f"\n✅ ENHANCEMENTS IN v3.0:")
print(f"   ✨ Day-precision age calculation (365.25 days/year)")
print(f"   ✨ Installation date preserved (Ekipman_Kurulum_Tarihi)")
print(f"   ✨ Age in days available (Ekipman_Yaşı_Gün)")
if USE_FIRST_WORKORDER_FALLBACK:
    wo_count = (equipment_df['Age_Source'] == 'FIRST_WORKORDER_PROXY').sum()
    print(f"   ✨ First work order fallback ({wo_count} equipment)")
print(f"   ✨ Enhanced date validation with diagnostics")
print(f"   ✨ Vectorized operations for better performance")

print(f"\n🚀 READY FOR NEXT PHASE:")
print(f"   → Run: 03_feature_engineering.py")
print(f"   → Create advanced features (age ratios, reliability scores, etc.)")
print("="*100)
