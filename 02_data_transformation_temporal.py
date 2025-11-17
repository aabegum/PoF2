"""
DATA TRANSFORMATION PIPELINE - TEMPORAL VERSION (NO LEAKAGE)
Turkish EDAŞ PoF Prediction Project

Purpose:
- Transform fault-level data to equipment-level features
- CRITICAL: Uses cutoff_date to prevent temporal leakage
- Features calculated ONLY from data before cutoff_date
- Target calculated from data AFTER cutoff_date

Key Difference from 02_data_transformation.py:
- OLD: reference_date = df['started at'].max() (uses ALL data)
- NEW: cutoff_date = user-specified (only historical data)

Usage:
  python 02_data_transformation_temporal.py --cutoff_date 2024-06-25

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import argparse
from datetime import datetime

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

print("="*100)
print(" "*20 + "DATA TRANSFORMATION PIPELINE - TEMPORAL VERSION (v4.0)")
print(" "*30 + "NO TEMPORAL LEAKAGE")
print("="*100)

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Transform fault data with temporal cutoff')
parser.add_argument('--cutoff_date', type=str, required=True,
                    help='Cutoff date (YYYY-MM-DD). Only data BEFORE this date will be used for features.')
parser.add_argument('--prediction_horizon', type=int, default=365,
                    help='Days after cutoff to predict (default: 365 for 12M)')
parser.add_argument('--output_suffix', type=str, default='',
                    help='Optional suffix for output filename (e.g., "_2024Q2")')

args = parser.parse_args()

# Parse cutoff date
try:
    cutoff_date = pd.Timestamp(args.cutoff_date)
except:
    print(f"❌ ERROR: Invalid cutoff_date format: {args.cutoff_date}")
    print("   Expected format: YYYY-MM-DD (e.g., 2024-06-25)")
    sys.exit(1)

prediction_horizon = args.prediction_horizon
target_start = cutoff_date + pd.Timedelta(days=1)
target_end = cutoff_date + pd.Timedelta(days=prediction_horizon)

print(f"\n⚙️  TEMPORAL CONFIGURATION:")
print(f"   Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"   Prediction Horizon: {prediction_horizon} days ({prediction_horizon/30:.1f} months)")
print(f"   Target Period: {target_start.strftime('%Y-%m-%d')} to {target_end.strftime('%Y-%m-%d')}")
print(f"   ✓ NO DATA LEAKAGE: Features use ONLY data before {cutoff_date.strftime('%Y-%m-%d')}")

# Validation constants
MIN_VALID_YEAR = 1950
MAX_VALID_YEAR = cutoff_date.year + 1  # Can't be after cutoff + 1 year

# ============================================================================
# STEP 1: LOAD FAULT-LEVEL DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING FAULT-LEVEL DATA")
print("="*100)

data_path = Path('data/combined_data.xlsx')

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    sys.exit(1)

df = pd.read_excel(data_path)
original_fault_count = len(df)

print(f"\n✓ Loaded: {len(df):,} faults × {df.shape[1]} columns")

# ============================================================================
# CRITICAL: TEMPORAL FILTER
# ============================================================================
print("\n" + "="*100)
print("STEP 1B: TEMPORAL FILTERING (NO LEAKAGE)")
print("="*100)

# Parse started at column
df['started at'] = pd.to_datetime(df['started at'], errors='coerce')

# Filter to ONLY historical data (before cutoff)
print(f"\nFiltering faults to BEFORE cutoff date ({cutoff_date.strftime('%Y-%m-%d')})...")

historical_faults = df[df['started at'] <= cutoff_date].copy()
future_faults = df[df['started at'] > cutoff_date].copy()

print(f"  Historical faults (for features): {len(historical_faults):,} ({len(historical_faults)/len(df)*100:.1f}%)")
print(f"  Future faults (excluded): {len(future_faults):,} ({len(future_faults)/len(df)*100:.1f}%)")

if len(historical_faults) < 100:
    print(f"\n⚠️  WARNING: Only {len(historical_faults)} historical faults available!")
    print(f"   Consider using an earlier cutoff date for more training data.")

# Continue with historical data only
df = historical_faults

print(f"\n✓ TEMPORAL FILTER APPLIED")
print(f"  Features will be calculated from {len(df):,} historical faults only")
print(f"  Reference date for all calculations: {cutoff_date.strftime('%Y-%m-%d')}")

# ============================================================================
# STEP 2: PARSING AND VALIDATING DATE COLUMNS
# ============================================================================
print("\n" + "="*100)
print("STEP 2: PARSING AND VALIDATING DATE COLUMNS (ENHANCED)")
print("="*100)

def parse_and_validate_date(date_series, column_name, min_year=MIN_VALID_YEAR, max_year=MAX_VALID_YEAR,
                            report=True, is_installation_date=False):
    """Parse and validate dates with smart validation"""

    # PRE-CHECK: Reject time-only values
    time_only_mask = pd.Series([False] * len(date_series), index=date_series.index)
    if pd.api.types.is_string_dtype(date_series) or pd.api.types.is_object_dtype(date_series):
        time_only_mask = date_series.astype(str).str.match(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*$', na=False)

    invalid_time_only = time_only_mask.sum()

    # Parse dates
    if pd.api.types.is_numeric_dtype(date_series):
        parsed = pd.to_datetime(date_series, unit='D', origin='1899-12-30', errors='coerce')
    else:
        date_series_clean = date_series.copy()
        date_series_clean[time_only_mask] = None
        parsed = pd.to_datetime(date_series_clean, errors='coerce', dayfirst=True)

    # Validation
    valid_mask = parsed.notna()

    # Reject Excel NULL
    excel_null_mask = (parsed == pd.Timestamp('1900-01-01'))
    valid_mask = valid_mask & ~excel_null_mask

    # Reject out of range years
    year_mask = (parsed.dt.year >= min_year) & (parsed.dt.year <= max_year)
    valid_mask = valid_mask & year_mask

    # For installation dates: reject suspicious recent dates with 00:00:00
    if is_installation_date:
        zero_time_mask = (parsed.notna() & (parsed.dt.hour == 0) &
                         (parsed.dt.minute == 0) & (parsed.dt.second == 0))
        recent_mask = parsed >= (cutoff_date - pd.Timedelta(days=30))
        suspicious_recent = zero_time_mask & recent_mask
        valid_mask = valid_mask & ~suspicious_recent

    # Apply valid mask
    result = parsed.where(valid_mask, None)

    if report:
        valid = valid_mask.sum()
        total = len(date_series)
        status = "✓" if valid/total > 0.95 else ("⚠" if valid/total > 0.80 else "❌")
        extra = f" [Time-only:{invalid_time_only}, NULL:{excel_null_mask.sum()}]" if is_installation_date else ""
        print(f"  {status} {column_name:28s}: {valid:6,}/{total:6,} ({valid/total*100:5.1f}%){extra}")

    return result

print("\nInstallation Dates (rejects NULL + suspicious recent 00:00:00):")
df['TESIS_TARIHI_validated'] = parse_and_validate_date(
    df['TESIS_TARIHI'], 'TESIS_TARIHI', report=True, is_installation_date=True
)
df['EDBS_IDATE_validated'] = parse_and_validate_date(
    df['EDBS_IDATE'], 'EDBS_IDATE', report=True, is_installation_date=False
)

print("\nFault Timestamps (normal validation):")
df['started at'] = parse_and_validate_date(
    df['started at'], 'started at', report=True, is_installation_date=False
)
df['ended at'] = parse_and_validate_date(
    df.get('ended at', pd.Series()), 'ended at', report=True, is_installation_date=False
)

print("\nWork Order Dates:")
if 'İş Emri Oluşturma Tarihi' in df.columns:
    df['Work_Order_Created'] = parse_and_validate_date(
        df['İş Emri Oluşturma Tarihi'], 'Work Order Creation', report=True
    )

# ============================================================================
# STEP 3: CALCULATING EQUIPMENT AGE (AS OF CUTOFF DATE)
# ============================================================================
print("\n" + "="*100)
print(f"STEP 3: CALCULATING EQUIPMENT AGE (AS OF {cutoff_date.strftime('%Y-%m-%d')})")
print("="*100)

def calculate_age_from_date(install_date, reference_date):
    """Calculate age in days and years from installation date to reference date"""
    if pd.isna(install_date):
        return None, None, None

    age_days = (reference_date - install_date).days

    if age_days < 0:
        return None, None, None  # Installation after reference (invalid)

    age_years = age_days / 365.25

    return install_date, age_days, age_years

print("\nCalculating dual age features (TESIS-primary + EDBS-primary)...")

# Initialize age columns
for prefix in ['', '_TESIS', '_EDBS']:
    df[f'Ekipman_Kurulum_Tarihi{prefix}'] = None
    df[f'Ekipman_Yaşı_Gün{prefix}'] = None
    df[f'Ekipman_Yaşı_Yıl{prefix}'] = None
    df[f'Yaş_Kaynak{prefix}'] = None

# Calculate ages
for idx, row in df.iterrows():
    # TESIS-primary age
    if pd.notna(row['TESIS_TARIHI_validated']):
        install, days, years = calculate_age_from_date(row['TESIS_TARIHI_validated'], cutoff_date)
        if install is not None:
            df.at[idx, 'Ekipman_Kurulum_Tarihi_TESIS'] = install
            df.at[idx, 'Ekipman_Yaşı_Gün_TESIS'] = days
            df.at[idx, 'Ekipman_Yaşı_Yıl_TESIS'] = years
            df.at[idx, 'Yaş_Kaynak_TESIS'] = 'TESIS'

    # EDBS-primary age
    if pd.notna(row['EDBS_IDATE_validated']):
        install, days, years = calculate_age_from_date(row['EDBS_IDATE_validated'], cutoff_date)
        if install is not None:
            df.at[idx, 'Ekipman_Kurulum_Tarihi_EDBS'] = install
            df.at[idx, 'Ekipman_Yaşı_Gün_EDBS'] = days
            df.at[idx, 'Ekipman_Yaşı_Yıl_EDBS'] = years
            df.at[idx, 'Yaş_Kaynak_EDBS'] = 'EDBS'

    # Default (EDBS-primary, fallback to TESIS)
    if pd.notna(row['EDBS_IDATE_validated']):
        install, days, years = calculate_age_from_date(row['EDBS_IDATE_validated'], cutoff_date)
        if install is not None:
            df.at[idx, 'Ekipman_Kurulum_Tarihi'] = install
            df.at[idx, 'Ekipman_Yaşı_Gün'] = days
            df.at[idx, 'Ekipman_Yaşı_Yıl'] = years
            df.at[idx, 'Age_Source'] = 'EDBS'
    elif pd.notna(row['TESIS_TARIHI_validated']):
        install, days, years = calculate_age_from_date(row['TESIS_TARIHI_validated'], cutoff_date)
        if install is not None:
            df.at[idx, 'Ekipman_Kurulum_Tarihi'] = install
            df.at[idx, 'Ekipman_Yaşı_Gün'] = days
            df.at[idx, 'Ekipman_Yaşı_Yıl'] = years
            df.at[idx, 'Age_Source'] = 'TESIS'
    else:
        df.at[idx, 'Age_Source'] = 'MISSING'

# Summary
source_counts = df['Age_Source'].value_counts()
valid_ages = df[df['Age_Source'] != 'MISSING']['Ekipman_Yaşı_Yıl']

print(f"\n✓ Age Calculation Complete (as of {cutoff_date.strftime('%Y-%m-%d')}):")
print(f"  Sources: ", end="")
print(" | ".join([f"{src}:{cnt:,}({cnt/len(df)*100:.1f}%)" for src, cnt in source_counts.items()]))
if len(valid_ages) > 0:
    print(f"  Range: {valid_ages.min():.1f}-{valid_ages.max():.1f}y, Mean={valid_ages.mean():.1f}y, Median={valid_ages.median():.1f}y")

# ============================================================================
# STEP 3B: FILL MISSING AGES WITH FIRST WORK ORDER (FALLBACK)
# ============================================================================
print("\n" + "="*100)
print("STEP 3B: FILLING MISSING AGES WITH FIRST WORK ORDER (VECTORIZED)")
print("="*100)

missing_age_count = (df['Age_Source'] == 'MISSING').sum()

if missing_age_count > 0:
    print(f"\n  Equipment with MISSING age (EDBS-primary): {missing_age_count} ({missing_age_count/len(df)*100:.1f}%)")
    print(f"  Attempting to use first work order date as proxy...")

    # Determine equipment ID column
    if 'cbs_id' in df.columns:
        equipment_id_col = 'cbs_id'
    elif 'Ekipman ID' in df.columns:
        equipment_id_col = 'Ekipman ID'
    else:
        print("  ⚠ No equipment ID column found - skipping fallback")
        equipment_id_col = None

    if equipment_id_col is not None and 'Work_Order_Created' in df.columns:
        print(f"\n  Using equipment ID column: {equipment_id_col}")

        # Get first work order date per equipment
        first_wo = df[df['Work_Order_Created'].notna()].groupby(equipment_id_col)['Work_Order_Created'].min()

        # Fill missing ages
        filled_count = 0
        for idx, row in df[df['Age_Source'] == 'MISSING'].iterrows():
            equip_id = row[equipment_id_col]
            if pd.notna(equip_id) and equip_id in first_wo.index:
                first_wo_date = first_wo[equip_id]
                install, days, years = calculate_age_from_date(first_wo_date, cutoff_date)
                if install is not None:
                    df.at[idx, 'Ekipman_Kurulum_Tarihi'] = install
                    df.at[idx, 'Ekipman_Yaşı_Gün'] = days
                    df.at[idx, 'Ekipman_Yaşı_Yıl'] = years
                    df.at[idx, 'Age_Source'] = 'FIRST_WORKORDER_PROXY'

                    # Also fill EDBS columns
                    df.at[idx, 'Ekipman_Kurulum_Tarihi_EDBS'] = install
                    df.at[idx, 'Ekipman_Yaşı_Gün_EDBS'] = days
                    df.at[idx, 'Ekipman_Yaşı_Yıl_EDBS'] = years
                    df.at[idx, 'Yaş_Kaynak_EDBS'] = 'WORKORDER'

                    # Also fill TESIS columns
                    df.at[idx, 'Ekipman_Kurulum_Tarihi_TESIS'] = install
                    df.at[idx, 'Ekipman_Yaşı_Gün_TESIS'] = days
                    df.at[idx, 'Ekipman_Yaşı_Yıl_TESIS'] = years
                    df.at[idx, 'Yaş_Kaynak_TESIS'] = 'WORKORDER'

                    filled_count += 1

        remaining_missing = (df['Age_Source'] == 'MISSING').sum()
        print(f"  ✓ Filled (EDBS-primary): {filled_count} using first work order proxy")
        print(f"  ✓ Filled (TESIS-primary): {filled_count} using first work order proxy")
        print(f"  ✓ Remaining MISSING: {remaining_missing} ({remaining_missing/len(df)*100:.1f}%)")

        # Updated distribution
        print(f"\n  Updated Age Source Distribution (EDBS-primary):")
        for src, cnt in df['Age_Source'].value_counts().items():
            print(f"    {src:20s}: {cnt:6,} ({cnt/len(df)*100:5.1f}%)")

# ============================================================================
# STEP 4/5: TEMPORAL FEATURES & FAILURE PERIODS (RELATIVE TO CUTOFF)
# ============================================================================
print("\n" + "="*80)
print("STEP 4/5: TEMPORAL FEATURES & FAILURE PERIODS (RELATIVE TO CUTOFF)")
print("="*80)

# Calculate time to repair
if 'ended at' in df.columns:
    df['Time_To_Repair_Hours'] = (df['ended at'] - df['started at']).dt.total_seconds() / 3600
    df['Time_To_Repair_Hours'] = df['Time_To_Repair_Hours'].clip(lower=0, upper=168)  # Max 1 week

# Seasonal flags
df['Month'] = df['started at'].dt.month
df['Summer_Peak_Flag'] = df['Month'].isin([6, 7, 8]).astype(int)
df['Winter_Peak_Flag'] = df['Month'].isin([12, 1, 2]).astype(int)

# CRITICAL: Failure period flags relative to CUTOFF DATE (not max date!)
cutoff_3m = cutoff_date - pd.Timedelta(days=90)
cutoff_6m = cutoff_date - pd.Timedelta(days=180)
cutoff_12m = cutoff_date - pd.Timedelta(days=365)

df['Fault_Last_3M'] = (df['started at'] >= cutoff_3m).astype(int)
df['Fault_Last_6M'] = (df['started at'] >= cutoff_6m).astype(int)
df['Fault_Last_12M'] = (df['started at'] >= cutoff_12m).astype(int)

print(f"\n✓ Temporal: Summer={df['Summer_Peak_Flag'].sum():,}, Winter={df['Winter_Peak_Flag'].sum():,}, Avg repair={df['Time_To_Repair_Hours'].mean():.1f}h")
print(f"✓ Periods (ref={cutoff_date.strftime('%Y-%m-%d')}): 3M={df['Fault_Last_3M'].sum():,}, 6M={df['Fault_Last_6M'].sum():,}, 12M={df['Fault_Last_12M'].sum():,}")

# ============================================================================
# STEP 5B: CUSTOMER IMPACT RATIOS (FAULT-LEVEL)
# ============================================================================
print("\n" + "="*80)
print("STEP 5B: CALCULATING CUSTOMER IMPACT RATIOS (FAULT-LEVEL)")
print("="*80)

customer_ratio_cols = []

if 'total customer count' in df.columns:
    total_customers = df['total customer count'].fillna(0)

    # Urban customer ratio
    if 'urban mv' in df.columns and 'urban lv' in df.columns:
        df['Urban_Customer_Ratio'] = (
            (df['urban mv'].fillna(0) + df['urban lv'].fillna(0)) /
            (total_customers + 1)
        ).clip(0, 1)
        customer_ratio_cols.append('Urban_Customer_Ratio')

    # Rural customer ratio
    if 'rural mv' in df.columns and 'rural lv' in df.columns:
        df['Rural_Customer_Ratio'] = (
            (df['rural mv'].fillna(0) + df['rural lv'].fillna(0)) /
            (total_customers + 1)
        ).clip(0, 1)
        customer_ratio_cols.append('Rural_Customer_Ratio')

    # MV customer ratio
    if 'urban mv' in df.columns and 'rural mv' in df.columns:
        suburban_mv = df['suburban mv'].fillna(0) if 'suburban mv' in df.columns else 0
        df['MV_Customer_Ratio'] = (
            (df['urban mv'].fillna(0) + suburban_mv + df['rural mv'].fillna(0)) /
            (total_customers + 1)
        ).clip(0, 1)
        customer_ratio_cols.append('MV_Customer_Ratio')

    if customer_ratio_cols:
        print(f"✓ Created {len(customer_ratio_cols)} fault-level customer ratios:")
        for col in customer_ratio_cols:
            print(f"  • {col}: Mean={df[col].mean():.2%}, Max={df[col].max():.2%}")
else:
    print("⚠ 'total customer count' column not found - skipping ratio calculation")

# ============================================================================
# STEP 6: EQUIPMENT IDENTIFICATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: EQUIPMENT IDENTIFICATION (SIMPLIFIED)")
print("="*80)

def get_equipment_id(row):
    """Smart Equipment ID with fallback"""
    if pd.notna(row.get('cbs_id')):
        return row['cbs_id']
    elif pd.notna(row.get('Ekipman ID')):
        return row['Ekipman ID']
    else:
        return f"UNKNOWN_{row.name}"

df['Ekipman_ID'] = df.apply(get_equipment_id, axis=1)

# Count unique equipment
unique_equip = df['Ekipman_ID'].nunique()
unknown_count = df['Ekipman_ID'].astype(str).str.startswith('UNKNOWN_', na=False).sum()

print(f"\n✓ ID Strategy: cbs_id({df['cbs_id'].notna().sum():,}) → Ekipman ID({unknown_count}) → Generated({unknown_count})")
print(f"  Total: {unique_equip:,} unique equipment from {len(df):,} faults (avg {len(df)/unique_equip:.1f} faults/equip)")

# ============================================================================
# STEP 6B: EQUIPMENT CLASS HARMONIZATION
# ============================================================================
print("\n--- Smart Equipment Classification Selection ---")

# Create unified Equipment_Class_Primary column
def get_equipment_class_primary(row):
    """Get primary equipment class with priority"""
    for col in ['Equipment_Type', 'Ekipman Sınıfı', 'Kesinti Ekipman Sınıfı', 'Ekipman Sınıf']:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return None

df['Equipment_Class_Primary'] = df.apply(get_equipment_class_primary, axis=1)

coverage = df['Equipment_Class_Primary'].notna().sum()
unique_types = df['Equipment_Class_Primary'].nunique()

print(f"✓ Unified Equipment Class created:")
print(f"  Priority: Equipment_Type → Ekipman Sınıfı → Kesinti Ekipman Sınıfı")
print(f"  Coverage: {coverage:,} ({coverage/len(df)*100:.1f}%)")
print(f"  Unique types (before harmonization): {unique_types}")

# Harmonization mapping
class_mapping = {
    'aghat': 'AG Hat',
    'AG Hat': 'AG Hat',
    'REKORTMAN': 'Rekortman',
    'Rekortman': 'Rekortman',
    'agdirek': 'AG Direk',
    'AG Direk': 'AG Direk',
    'OGAGTRF': 'OG/AG Trafo',
    'OG/AG Trafo': 'OG/AG Trafo',
    'Trafo Bina Tip': 'OG/AG Trafo',
    'SDK': 'AG Pano Box',
    'AG Pano': 'AG Pano Box',
    'AG Pano Box': 'AG Pano Box',
    'anahtar': 'AG Anahtar',
    'AG Anahtar': 'AG Anahtar',
    'OGHAT': 'OG Hat',
    'OG Hat': 'OG Hat',
    'PANO': 'Pano',
    'Pano': 'Pano',
    'Ayırıcı': 'Ayırıcı',
    'KESİCİ': 'Kesici',
    'Kesici': 'Kesici',
    'Armatür': 'Armatür',
    'ENHDirek': 'ENH Direk',
    'ENH Direk': 'ENH Direk',
    'Bina': 'Bina',
}

df['Equipment_Class_Primary'] = df['Equipment_Class_Primary'].map(class_mapping).fillna(df['Equipment_Class_Primary'])

harmonized_types = df['Equipment_Class_Primary'].nunique()
print(f"\n--- Equipment Class Harmonization ---")
print(f"✓ Equipment classes harmonized:")
print(f"  Before: {unique_types} types → After: {harmonized_types} types")

# ============================================================================
# STEP 7: AGGREGATE TO EQUIPMENT LEVEL
# ============================================================================
print("\n" + "="*100)
print("STEP 7: AGGREGATING TO EQUIPMENT LEVEL")
print("="*100)

# Sort by TESIS date priority
df_sorted = df.sort_values('TESIS_TARIHI_validated', ascending=False, na_position='last')

print(f"\n  ✓ Sorted data to prioritize TESIS_TARIHI as primary age source during aggregation")

equipment_id_col = 'Ekipman_ID'

# Aggregation dictionary
agg_dict = {
    # Geographic
    'KOORDINAT_X': 'first',
    'KOORDINAT_Y': 'first',
    'İl': 'first',
    'İlçe': 'first',
    'Mahalle': 'first',

    # Age data (all variants)
    'Ekipman_Kurulum_Tarihi': 'first',
    'Ekipman_Yaşı_Gün': 'first',
    'Ekipman_Yaşı_Yıl': 'first',
    'Age_Source': 'first',

    'Ekipman_Kurulum_Tarihi_TESIS': 'first',
    'Ekipman_Yaşı_Gün_TESIS': 'first',
    'Ekipman_Yaşı_Yıl_TESIS': 'first',
    'Yaş_Kaynak_TESIS': 'first',

    'Ekipman_Kurulum_Tarihi_EDBS': 'first',
    'Ekipman_Yaşı_Gün_EDBS': 'first',
    'Ekipman_Yaşı_Yıl_EDBS': 'first',
    'Yaş_Kaynak_EDBS': 'first',

    # Fault history
    'started at': ['count', 'min', 'max'],
    'Fault_Last_3M': 'sum',
    'Fault_Last_6M': 'sum',
    'Fault_Last_12M': 'sum',

    # Temporal features
    'Summer_Peak_Flag': 'sum',
    'Winter_Peak_Flag': 'sum',
    'Time_To_Repair_Hours': ['mean', 'max'],
}

# Add customer ratio columns
for ratio_col in customer_ratio_cols:
    if ratio_col in df_sorted.columns:
        agg_dict[ratio_col] = 'mean'

# Add cause code if available
if 'cause code' in df_sorted.columns:
    agg_dict['cause code'] = ['first', 'last', lambda x: x.mode()[0] if len(x.mode()) > 0 else None]
    print("\n  ✓ Found: cause code (will aggregate first, last, and most common)")

# Add customer impact columns
customer_impact_cols = [
    'urban mv+suburban mv', 'urban lv+suburban lv',
    'urban mv', 'urban lv', 'suburban mv', 'suburban lv',
    'rural mv', 'rural lv', 'total customer count'
]

print("  Checking for customer impact columns...")
for col in customer_impact_cols:
    if col in df_sorted.columns:
        agg_dict[col] = ['mean', 'max']
        print(f"  ✓ Found: {col}")

# Add optional specification columns
optional_spec_cols = {
    'component voltage': 'first',
    'MARKA': 'first',
    'MARKA_MODEL': 'first',
    'FIRMA': 'first'
}

print("  Checking for optional specification columns...")
for col, agg_func in optional_spec_cols.items():
    if col in df_sorted.columns:
        agg_dict[col] = agg_func
        print(f"  ✓ Found: {col}")

# Add Equipment_Class_Primary
if 'Equipment_Class_Primary' in df_sorted.columns:
    agg_dict['Equipment_Class_Primary'] = 'first'

print(f"\n✓ Aggregating {len(df_sorted):,} fault records to equipment level...")
equipment_df = df_sorted.groupby(equipment_id_col).agg(agg_dict).reset_index()
equipment_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in equipment_df.columns.values]

print(f"✓ Created {len(equipment_df):,} equipment records from {original_fault_count:,} faults")

# ============================================================================
# STEP 8: RENAME COLUMNS
# ============================================================================
print("\n" + "="*100)
print("STEP 8: CREATING FINAL FEATURES")
print("="*100)

# Rename aggregated columns
rename_dict = {
    'started at_count': 'Toplam_Arıza_Sayisi_Lifetime',
    'started at_min': 'İlk_Arıza_Tarihi',
    'started at_max': 'Son_Arıza_Tarihi',
    'Fault_Last_3M_sum': 'Arıza_Sayısı_3ay',
    'Fault_Last_6M_sum': 'Arıza_Sayısı_6ay',
    'Fault_Last_12M_sum': 'Arıza_Sayısı_12ay',
}

# Add cause code renames if applicable
if 'cause code_first' in equipment_df.columns:
    rename_dict['cause code_first'] = 'Arıza_Nedeni_İlk'
if 'cause code_last' in equipment_df.columns:
    rename_dict['cause code_last'] = 'Arıza_Nedeni_Son'

equipment_df = equipment_df.rename(columns=rename_dict)

# Add cause code features
if 'cause code_<lambda_0>' in equipment_df.columns:
    print("\nCalculating cause code features...")

    # Cause diversity
    cause_diversity = df_sorted.groupby('Ekipman_ID')['cause code'].nunique()
    equipment_df['Arıza_Nedeni_Çeşitlilik'] = equipment_df['Ekipman_ID'].map(cause_diversity).fillna(0)

    # Cause consistency
    total_faults_per_equip = df_sorted.groupby('Ekipman_ID').size()
    max_cause_per_equip = df_sorted.groupby(['Ekipman_ID', 'cause code']).size().groupby('Ekipman_ID').max()
    cause_consistency = (max_cause_per_equip / total_faults_per_equip).reindex(equipment_df['Ekipman_ID']).fillna(0).values
    equipment_df['Arıza_Nedeni_Tutarlılık'] = cause_consistency

    print(f"  ✓ Created Arıza_Nedeni_Çeşitlilik (cause diversity)")
    print(f"  ✓ Created Arıza_Nedeni_Tutarlılık (cause consistency)")

# ============================================================================
# STEP 10: CALCULATE MTBF
# ============================================================================
print("\nCalculating MTBF (Mean Time Between Failures)...")

def calculate_mtbf(group):
    """Calculate MTBF from fault dates"""
    if len(group) < 2:
        return None

    fault_dates = sorted(group['started at'].dropna())

    if len(fault_dates) < 2:
        return None

    intervals = [(fault_dates[i+1] - fault_dates[i]).days for i in range(len(fault_dates)-1)]

    if len(intervals) == 0:
        return None

    mtbf_days = np.mean(intervals)

    return mtbf_days

mtbf_values = df_sorted.groupby('Ekipman_ID').apply(calculate_mtbf)
equipment_df['MTBF_Gün'] = equipment_df['Ekipman_ID'].map(mtbf_values)

mtbf_available = equipment_df['MTBF_Gün'].notna().sum()
print(f"  ✓ MTBF calculable for {mtbf_available:,} equipment")

# ============================================================================
# STEP 11: DAYS SINCE LAST FAILURE (AS OF CUTOFF DATE)
# ============================================================================
print("\nCalculating days since last failure (as of cutoff date)...")

equipment_df['Son_Arıza_Gun_Sayisi'] = (cutoff_date - equipment_df['Son_Arıza_Tarihi']).dt.days

print(f"  ✓ Days since last failure calculated (reference: {cutoff_date.strftime('%Y-%m-%d')})")

# ============================================================================
# STEP 12: RECURRING FAULTS (WITHIN HISTORICAL PERIOD)
# ============================================================================
print("\n" + "="*100)
print("STEP 12: DETECTING RECURRING FAULTS")
print("="*100)

print("\nAnalyzing recurring fault patterns...")

def detect_recurring_faults(group, window_days=90):
    """Detect if equipment has recurring faults within window"""
    fault_dates = sorted(group['started at'].dropna())

    if len(fault_dates) < 2:
        return 0

    for i in range(len(fault_dates)-1):
        time_diff = (fault_dates[i+1] - fault_dates[i]).days
        if time_diff <= window_days:
            return 1

    return 0

# 30-day recurring
recurring_30 = df_sorted.groupby('Ekipman_ID').apply(lambda g: detect_recurring_faults(g, 30))
equipment_df['Tekrarlayan_Arıza_30gün_Flag'] = equipment_df['Ekipman_ID'].map(recurring_30).fillna(0).astype(int)

# 90-day recurring
recurring_90 = df_sorted.groupby('Ekipman_ID').apply(lambda g: detect_recurring_faults(g, 90))
equipment_df['Tekrarlayan_Arıza_90gün_Flag'] = equipment_df['Ekipman_ID'].map(recurring_90).fillna(0).astype(int)

print(f"✓ Recurring faults (30 days): {equipment_df['Tekrarlayan_Arıza_30gün_Flag'].sum():,} equipment")
print(f"✓ Recurring faults (90 days): {equipment_df['Tekrarlayan_Arıza_90gün_Flag'].sum():,} equipment")

# ============================================================================
# STEP 13: CREATE TARGET VARIABLE (FROM FUTURE DATA)
# ============================================================================
print("\n" + "="*100)
print("STEP 13: CREATING TARGET VARIABLE (FROM FUTURE DATA)")
print("="*100)

# Load ALL fault data (including future)
df_all = pd.read_excel(data_path)
df_all['started at'] = pd.to_datetime(df_all['started at'], errors='coerce')

# Filter to target period
target_faults = df_all[
    (df_all['started at'] > target_start) &
    (df_all['started at'] <= target_end)
].copy()

print(f"\nTarget Period: {target_start.strftime('%Y-%m-%d')} to {target_end.strftime('%Y-%m-%d')}")
print(f"  Faults in target period: {len(target_faults):,}")

# Add equipment ID to target faults
target_faults['Ekipman_ID'] = target_faults.apply(get_equipment_id, axis=1)

# Count faults per equipment in target period
target_counts = target_faults.groupby('Ekipman_ID').size()

# Create binary target (failure yes/no)
equipment_df['Target_Failure_Binary'] = 0
equipment_df.loc[equipment_df['Ekipman_ID'].isin(target_counts.index), 'Target_Failure_Binary'] = 1

# Create count target (number of failures)
equipment_df['Target_Failure_Count'] = equipment_df['Ekipman_ID'].map(target_counts).fillna(0).astype(int)

# Statistics
failures = equipment_df['Target_Failure_Binary'].sum()
no_failures = len(equipment_df) - failures

print(f"\n✓ Target Variable Created:")
print(f"  Equipment with failures in target period: {failures:,} ({failures/len(equipment_df)*100:.1f}%)")
print(f"  Equipment without failures: {no_failures:,} ({no_failures/len(equipment_df)*100:.1f}%)")
print(f"  Total target faults: {equipment_df['Target_Failure_Count'].sum():,}")
print(f"  Avg faults per failing equipment: {equipment_df[equipment_df['Target_Failure_Binary']==1]['Target_Failure_Count'].mean():.2f}")

# ============================================================================
# STEP 14: SAVE RESULTS
# ============================================================================
print("\n" + "="*100)
print("STEP 14: SAVING RESULTS")
print("="*100)

# Create output filename with suffix
output_suffix = args.output_suffix if args.output_suffix else f"_{cutoff_date.strftime('%Y%m%d')}"
output_path = Path(f'data/equipment_level_data_temporal{output_suffix}.csv')

equipment_df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path} ({len(equipment_df):,} records)")

# Create metadata file
metadata = {
    'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
    'prediction_horizon_days': prediction_horizon,
    'target_start': target_start.strftime('%Y-%m-%d'),
    'target_end': target_end.strftime('%Y-%m-%d'),
    'historical_faults': len(df),
    'target_faults': len(target_faults),
    'equipment_count': len(equipment_df),
    'failure_rate': float(failures / len(equipment_df)),
    'features': equipment_df.shape[1],
}

import json
metadata_path = Path(f'data/equipment_level_data_temporal{output_suffix}_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved: {metadata_path} (metadata)")

# Save feature documentation
feature_docs = pd.DataFrame({
    'Feature': equipment_df.columns,
    'Type': equipment_df.dtypes.astype(str),
    'Non_Null': equipment_df.notna().sum(),
    'Null_Count': equipment_df.isnull().sum(),
})
feature_docs.to_csv(f'data/feature_documentation_temporal{output_suffix}.csv', index=False)
print(f"✓ Saved: data/feature_documentation_temporal{output_suffix}.csv ({len(feature_docs)} features)")

# ============================================================================
# TRANSFORMATION COMPLETE
# ============================================================================
print("\n" + "="*100)
print("TRANSFORMATION COMPLETE - TEMPORAL VERSION (NO LEAKAGE)")
print("="*100)

print(f"\n📊 TRANSFORMATION SUMMARY:")
print(f"   • Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"   • Historical Faults: {len(df):,} (for features)")
print(f"   • Target Faults: {len(target_faults):,} (for target variable)")
print(f"   • Equipment Records: {len(equipment_df):,}")
print(f"   • Failure Rate: {failures/len(equipment_df)*100:.1f}%")
print(f"   • Total Features: {equipment_df.shape[1]} columns")

print(f"\n🎯 KEY VALIDATION:")
print(f"   ✓ NO TEMPORAL LEAKAGE: Features use only data before {cutoff_date.strftime('%Y-%m-%d')}")
print(f"   ✓ Target from future: {target_start.strftime('%Y-%m-%d')} to {target_end.strftime('%Y-%m-%d')}")
print(f"   ✓ Proper train/test split for ML modeling")

print(f"\n🚀 READY FOR FEATURE ENGINEERING:")
print(f"   → Run: python 03_feature_engineering.py --input {output_path}")
print(f"   → Or continue with model training")

print("\n" + "="*100)
print(f"{'TEMPORAL TRANSFORMATION PIPELINE COMPLETE':^100}")
print("="*100)
