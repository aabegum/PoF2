"""
DATA TRANSFORMATION: FAULT-LEVEL → EQUIPMENT-LEVEL v3.1 (ENHANCED)
Turkish EDAŞ PoF Prediction Project

ENHANCEMENTS in v3.1:
✓ Smart date validation: Rejects Excel NULL + suspicious recent dates (not all 00:00:00)
✓ Preserves valid dates with 00:00:00 timestamps (normal Excel date storage)
✓ Simplified Equipment ID (prevents grouping bug): cbs_id → Ekipman ID → Generated unique ID
✓ Day-precision age calculation (not just year)
✓ Optional first work order fallback for missing ages
✓ Vectorized operations for better performance
✓ Complete audit trail (install date, age source, age in days)

Key Features:
✓ Smart Equipment ID (SIMPLIFIED - prevents grouping bug)
✓ Unified Equipment Classification (Equipment_Type → Ekipman Sınıfı → fallbacks)
✓ Age source tracking (TESIS_TARIHI vs EDBS_IDATE vs FIRST_WORKORDER_PROXY)
✓ Professional date validation (rejects Excel NULL + suspicious recent dates only)
✓ Failure history aggregation (3/6/12 months)
✓ MTBF calculation
✓ Recurring fault detection (30/90 days)
✓ Customer impact columns (all MV/LV categories)
✓ Optional specifications (voltage_level, kVa_rating) - future-proof

Priority Logic:
- Equipment ID: cbs_id → Ekipman ID → Generated unique ID (no grouping)
- Equipment Class: Equipment_Type → Ekipman Sınıfı → Kesinti Ekipman Sınıfı
- Installation Date: TESIS_TARIHI → EDBS_IDATE → First Work Order (optional)
- Date Validation: Rejects Excel NULL (1900-01-01) + suspicious recent dates with 00:00:00

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

# Constants (dynamic - updates based on current date)
CURRENT_YEAR = datetime.now().year
MIN_VALID_YEAR = 1950
MAX_VALID_YEAR = datetime.now().year + 1  # Allow dates up to next year for data entry flexibility
REFERENCE_DATE = pd.Timestamp(datetime.now())  # Use current date as reference

# Feature flags
USE_FIRST_WORKORDER_FALLBACK = True  # Set to True to enable Option 3 (first work order as age proxy)

print("="*100)
print(" "*25 + "DATA TRANSFORMATION PIPELINE v3.1 (ENHANCED)")
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

def parse_and_validate_date(date_series, column_name, min_year=MIN_VALID_YEAR, max_year=MAX_VALID_YEAR,
                            report=True, is_installation_date=False):
    """
    Parse and validate dates with smart validation
    Rejects Excel NULL + time-only values + suspicious recent dates

    Args:
        date_series: Series of date values
        column_name: Name for reporting
        min_year: Minimum valid year (default: 1950)
        max_year: Maximum valid year (default: 2025)
        report: Whether to print statistics (default: True)
        is_installation_date: If True, reject recent dates with 00:00:00 (likely defaults)

    Returns:
        Series of validated datetime values (invalid → NaT)

    Note:
        - ALWAYS rejects: 1900-01-01 (Excel NULL), time-only values like "00:00:00"
        - For installation dates: Rejects recent (<30 days) dates with 00:00:00 only
        - Preserves: Old dates with 00:00:00 (normal Excel date storage)
    """
    # PRE-CHECK: Reject time-only values (e.g., "00:00:00", "12:30:00") before parsing
    time_only_mask = pd.Series([False] * len(date_series), index=date_series.index)
    if pd.api.types.is_string_dtype(date_series) or pd.api.types.is_object_dtype(date_series):
        # Check if value matches time pattern (HH:MM:SS or similar)
        time_only_mask = date_series.astype(str).str.match(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*$', na=False)

    invalid_time_only = time_only_mask.sum()

    # Check if data is Excel serial date (integer/float format)
    if pd.api.types.is_numeric_dtype(date_series):
        # Excel serial dates: days since 1900-01-01 (Windows Excel)
        # Valid range: ~18263 (1950) to ~45657 (2025)
        # Origin = 1899-12-30 because Excel incorrectly treats 1900 as leap year
        parsed = pd.to_datetime(date_series, unit='D', origin='1899-12-30', errors='coerce')
    else:
        # Parse dates with Turkish date format support (DD/MM/YYYY)
        # Set time-only values to NaT before parsing
        date_series_clean = date_series.copy()
        date_series_clean[time_only_mask] = None
        parsed = pd.to_datetime(date_series_clean, errors='coerce', dayfirst=True)

    # Initialize validation masks
    valid_mask = (
        parsed.notna() &
        (parsed.dt.year >= min_year) &
        (parsed.dt.year <= max_year)
    )

    # 1. ALWAYS reject Excel NULL (1900-01-01 exactly)
    excel_null_mask = (parsed == pd.Timestamp('1900-01-01'))
    invalid_excel_null = excel_null_mask.sum()
    valid_mask = valid_mask & ~excel_null_mask

    # 2. For INSTALLATION dates: Reject ONLY suspicious 00:00:00 timestamps
    #    Valid: 2015-03-20 00:00:00 (normal date storage)
    #    Invalid: Recent dates with 00:00:00 (likely Excel =TODAY() defaults)
    invalid_zero_time = 0
    invalid_recent = 0

    if is_installation_date:
        # Identify dates with 00:00:00 timestamp
        zero_time_mask = (
            parsed.notna() &
            (parsed.dt.hour == 0) &
            (parsed.dt.minute == 0) &
            (parsed.dt.second == 0)
        )

        # Only reject if ALSO within 30 days of reference date (suspicious)
        recent_mask = parsed >= (REFERENCE_DATE - pd.Timedelta(days=30))
        suspicious_recent = zero_time_mask & recent_mask

        invalid_zero_time = suspicious_recent.sum()
        invalid_recent = suspicious_recent.sum()
        valid_mask = valid_mask & ~suspicious_recent

        # Note: Old dates with 00:00:00 are KEPT (normal date storage)

    # Categorize other invalid dates
    invalid_old = (parsed.notna() & (parsed.dt.year < min_year) & ~excel_null_mask).sum()
    invalid_future = (parsed.notna() & (parsed.dt.year > max_year)).sum()

    # Set invalid to NaT
    parsed[~valid_mask] = pd.NaT

    # Report statistics (compact format)
    if report:
        total = len(date_series)
        valid = valid_mask.sum()
        invalid_total = invalid_time_only + invalid_excel_null + invalid_zero_time + invalid_old + invalid_future

        status = "✓" if valid/total > 0.95 else ("⚠" if valid/total > 0.80 else "❌")
        print(f"  {status} {column_name:28s}: {valid:6,}/{total:6,} ({valid/total*100:5.1f}%) ", end="")

        # Show only significant issues in one line
        issues = []
        if invalid_time_only > 0:
            issues.append(f"Time-only:{invalid_time_only}")
        if invalid_excel_null > 0:
            issues.append(f"NULL:{invalid_excel_null}")
        if invalid_zero_time > 0:
            issues.append(f"Suspicious recent:{invalid_zero_time}")
        if invalid_old > 0:
            issues.append(f"<{min_year}:{invalid_old}")
        if invalid_future > 0:
            issues.append(f">{max_year}:{invalid_future}")

        if issues:
            print(f"[{', '.join(issues)}]")
        else:
            print()

    return parsed

# Parse and validate all date columns
print("\nInstallation Dates (rejects NULL + suspicious recent 00:00:00):")
df['TESIS_TARIHI_parsed'] = parse_and_validate_date(df['TESIS_TARIHI'], 'TESIS_TARIHI', is_installation_date=True)
df['EDBS_IDATE_parsed'] = parse_and_validate_date(df['EDBS_IDATE'], 'EDBS_IDATE', is_installation_date=True)

print("\nFault Timestamps (normal validation):")
df['started at'] = parse_and_validate_date(df['started at'], 'started at', min_year=2020, report=True, is_installation_date=False)
df['ended at'] = parse_and_validate_date(df['ended at'], 'ended at', min_year=2020, report=True, is_installation_date=False)

# Parse work order creation date (for fallback option)
if 'Oluşturma Tarihi Sıralama' in df.columns or 'Oluşturulma_Tarihi' in df.columns:
    creation_col = 'Oluşturma Tarihi Sıralama' if 'Oluşturma Tarihi Sıralama' in df.columns else 'Oluşturulma_Tarihi'
    print("\nWork Order Dates:")
    df['Oluşturulma_Tarihi'] = parse_and_validate_date(df[creation_col], 'Work Order Creation', min_year=2015, report=True, is_installation_date=False)
else:
    df['Oluşturulma_Tarihi'] = pd.NaT

# ============================================================================
# STEP 3: ENHANCED EQUIPMENT AGE CALCULATION
# ============================================================================
print("\n" + "="*100)
print("STEP 3: CALCULATING EQUIPMENT AGE (DAY PRECISION)")
print("="*100)

def calculate_age_tesis_priority(row):
    """
    Calculate age with TESIS_TARIHI as PRIMARY (commissioning/database entry date)
    Priority: TESIS_TARIHI → EDBS_IDATE → Work Order
    """
    ref_date = REFERENCE_DATE

    if pd.notna(row['TESIS_TARIHI_parsed']) and row['TESIS_TARIHI_parsed'] < ref_date:
        age_days = (ref_date - row['TESIS_TARIHI_parsed']).days
        return age_days, 'TESIS', row['TESIS_TARIHI_parsed']

    if pd.notna(row['EDBS_IDATE_parsed']) and row['EDBS_IDATE_parsed'] < ref_date:
        age_days = (ref_date - row['EDBS_IDATE_parsed']).days
        return age_days, 'EDBS', row['EDBS_IDATE_parsed']

    return None, 'MISSING', None

def calculate_age_edbs_priority(row):
    """
    Calculate age with EDBS_IDATE as PRIMARY (physical installation date)
    Priority: EDBS_IDATE → TESIS_TARIHI → Work Order
    """
    ref_date = REFERENCE_DATE

    if pd.notna(row['EDBS_IDATE_parsed']) and row['EDBS_IDATE_parsed'] < ref_date:
        age_days = (ref_date - row['EDBS_IDATE_parsed']).days
        return age_days, 'EDBS', row['EDBS_IDATE_parsed']

    if pd.notna(row['TESIS_TARIHI_parsed']) and row['TESIS_TARIHI_parsed'] < ref_date:
        age_days = (ref_date - row['TESIS_TARIHI_parsed']).days
        return age_days, 'TESIS', row['TESIS_TARIHI_parsed']

    return None, 'MISSING', None

print("\nCalculating dual age features (TESIS-primary + EDBS-primary)...")

# Calculate TESIS-primary age (commissioning age)
results_tesis = df.apply(calculate_age_tesis_priority, axis=1, result_type='expand')
results_tesis.columns = ['Ekipman_Yaşı_Gün_TESIS', 'Yaş_Kaynak_TESIS', 'Kurulum_Tarihi_TESIS']
df[['Ekipman_Yaşı_Gün_TESIS', 'Yaş_Kaynak_TESIS', 'Kurulum_Tarihi_TESIS']] = results_tesis
df['Ekipman_Yaşı_Yıl_TESIS'] = df['Ekipman_Yaşı_Gün_TESIS'] / 365.25

# Calculate EDBS-primary age (installation age)
results_edbs = df.apply(calculate_age_edbs_priority, axis=1, result_type='expand')
results_edbs.columns = ['Ekipman_Yaşı_Gün_EDBS', 'Yaş_Kaynak_EDBS', 'Kurulum_Tarihi_EDBS']
df[['Ekipman_Yaşı_Gün_EDBS', 'Yaş_Kaynak_EDBS', 'Kurulum_Tarihi_EDBS']] = results_edbs
df['Ekipman_Yaşı_Yıl_EDBS'] = df['Ekipman_Yaşı_Gün_EDBS'] / 365.25

# Create primary age columns (default to TESIS)
df['Ekipman_Yaşı_Gün'] = df['Ekipman_Yaşı_Gün_TESIS']
df['Ekipman_Yaşı_Yıl'] = df['Ekipman_Yaşı_Yıl_TESIS']
df['Yaş_Kaynak'] = df['Yaş_Kaynak_TESIS']
df['Ekipman_Kurulum_Tarihi'] = df['Kurulum_Tarihi_TESIS']

# Compact statistics
source_counts = df['Yaş_Kaynak_TESIS'].value_counts()
valid_ages = df[df['Yaş_Kaynak_TESIS'] != 'MISSING']['Ekipman_Yaşı_Yıl_TESIS']

print(f"\n✓ Age Calculation Complete:")
print(f"  Sources: ", end="")
print(" | ".join([f"{src}:{cnt:,}({cnt/len(df)*100:.1f}%)" for src, cnt in source_counts.items()]))
if len(valid_ages) > 0:
    print(f"  Range: {valid_ages.min():.1f}-{valid_ages.max():.1f}y, Mean={valid_ages.mean():.1f}y, Median={valid_ages.median():.1f}y")

# Age distribution summary (compact)
valid_ages = df[df['Yaş_Kaynak'] != 'MISSING']['Ekipman_Yaşı_Yıl']
if len(valid_ages) > 0:
    age_bins = [0, 10, 20, 30, 50, 200]
    age_labels = ['0-10y', '10-20y', '20-30y', '30-50y', '50+y']
    age_dist = pd.cut(valid_ages, bins=age_bins, labels=age_labels).value_counts().sort_index()

    print(f"  Age Distribution: ", end="")
    print(" | ".join([f"{lbl}:{cnt}({cnt/len(valid_ages)*100:.0f}%)" for lbl, cnt in age_dist.items()]))

    # Warnings (compact)
    warnings = []
    if (valid_ages > 75).sum() > 0:
        warnings.append(f"{(valid_ages > 75).sum()} equipment >75y")
    if valid_ages.median() < 1:
        warnings.append(f"Median age {valid_ages.median():.1f}y (low!)")
    if warnings:
        print(f"  ⚠️  " + ", ".join(warnings))

# ============================================================================
# STEP 3B: OPTIONAL FIRST WORK ORDER FALLBACK
# ============================================================================
if USE_FIRST_WORKORDER_FALLBACK:
    print("\n" + "="*100)
    print("STEP 3B: FILLING MISSING AGES WITH FIRST WORK ORDER (VECTORIZED)")
    print("="*100)

    # Check missing for BOTH age types
    missing_mask_tesis = df['Yaş_Kaynak_TESIS'] == 'MISSING'
    missing_mask_edbs = df['Yaş_Kaynak_EDBS'] == 'MISSING'
    missing_count = missing_mask_edbs.sum()  # Use EDBS as reference

    if missing_count > 0 and 'Oluşturulma_Tarihi' in df.columns:
        print(f"\n  Equipment with MISSING age (EDBS-primary): {missing_count:,} ({missing_count/len(df)*100:.1f}%)")
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

            # Fill TESIS-primary missing ages
            fill_mask_tesis = (
                missing_mask_tesis &
                df['_first_wo'].notna() &
                (age_from_wo > 0)
            )
            df.loc[fill_mask_tesis, 'Ekipman_Yaşı_Gün_TESIS'] = age_from_wo[fill_mask_tesis]
            df.loc[fill_mask_tesis, 'Ekipman_Yaşı_Yıl_TESIS'] = age_from_wo[fill_mask_tesis] / 365.25
            df.loc[fill_mask_tesis, 'Yaş_Kaynak_TESIS'] = 'WORKORDER'
            df.loc[fill_mask_tesis, 'Kurulum_Tarihi_TESIS'] = df.loc[fill_mask_tesis, '_first_wo']

            # Fill EDBS-primary missing ages
            fill_mask_edbs = (
                missing_mask_edbs &
                df['_first_wo'].notna() &
                (age_from_wo > 0)
            )
            df.loc[fill_mask_edbs, 'Ekipman_Yaşı_Gün_EDBS'] = age_from_wo[fill_mask_edbs]
            df.loc[fill_mask_edbs, 'Ekipman_Yaşı_Yıl_EDBS'] = age_from_wo[fill_mask_edbs] / 365.25
            df.loc[fill_mask_edbs, 'Yaş_Kaynak_EDBS'] = 'WORKORDER'
            df.loc[fill_mask_edbs, 'Kurulum_Tarihi_EDBS'] = df.loc[fill_mask_edbs, '_first_wo']

            # Update default age columns (use EDBS as default)
            df.loc[fill_mask_edbs, 'Ekipman_Yaşı_Gün'] = age_from_wo[fill_mask_edbs]
            df.loc[fill_mask_edbs, 'Ekipman_Yaşı_Yıl'] = age_from_wo[fill_mask_edbs] / 365.25
            df.loc[fill_mask_edbs, 'Yaş_Kaynak'] = 'FIRST_WORKORDER_PROXY'
            df.loc[fill_mask_edbs, 'Ekipman_Kurulum_Tarihi'] = df.loc[fill_mask_edbs, '_first_wo']

            # Cleanup temporary column
            df.drop(columns=['_first_wo'], inplace=True)

            filled_count_edbs = fill_mask_edbs.sum()
            filled_count_tesis = fill_mask_tesis.sum()
            remaining_missing = (df['Yaş_Kaynak_EDBS'] == 'MISSING').sum()

            print(f"  ✓ Filled (EDBS-primary): {filled_count_edbs:,} using first work order proxy")
            print(f"  ✓ Filled (TESIS-primary): {filled_count_tesis:,} using first work order proxy")
            print(f"  ✓ Remaining MISSING: {remaining_missing:,} ({remaining_missing/len(df)*100:.1f}%)")

            # Final age statistics
            if filled_count_edbs > 0 or filled_count_tesis > 0:
                print(f"\n  Updated Age Source Distribution (EDBS-primary):")
                for source, count in df['Yaş_Kaynak_EDBS'].value_counts().items():
                    pct = count / len(df) * 100
                    print(f"    {source:15s}: {count:6,} ({pct:5.1f}%)")
        else:
            print(f"  ⚠️  Equipment ID column not found - cannot use first work order fallback")
    elif missing_count == 0:
        print(f"\n  ✓ No missing ages - first work order fallback not needed")
    else:
        print(f"\n  ⚠️  Work order creation date not available - cannot use fallback")

# STEP 4 & 5: Temporal Features + Failure Periods
print("\n" + "="*80)
print("STEP 4/5: TEMPORAL FEATURES & FAILURE PERIODS")
print("="*80)

df['Fault_Month'] = df['started at'].dt.month
df['Summer_Peak_Flag'] = df['Fault_Month'].isin([6, 7, 8, 9]).astype(int)
df['Winter_Peak_Flag'] = df['Fault_Month'].isin([12, 1, 2]).astype(int)
df['Time_To_Repair_Hours'] = (df['ended at'] - df['started at']).dt.total_seconds() / 3600

reference_date = df['started at'].max()
cutoff_3m = reference_date - pd.Timedelta(days=90)
cutoff_6m = reference_date - pd.Timedelta(days=180)
cutoff_12m = reference_date - pd.Timedelta(days=365)

df['Fault_Last_3M'] = (df['started at'] >= cutoff_3m).astype(int)
df['Fault_Last_6M'] = (df['started at'] >= cutoff_6m).astype(int)
df['Fault_Last_12M'] = (df['started at'] >= cutoff_12m).astype(int)

print(f"\n✓ Temporal: Summer={df['Summer_Peak_Flag'].sum():,}, Winter={df['Winter_Peak_Flag'].sum():,}, Avg repair={df['Time_To_Repair_Hours'].mean():.1f}h")
print(f"✓ Periods (ref={reference_date.strftime('%Y-%m-%d')}): 3M={df['Fault_Last_3M'].sum():,}, 6M={df['Fault_Last_6M'].sum():,}, 12M={df['Fault_Last_12M'].sum():,}")

# STEP 6: Equipment Identification
print("\n" + "="*80)
print("STEP 6: EQUIPMENT IDENTIFICATION (SIMPLIFIED)")
print("="*80)

# Create unified equipment ID with fallback logic
def get_equipment_id(row):
    """
    Get equipment ID with smart fallback (SIMPLIFIED - cbs_id full coverage)
    Priority: cbs_id → Ekipman ID → Generate unique ID

    Note: Generates unique ID for equipment without proper IDs to prevent grouping
    """
    if pd.notna(row.get('cbs_id')):
        return row['cbs_id']
    elif pd.notna(row.get('Ekipman ID')):
        return row['Ekipman ID']
    else:
        # Generate unique ID to prevent grouping all missing IDs together
        return f"UNKNOWN_{row.name}"

df['Equipment_ID_Primary'] = df.apply(get_equipment_id, axis=1)

# Statistics
primary_coverage = df['Equipment_ID_Primary'].notna().sum()
unique_equipment = df['Equipment_ID_Primary'].nunique()

# Count by source
cbs_count = df['cbs_id'].notna().sum()
ekipman_count = df[df['cbs_id'].isna() & df['Ekipman ID'].notna()].shape[0]
unknown_count = df['Equipment_ID_Primary'].astype(str).str.startswith('UNKNOWN_', na=False).sum()

print(f"\n✓ ID Strategy: cbs_id({cbs_count:,}) → Ekipman ID({ekipman_count:,}) → Generated({unknown_count:,})")
print(f"  Total: {unique_equipment:,} unique equipment from {len(df):,} faults (avg {len(df)/unique_equipment:.1f} faults/equip)")

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

# Sort by TESIS Age_Source to prioritize during aggregation (TESIS = commissioning age)
source_priority_tesis = {'TESIS': 0, 'EDBS': 1, 'WORKORDER': 2, 'MISSING': 3}
df['_source_priority'] = df['Yaş_Kaynak_TESIS'].map(source_priority_tesis).fillna(99)
df = df.sort_values('_source_priority')
df = df.drop(columns=['_source_priority'])

print("\n  ✓ Sorted data to prioritize TESIS_TARIHI as primary age source during aggregation")

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

    # DUAL Age data (default = TESIS-primary commissioning age)
    'Ekipman_Kurulum_Tarihi': 'first',
    'Ekipman_Yaşı_Gün': 'first',
    'Ekipman_Yaşı_Yıl': 'first',
    'Age_Source': 'first',

    # TESIS-primary age (commissioning age - DEFAULT for modeling)
    'Kurulum_Tarihi_TESIS': 'first',
    'Ekipman_Yaşı_Gün_TESIS': 'first',
    'Ekipman_Yaşı_Yıl_TESIS': 'first',
    'Yaş_Kaynak_TESIS': 'first',

    # EDBS-primary age (EdaBİS database entry age ~2017+, NOT physical installation)
    'Kurulum_Tarihi_EDBS': 'first',
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
print(f"   • Equipment ID Strategy: cbs_id → Ekipman ID → Generated unique ID (prevents grouping)")
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

print(f"\n✅ ENHANCEMENTS IN v3.1:")
print(f"   ✨ Smart date validation (rejects Excel NULL + suspicious recent dates)")
unknown_equip_count = equipment_df['Ekipman_ID'].astype(str).str.startswith('UNKNOWN_', na=False).sum()
print(f"   ✨ Simplified Equipment ID (prevents grouping bug for {unknown_equip_count} equipment)")
print(f"   ✨ Day-precision age calculation (365.25 days/year)")
print(f"   ✨ Installation date preserved (Ekipman_Kurulum_Tarihi)")
print(f"   ✨ Age in days available (Ekipman_Yaşı_Gün)")
if USE_FIRST_WORKORDER_FALLBACK:
    wo_count = (equipment_df['Age_Source'] == 'FIRST_WORKORDER_PROXY').sum()
    print(f"   ✨ First work order fallback ({wo_count} equipment)")
print(f"   ✨ Vectorized operations for better performance")

print(f"\n🚀 READY FOR NEXT PHASE:")
print(f"   → Run: 03_feature_engineering.py")
print(f"   → Create advanced features (age ratios, reliability scores, etc.)")
print("="*100)
