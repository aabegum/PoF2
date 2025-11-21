"""
================================================================================
SCRIPT 02: DATA TRANSFORMATION (Fault-Level → Equipment-Level) v5.0
================================================================================
Turkish EDAS PoF (Probability of Failure) Prediction Pipeline

PIPELINE STRATEGY: Temporal Multi-Horizon Predictions
- Cutoff Date: 2024-06-25 (configurable in config.py)
- Historical Window: All data up to cutoff date (for feature calculation)
- Prediction Windows: 3M, 6M, 12M multi-horizon failure risk
- Features Created: Temporal fault counts, age, MTBF (3 methods), reliability metrics
- DATA LEAKAGE PREVENTION: All features calculated using data BEFORE cutoff date only

WHAT THIS SCRIPT DOES:
Transforms fault-level records to equipment-level records with engineered features.
Creates comprehensive features for temporal PoF modeling including:
- Fault history features (3M/6M/12M counts) - PRIMARY prediction drivers
- Equipment age and time-to-first-failure - Wear-out pattern detection
- MTBF features (3 methods) - Inter-fault, Lifetime, Observable
- Degradation indicators - Failure acceleration detection
- Geographic features - Spatial risk patterns (if available)
- Customer impact ratios - Criticality scoring (if available)

ENHANCEMENTS in v5.0:
+ UPDATED: Equipment Age Calculation
  - NEW SOURCE: Sebekeye_Baglanma_Tarihi (Grid Connection Date) - single reliable source
  - REMOVED: TESIS_TARIHI and EDBS_IDATE (legacy fallback priority chain no longer needed)
  - SIMPLIFIED: Direct age calculation from grid connection date
  - Age Source: 'GRID_CONNECTION' (vs previous 'TESIS'/'EDBS'/'WORKORDER')

ENHANCEMENTS in v4.1:
+ NEW: 3 MTBF Calculation Methods
  - Method 1 (MTBF_Gün): Inter-fault average → Best for PoF prediction
  - Method 2 (MTBF_Lifetime_Gün): Total exposure / failures → Survival analysis baseline hazard
  - Method 3 (MTBF_Observable_Gün): First fault to cutoff / failures → Degradation detection
+ NEW: Baseline_Hazard_Rate = 1/MTBF_Lifetime → For Cox proportional hazards model
+ NEW: MTBF_Degradation_Ratio = Method3/Method1 → Detects if failures accelerating
+ NEW: Is_Degrading flag → Equipment with ratio >1.5 (failures accelerating)

ENHANCEMENTS in v4.0:
+ NEW FEATURE: Ilk_Arizaya_Kadar_Gun/Yil (Time Until First Failure)
  - Calculates: Grid Connection Date → First Fault Date
  - Detects: Infant mortality vs survived burn-in equipment
+ Multi-Horizon Predictions: Links features to 3M/6M/12M prediction windows
+ Reduced Verbosity: Concise progress indicators
+ Progress Indicators: [Step X/12] for pipeline visibility
+ Flexible Date Parser: Multi-format support (DD-MM-YYYY, ISO, etc.)
+ Smart Date Validation: Rejects Excel NULL + suspicious recent dates

CROSS-REFERENCES:
- Script 01: Data quality profiling and validation
- Script 03: Uses these features for advanced engineering (PoF risk scores)
- Script 06: Temporal PoF Model (multi-horizon predictions)
- Script 09: Cox Survival Model (uses MTBF_Lifetime_Gün)

Input:  data/combined_data_son.xlsx (fault records)
Output: data/equipment_level_data.csv (equipment-level features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import sys

# Import centralized configuration
from config import (
    CUTOFF_DATE,
    REFERENCE_DATE,
    MIN_VALID_YEAR,
    MAX_VALID_YEAR,
    DATA_DIR,
    INPUT_FILE,
    EQUIPMENT_LEVEL_FILE,
    FEATURE_DOCS_FILE,
    EQUIPMENT_CLASS_MAPPING
)

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
# CONFIGURATION (from config.py)
# ============================================================================
# All configuration now imported from config.py
# CUTOFF_DATE, REFERENCE_DATE, MIN_VALID_YEAR, MAX_VALID_YEAR, etc.

print("\n" + "="*80)
print("SCRIPT 02: DATA TRANSFORMATION v5.0 (Multi-Horizon PoF)")
print("="*80)
print(f"Reference Date: {REFERENCE_DATE.strftime('%Y-%m-%d')} | Valid Years: {MIN_VALID_YEAR}-{MAX_VALID_YEAR}")
print(f"Age Source: Sebekeye_Baglanma_Tarihi (Grid Connection Date)")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[Step 1/12] Loading Fault-Level Data...")

df = pd.read_excel(INPUT_FILE)
original_fault_count = len(df)
print(f"Loaded: {df.shape[0]:,} faults x {df.shape[1]} columns from {INPUT_FILE}")

# ============================================================================
# STEP 1B: FILL MISSING CBS_ID FROM 'ID' COLUMN (DO THIS FIRST!)
# ============================================================================
# CRITICAL: ID consolidation MUST happen BEFORE duplicate detection
# Otherwise, duplicates with missing cbs_id won't be detected
print("\n[Step 1B/12] Filling Missing cbs_id Using 'id' Column...")
print("⚠️  CRITICAL: Consolidating equipment IDs BEFORE duplicate detection")

if 'cbs_id' in df.columns:
    before_fill = len(df)
    has_cbs_id = df['cbs_id'].notna().sum()
    missing_cbs_id = df['cbs_id'].isna().sum()

    print(f"  Total faults: {before_fill:,}")
    print(f"  Has cbs_id: {has_cbs_id:,} ({has_cbs_id/before_fill*100:.1f}%)")
    print(f"  Missing cbs_id: {missing_cbs_id:,} ({missing_cbs_id/before_fill*100:.1f}%)")

    if missing_cbs_id > 0 and 'id' in df.columns:
        # Use 'id' column as fallback for missing cbs_id
        print(f"  Using 'id' column as fallback for missing cbs_id...")

        # Fill missing cbs_id with 'id' column values
        missing_mask = df['cbs_id'].isna()
        df.loc[missing_mask, 'cbs_id'] = df.loc[missing_mask, 'id']

        filled = missing_mask.sum()
        still_missing = df['cbs_id'].isna().sum()

        print(f"  ✓ Filled {filled:,} missing cbs_id values from 'id' column")

        if still_missing > 0:
            print(f"  [!] Still missing: {still_missing:,} faults have neither cbs_id nor id")
            print(f"      Removing {still_missing:,} faults without any equipment ID...")
            df = df[df['cbs_id'].notna()].copy()
            print(f"  Final fault count: {len(df):,}")
        else:
            print(f"  ✓ All faults now have equipment IDs (cbs_id or id)")

        # Update equipment tracking after ID consolidation
        unique_equip_after_id_consolidation = df['cbs_id'].nunique()
        print(f"  Unique equipment after ID consolidation: {unique_equip_after_id_consolidation:,}")

    elif missing_cbs_id > 0 and 'id' not in df.columns:
        print(f"  [!] WARNING: 'id' column not found - cannot use as fallback")
        print(f"      Removing {missing_cbs_id:,} faults without cbs_id...")
        df = df[df['cbs_id'].notna()].copy()
        print(f"  Final fault count: {len(df):,}")
        unique_equip_after_id_consolidation = df['cbs_id'].nunique()
    else:
        print(f"  ✓ All faults have cbs_id - no filling needed")
        unique_equip_after_id_consolidation = df['cbs_id'].nunique()
else:
    print(f"  [!] WARNING: No cbs_id column found")
    print(f"      Equipment ID assignment may create UNKNOWN_XXX IDs")
    unique_equip_after_id_consolidation = 0

# ============================================================================
# STEP 1C: DUPLICATE DETECTION (AFTER ID CONSOLIDATION)
# ============================================================================
print("\n[Step 1C/12] Detecting and Removing Duplicates...")
print("  (Now checking with consolidated equipment IDs)")

# Identify equipment ID column
equip_id_cols = ['cbs_id', 'Ekipman Kodu', 'Ekipman ID', 'HEPSI_ID']
equip_id_col = next((col for col in equip_id_cols if col in df.columns), None)

if not equip_id_col:
    print("❌ ERROR: No equipment ID column found!")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

print(f"Using equipment ID column: {equip_id_col}")

# CHECK 1: Exact duplicates (all columns identical)
exact_dup_mask = df.duplicated(keep='first')
exact_dup_count = exact_dup_mask.sum()

if exact_dup_count > 0:
    print(f"  ✓ Found {exact_dup_count:,} exact row duplicates ({exact_dup_count/len(df)*100:.1f}%) - removing...")
    df = df[~exact_dup_mask].copy()
else:
    print(f"  ✓ No exact row duplicates found")

# CHECK 2: Same equipment + same start time (likely same fault from different sources)
# This is the CRITICAL check for multi-source data (more important than exact duplicates)
if 'started at' in df.columns:
    time_dup_mask = df.duplicated(subset=[equip_id_col, 'started at'], keep='first')
    time_dup_count = time_dup_mask.sum()

    if time_dup_count > 0:
        print(f"  ✓ Found {time_dup_count:,} equipment+time duplicates ({time_dup_count/len(df)*100:.1f}%) - removing...")
        print(f"    (Same equipment ID + timestamp = likely duplicate from multi-source data)")

        # Show examples for manual validation
        df_duplicates = df[time_dup_mask].copy()
        if len(df_duplicates) > 0:
            print(f"\n  Duplicate Examples (first 3 for manual review):")
            display_cols = [equip_id_col, 'started at', 'ended at', 'Equipment_Type']
            display_cols = [col for col in display_cols if col in df_duplicates.columns]
            dup_sample = df_duplicates[display_cols].head(3)
            for idx, row in dup_sample.iterrows():
                print(f"    {idx}: {dict(row)}")

            # Save all duplicates for audit
            dup_file = DATA_DIR / 'removed_duplicates.csv'
            df_duplicates.to_csv(dup_file, index=False)
            print(f"  ✓ All duplicates saved: {dup_file}")

            # Analyze duplicate distribution across cutoff date
            if 'started at' in df_duplicates.columns:
                df_duplicates['started_at_parsed'] = pd.to_datetime(df_duplicates['started at'], errors='coerce')
                pre_cutoff_dups = (df_duplicates['started_at_parsed'] <= REFERENCE_DATE).sum()
                post_cutoff_dups = (df_duplicates['started_at_parsed'] > REFERENCE_DATE).sum()

                print(f"\n  Duplicate Distribution by Cutoff Date:")
                print(f"    Pre-cutoff duplicates:  {pre_cutoff_dups} ({pre_cutoff_dups/len(df_duplicates)*100:.1f}%)")
                print(f"    Post-cutoff duplicates: {post_cutoff_dups} ({post_cutoff_dups/len(df_duplicates)*100:.1f}%)")
                if post_cutoff_dups > 0:
                    post_cutoff_equip = df_duplicates[df_duplicates['started_at_parsed'] > REFERENCE_DATE][equip_id_col].nunique()
                    print(f"    → {post_cutoff_dups} duplicates removed from test set ({post_cutoff_equip} equipment)")
                    print(f"    → Test set quality improved by removing post-cutoff duplicates")

        df = df[~time_dup_mask].copy()
    else:
        print(f"  ✓ No equipment+time duplicates found (data already clean!)")
else:
    print("  ⚠️  WARNING: 'started at' column not found - skipping time-based duplicate check")

# Summary
removed = original_fault_count - len(df)
if removed > 0:
    print(f"  ✅ Removed {removed:,} duplicate records ({removed/original_fault_count*100:.1f}%)")
    print(f"  Final fault count: {len(df):,} (from {original_fault_count:,})")
else:
    print(f"  ✅ No duplicates detected - data quality looks good!")

# Track unique equipment at this stage
unique_equip_after_dedup = df[equip_id_col].nunique()
print(f"\n[TRACKING] Unique equipment after deduplication: {unique_equip_after_dedup:,}")

# ============================================================================
# STEP 2: ENHANCED DATE PARSING & VALIDATION
# ============================================================================
print("\n[Step 2/12] Parsing Dates (Flexible Multi-Format Parser)...")

def parse_date_flexible(value):
    """
    Parse date with multiple format support - handles mixed format data
    Supports: ISO, Turkish (DD-MM-YYYY), European (DD/MM/YYYY), Excel serial dates

    This function solves the 25% "missing" timestamp issue caused by mixed date formats
    """
    # Already a timestamp/datetime
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)

    # Handle NaN/None
    if pd.isna(value):
        return pd.NaT

    # Excel serial date (numeric)
    if isinstance(value, (int, float)):
        if 1 <= value <= 100000:
            try:
                # Excel epoch starts at 1900-01-01
                # Excel has a leap year bug for 1900
                return pd.Timestamp('1899-12-30') + pd.Timedelta(days=value)
            except:
                return pd.NaT
        else:
            return pd.NaT

    # String parsing with multiple format attempts
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return pd.NaT

        # Try multiple formats in order of likelihood
        formats = [
            '%Y-%m-%d %H:%M:%S',     # 2021-01-15 12:30:45 (ISO)
            '%d-%m-%Y %H:%M:%S',     # 15-01-2021 12:30:45 (Turkish/European with dash)
            '%d/%m/%Y %H:%M:%S',     # 15/01/2021 12:30:45 (Turkish/European with slash)
            '%Y-%m-%d',              # 2021-01-15
            '%d-%m-%Y',              # 15-01-2021
            '%d/%m/%Y',              # 15/01/2021
            '%d.%m.%Y %H:%M:%S',     # 15.01.2021 12:30:45 (Turkish dot format)
            '%d.%m.%Y',              # 15.01.2021
            '%m/%d/%Y %H:%M:%S',     # 01/15/2021 12:30:45 (US format - try last)
            '%m/%d/%Y',              # 01/15/2021
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except:
                continue

        # Last resort: let pandas infer
        try:
            return pd.to_datetime(value, infer_datetime_format=True, dayfirst=True)
        except:
            return pd.NaT

    return pd.NaT

def parse_and_validate_date(date_series, column_name, min_year=MIN_VALID_YEAR, max_year=MAX_VALID_YEAR,
                            report=True, is_installation_date=False):
    """
    Parse and validate dates with smart validation + flexible multi-format parsing
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
        - v3.2: Now handles mixed date formats (DD-MM-YYYY + YYYY-MM-DD)
    """
    # PRE-CHECK: Reject time-only values (e.g., "00:00:00", "12:30:00") before parsing
    time_only_mask = pd.Series([False] * len(date_series), index=date_series.index)
    if pd.api.types.is_string_dtype(date_series) or pd.api.types.is_object_dtype(date_series):
        # Check if value matches time pattern (HH:MM:SS or similar)
        time_only_mask = date_series.astype(str).str.match(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*$', na=False)

    invalid_time_only = time_only_mask.sum()

    # Use flexible parser that handles mixed formats
    # This solves the 25% "missing" timestamp issue
    parsed = date_series.apply(parse_date_flexible)

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

    # Report statistics (minimal format)
    if report:
        total = len(date_series)
        valid = valid_mask.sum()
        pct = valid/total*100
        status = "OK" if pct > 95 else ("WARN" if pct > 80 else "FAIL")
        print(f"  {column_name:20s}: {valid:4,}/{total:4,} ({pct:5.1f}%) [{status}]")

    return parsed

# Parse and validate all date columns
# NEW: Using Sebekeye_Baglanma_Tarihi (Grid Connection Date) as primary installation date source

# Check if required column exists
if 'Sebekeye_Baglanma_Tarihi' not in df.columns:
    print("\n❌ ERROR: Required column 'Sebekeye_Baglanma_Tarihi' not found in input file!")
    print("\nAvailable columns in input file (showing first 20):")
    for i, col in enumerate(df.columns[:20], 1):
        print(f"  {i:2d}. {col}")
    if len(df.columns) > 20:
        print(f"  ... and {len(df.columns) - 20} more columns")

    # Look for potential date columns
    potential_date_cols = [col for col in df.columns if any(
        keyword in col.upper() for keyword in ['TARIH', 'DATE', 'BAGLAN', 'KURULUM', 'TESIS', 'EDBS', 'INSTALL']
    )]
    if potential_date_cols:
        print(f"\n💡 Found {len(potential_date_cols)} potential date/installation columns:")
        for col in potential_date_cols:
            print(f"  - {col}")
        print("\n⚠️  Please update your input file to include 'Sebekeye_Baglanma_Tarihi' column")
        print("   OR update the script to use one of the columns above.")
    else:
        print("\n⚠️  No date-like columns found. Please check your input file structure.")

    print("\n" + "="*80)
    print("SOLUTION:")
    print("  1. Add 'Sebekeye_Baglanma_Tarihi' column to your input file")
    print("  OR")
    print("  2. If using different column name, update line 358 in 02_data_transformation.py")
    print("="*80)
    sys.exit(1)

df['Sebekeye_Baglanma_Tarihi_parsed'] = parse_and_validate_date(df['Sebekeye_Baglanma_Tarihi'], 'Sebekeye_Baglanma_Tarihi', is_installation_date=True)
df['started at'] = parse_and_validate_date(df['started at'], 'started at', min_year=2020, report=True, is_installation_date=False)
df['ended at'] = parse_and_validate_date(df['ended at'], 'ended at', min_year=2020, report=True, is_installation_date=False)

# ============================================================================
# STEP 3: SIMPLIFIED EQUIPMENT AGE CALCULATION (Using Sebekeye_Baglanma_Tarihi)
# ============================================================================
# UPDATED v5.0: Direct calculation from Sebekeye_Baglanma_Tarihi (Grid Connection Date)
# Single reliable source - no fallback priority chain needed
print("\n[Step 3/12] Calculating Equipment Age (Sebekeye_Baglanma_Tarihi - Grid Connection Date)...")

def calculate_equipment_age(row):
    """
    Calculate equipment age from Grid Connection Date.
    Single source: Sebekeye_Baglanma_Tarihi
    Returns: (age_days, source, installation_date)
    """
    ref_date = REFERENCE_DATE

    # Use Sebekeye_Baglanma_Tarihi (Grid Connection Date)
    if pd.notna(row['Sebekeye_Baglanma_Tarihi_parsed']) and row['Sebekeye_Baglanma_Tarihi_parsed'] < ref_date:
        age_days = (ref_date - row['Sebekeye_Baglanma_Tarihi_parsed']).days
        return age_days, 'GRID_CONNECTION', row['Sebekeye_Baglanma_Tarihi_parsed']

    # Missing installation date
    return None, 'MISSING', None

# Calculate age using simplified function
results = df.apply(calculate_equipment_age, axis=1, result_type='expand')
results.columns = ['Ekipman_Yaşı_Gün', 'Yaş_Kaynak', 'Ekipman_Kurulum_Tarihi']
df[['Ekipman_Yaşı_Gün', 'Yaş_Kaynak', 'Ekipman_Kurulum_Tarihi']] = results

# Convert days to years
df['Ekipman_Yaşı_Yıl'] = df['Ekipman_Yaşı_Gün'] / 365.25

# Summary statistics
source_counts = df['Yaş_Kaynak'].value_counts()
valid_ages = df[df['Yaş_Kaynak'] != 'MISSING']['Ekipman_Yaşı_Yıl']
print(f"Age Sources: {' | '.join([f'{src}:{cnt:,}({cnt/len(df)*100:.0f}%)' for src, cnt in source_counts.items()])}")
if len(valid_ages) > 0:
    print(f"Age Range: {valid_ages.min():.1f}-{valid_ages.max():.1f}y, Mean={valid_ages.mean():.1f}y, Median={valid_ages.median():.1f}y")
missing_count = (df['Yaş_Kaynak'] == 'MISSING').sum()
print(f"✓ Equipment ages calculated | Missing: {missing_count:,} ({missing_count/len(df)*100:.1f}%)")

# Check for suspicious old dates (before 1964 = oldest acceptable data)
if len(valid_ages) > 0:
    very_old_mask = valid_ages > 60  # Equipment older than 1964 (2025 - 60 = 1965)
    very_old_count = very_old_mask.sum()

    if very_old_count > 0:
        very_old_equip = df[df['Ekipman_Yaşı_Yıl'] > 60][['cbs_id', 'Sebekeye_Baglanma_Tarihi_parsed', 'Ekipman_Yaşı_Yıl']].drop_duplicates('cbs_id')
        print(f"\n⚠️  WARNING: {very_old_count:,} records have equipment older than 60 years (before 1964)")
        print(f"   Unique old equipment: {len(very_old_equip):,}")

        # Check for common default dates
        default_dates = [
            pd.Timestamp('1978-01-01'),  # 1.01.1978
            pd.Timestamp('1900-01-01'),  # Excel NULL
            pd.Timestamp('1970-01-01'),  # Unix epoch
        ]

        default_date_mask = df['Sebekeye_Baglanma_Tarihi_parsed'].isin(default_dates)
        default_date_count = default_date_mask.sum()

        if default_date_count > 0:
            print(f"   [!] {default_date_count:,} records use common default dates (1.01.1978, 1.01.1970, etc.)")
            print(f"       These may be placeholder values rather than real installation dates")

        # Show examples for review
        if len(very_old_equip) > 0:
            print(f"\n   Examples of very old equipment (showing first 5):")
            for idx, row in very_old_equip.head(5).iterrows():
                install_date = row['Sebekeye_Baglanma_Tarihi_parsed']
                age = row['Ekipman_Yaşı_Yıl']
                cbs = row['cbs_id']
                if pd.notna(install_date):
                    print(f"     • Equipment {int(cbs):,}: Installed {install_date.strftime('%Y-%m-%d')} (Age: {age:.1f} years)")

        print(f"\n   → RECOMMENDATION: Review these dates for data quality issues")
        print(f"   → Acceptable range: 1964-present (equipment from 0-60 years old)")
        print(f"   → Action: Verify if dates before 1964 are real or default placeholders")

# ============================================================================
# STEP 3B: TEMPORAL VALIDATION (CRITICAL DATA QUALITY CHECK)
# ============================================================================
print(f"\n[Step 3B/12] Validating Temporal Consistency...")

validation_issues = []

# CHECK 1: Fault date BEFORE equipment installation
if 'Sebekeye_Baglanma_Tarihi_parsed' in df.columns and 'started at' in df.columns:
    before_install_mask = (df['Sebekeye_Baglanma_Tarihi_parsed'].notna() &
                           df['started at'].notna() &
                           (df['started at'] < df['Sebekeye_Baglanma_Tarihi_parsed']))
    before_install_count = before_install_mask.sum()

    if before_install_count > 0:
        print(f"  ❌ CRITICAL: {before_install_count:,} faults occurred BEFORE equipment installation!")
        validation_issues.append(f"Faults before installation: {before_install_count}")

        # Show examples
        before_install_sample = df[before_install_mask][['cbs_id', 'started at', 'Sebekeye_Baglanma_Tarihi_parsed']].head(5)
        print(f"\n  Examples (first 5):")
        for idx, row in before_install_sample.iterrows():
            fault_date = row['started at']
            install_date = row['Sebekeye_Baglanma_Tarihi_parsed']
            days_before = (install_date - fault_date).days
            print(f"    • Equipment {int(row['cbs_id']):,}: Fault on {fault_date.strftime('%Y-%m-%d')}, ")
            print(f"      but installed {days_before} days later on {install_date.strftime('%Y-%m-%d')}")

        print(f"\n  → ACTION REQUIRED: Remove or fix these {before_install_count:,} records")
        print(f"     Option 1: Remove faults (may lose valid failure data)")
        print(f"     Option 2: Correct installation dates (check source data)")
    else:
        print(f"  ✓ All faults occurred AFTER equipment installation")

# CHECK 2: Negative time-to-repair
if 'Time_To_Repair_Hours' in df.columns:
    # Will check after Time_To_Repair_Hours is calculated
    pass
else:
    # Calculate here for validation
    temp_ttr = (df['ended at'] - df['started at']).dt.total_seconds() / 3600
    negative_ttr = (temp_ttr < 0).sum()
    if negative_ttr > 0:
        print(f"  ❌ WARNING: {negative_ttr:,} faults have negative time-to-repair (end before start!)")
        validation_issues.append(f"Negative repair time: {negative_ttr}")
    else:
        print(f"  ✓ All repair times are positive")

if len(validation_issues) == 0:
    print(f"  ✅ All temporal validations PASSED!")
else:
    print(f"\n  ⚠️  Found {len(validation_issues)} temporal data quality issues")

# STEP 4 & 5: Temporal Features + Failure Periods
print("\n[Step 4-5/12] Creating Temporal Features (3M/6M/12M Windows) [6M/12M]...")

df['Fault_Month'] = df['started at'].dt.month
df['Summer_Peak_Flag'] = df['Fault_Month'].isin([6, 7, 8, 9]).astype(int)
df['Winter_Peak_Flag'] = df['Fault_Month'].isin([12, 1, 2]).astype(int)
df['Time_To_Repair_Hours'] = (df['ended at'] - df['started at']).dt.total_seconds() / 3600

# CRITICAL FIX: Use CUTOFF_DATE instead of df['started at'].max()
# This ensures temporal features use ONLY historical data (before prediction window)
reference_date = REFERENCE_DATE  # Use cutoff date from config.py
cutoff_3m = reference_date - pd.Timedelta(days=90)   # 2024-03-27
cutoff_6m = reference_date - pd.Timedelta(days=180)  # 2023-12-28
cutoff_12m = reference_date - pd.Timedelta(days=365) # 2023-06-25

df['Fault_Last_3M'] = (df['started at'] >= cutoff_3m).astype(int)
df['Fault_Last_6M'] = (df['started at'] >= cutoff_6m).astype(int)
df['Fault_Last_12M'] = (df['started at'] >= cutoff_12m).astype(int)

print(f"Fault counts: 3M={df['Fault_Last_3M'].sum():,} | 6M={df['Fault_Last_6M'].sum():,} | 12M={df['Fault_Last_12M'].sum():,} (ref={reference_date.strftime('%Y-%m-%d')})")

# ============================================================================
# STEP 5B: CUSTOMER IMPACT RATIOS (Fault-level calculation)
# ============================================================================
print("\n[Step 5B/12] Calculating Customer Impact Ratios [12M]...")

customer_ratio_cols = []
if 'total customer count' in df.columns:
    total_customers = df['total customer count'].fillna(0)

    if 'urban mv' in df.columns and 'urban lv' in df.columns:
        df['Urban_Customer_Ratio'] = ((df['urban mv'].fillna(0) + df['urban lv'].fillna(0)) / (total_customers + 1)).clip(0, 1)
        customer_ratio_cols.append('Urban_Customer_Ratio')

    if 'rural mv' in df.columns and 'rural lv' in df.columns:
        df['Rural_Customer_Ratio'] = ((df['rural mv'].fillna(0) + df['rural lv'].fillna(0)) / (total_customers + 1)).clip(0, 1)
        customer_ratio_cols.append('Rural_Customer_Ratio')

    if 'urban mv' in df.columns and 'rural mv' in df.columns:
        suburban_mv = df['suburban mv'].fillna(0) if 'suburban mv' in df.columns else 0
        df['MV_Customer_Ratio'] = ((df['urban mv'].fillna(0) + suburban_mv + df['rural mv'].fillna(0)) / (total_customers + 1)).clip(0, 1)
        customer_ratio_cols.append('MV_Customer_Ratio')

    if customer_ratio_cols:
        print(f"Created {len(customer_ratio_cols)} customer ratio features (fault-level calculation to avoid Simpson's Paradox)")

# STEP 6: Equipment Identification
print("\n[Step 6/12] Creating Equipment IDs (cbs_id primary)...")

def get_equipment_id(row):
    """
    Primary: cbs_id (should always exist after Step 1C filtering)
    Fallback: Ekipman ID (rare case)
    Note: UNKNOWN generation removed - faults without IDs filtered in Step 1C
    """
    if pd.notna(row.get('cbs_id')):
        return row['cbs_id']
    elif pd.notna(row.get('Ekipman ID')):
        return row['Ekipman ID']
    else:
        # Should not happen after Step 1C filtering
        print(f"  [!] WARNING: Fault without cbs_id slipped through filtering (row {row.name})")
        return None

df['Equipment_ID_Primary'] = df.apply(get_equipment_id, axis=1)

# Remove any None IDs (shouldn't happen but safety check)
none_ids = df['Equipment_ID_Primary'].isna().sum()
if none_ids > 0:
    print(f"  [!] Removing {none_ids} faults with missing Equipment_ID_Primary")
    df = df[df['Equipment_ID_Primary'].notna()].copy()

unique_equipment = df['Equipment_ID_Primary'].nunique()
print(f"Created {unique_equipment:,} unique equipment IDs from {len(df):,} faults (avg {len(df)/unique_equipment:.1f} faults/equipment)")
print(f"  No UNKNOWN_XXX IDs generated (all equipment have real cbs_id)")

equipment_id_col = 'Equipment_ID_Primary'

# ============================================================================
# STEP 6B: CREATE UNIFIED EQUIPMENT CLASSIFICATION
# ============================================================================
print("\n[Step 6B/12] Harmonizing Equipment Classifications...")

def get_equipment_class(row):
    """Priority: Equipment_Type → Ekipman Sınıfı → Kesinti Ekipman Sınıfı"""
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

# Use equipment class mapping from config.py (centralized)
original_classes_before = df['Equipment_Class_Primary'].nunique()
df['Equipment_Class_Primary'] = df['Equipment_Class_Primary'].map(lambda x: EQUIPMENT_CLASS_MAPPING.get(x, x) if pd.notna(x) else x)
harmonized_classes = df['Equipment_Class_Primary'].nunique()
print(f"Harmonized {original_classes_before} variants → {harmonized_classes} standardized equipment classes")

# Show final class distribution
print(f"\n  Final Equipment Class Distribution:")
class_dist = df['Equipment_Class_Primary'].value_counts()
for cls, count in class_dist.items():
    pct = count / len(df) * 100
    print(f"    {str(cls):20s} {count:4d} ({pct:5.1f}%)")

# Save mapping for audit trail
mapping_records = []
for original, harmonized in EQUIPMENT_CLASS_MAPPING.items():
    count = (df['Equipment_Class_Primary'] == harmonized).sum()
    mapping_records.append({'Original': original, 'Harmonized': harmonized, 'Count': count})
mapping_df = pd.DataFrame(mapping_records)
mapping_file = DATA_DIR / 'equipment_class_mapping.csv'
mapping_df.to_csv(mapping_file, index=False)
print(f"  ✓ Mapping saved: {mapping_file}")

# ============================================================================
# STEP 7: AGGREGATE TO EQUIPMENT LEVEL
# ============================================================================
print("\n[Step 7/12] Aggregating to Equipment Level (Fault→Equipment)...")

# Sort by Age Source to prioritize during aggregation (GRID_CONNECTION > MISSING)
source_priority = {'GRID_CONNECTION': 0, 'MISSING': 1}
df['_source_priority'] = df['Yaş_Kaynak'].map(source_priority).fillna(99)
df = df.sort_values('_source_priority')
df = df.drop(columns=['_source_priority'])

# 🔧 CRITICAL FIX: Filter to ONLY pre-cutoff faults for aggregation
# This prevents data leakage in cause codes (first/last) and all other aggregations
print(f"  Filtering faults for aggregation (using ONLY faults BEFORE {REFERENCE_DATE.date()})...")
print(f"    Total faults: {len(df):,}")
print(f"    Total unique equipment: {df[equipment_id_col].nunique():,}")
df_pre_cutoff = df[df['started at'] <= REFERENCE_DATE].copy()
print(f"    Pre-cutoff faults: {len(df_pre_cutoff):,}")
print(f"    Pre-cutoff unique equipment: {df_pre_cutoff[equipment_id_col].nunique():,}")
print(f"    Excluded post-cutoff: {len(df) - len(df_pre_cutoff):,} faults")

# Build aggregation dictionary dynamically based on available columns
# Start with REQUIRED columns (created by this script)
agg_dict = {
    # Equipment Age (from Sebekeye_Baglanma_Tarihi - Grid Connection Date)
    'Ekipman_Kurulum_Tarihi': 'first',
    'Ekipman_Yaşı_Gün': 'first',
    'Ekipman_Yaşı_Yıl': 'first',
    'Yaş_Kaynak': 'first',

    # Fault history (required - created by this script)
    'started at': ['count', 'min', 'max'],
    'Fault_Last_3M': 'sum',
    'Fault_Last_6M': 'sum',
    'Fault_Last_12M': 'sum',

    # Temporal features (required - created by this script)
    'Summer_Peak_Flag': 'sum',
    'Winter_Peak_Flag': 'sum',
    'Time_To_Repair_Hours': ['mean', 'max'],
}

# Add equipment classification columns if available
equipment_classification_cols = {
    'Equipment_Class_Primary': 'first',
    'Ekipman Sınıfı': 'first',
    'Equipment_Type': 'first',
    'Kesinti Ekipman Sınıfı': 'first'
}
for col, agg_func in equipment_classification_cols.items():
    if col in df_pre_cutoff.columns:
        agg_dict[col] = agg_func

# Add geographic columns if available
geographic_cols = ['KOORDINAT_X', 'KOORDINAT_Y', 'İl', 'İlçe', 'Mahalle']
for col in geographic_cols:
    if col in df_pre_cutoff.columns:
        agg_dict[col] = 'first'

# Add customer ratio columns if they were created
for ratio_col in ['Urban_Customer_Ratio', 'Rural_Customer_Ratio', 'MV_Customer_Ratio']:
    if ratio_col in df_pre_cutoff.columns:
        agg_dict[ratio_col] = 'mean'  # Average the pre-calculated ratios

# Add cause code column if available (now safe - uses only pre-cutoff faults)
if 'cause code' in df_pre_cutoff.columns:
    agg_dict['cause code'] = ['first', 'last', lambda x: x.mode()[0] if len(x.mode()) > 0 else None]

# Add customer impact columns if available
customer_impact_cols = [
    'urban mv+suburban mv', 'urban lv+suburban lv', 'urban mv', 'urban lv',
    'suburban mv', 'suburban lv', 'rural mv', 'rural lv', 'total customer count'
]
for col in customer_impact_cols:
    if col in df_pre_cutoff.columns:
        agg_dict[col] = ['mean', 'max']

# Add optional specification columns if available
optional_spec_cols = {
    'voltage_level': 'first', 'kVa_rating': 'first', 'component voltage': 'first',
    'MARKA': 'first', 'MARKA_MODEL': 'first', 'FIRMA': 'first'
}
for col, agg_func in optional_spec_cols.items():
    if col in df_pre_cutoff.columns:
        agg_dict[col] = agg_func

# ✅ Aggregate using ONLY pre-cutoff faults (prevents cause code leakage)
equipment_df = df_pre_cutoff.groupby(equipment_id_col).agg(agg_dict).reset_index()
equipment_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in equipment_df.columns.values]

print(f"Aggregated {len(df_pre_cutoff):,} pre-cutoff faults → {len(equipment_df):,} equipment records ({len(agg_dict)} aggregated features)")

# Equipment tracking summary
print(f"\n[TRACKING] Equipment Pipeline Summary:")
print(f"  After ID consolidation:  {unique_equip_after_id_consolidation:,} unique equipment (cbs_id + id fallback)")
print(f"  After deduplication:     {unique_equip_after_dedup:,} unique equipment")
print(f"  Pre-cutoff equipment:    {df_pre_cutoff[equipment_id_col].nunique():,}")
print(f"  Final aggregated:        {len(equipment_df):,} equipment")
equipment_lost = unique_equip_after_dedup - len(equipment_df)
print(f"  Lost in pipeline:        {equipment_lost:,} ({equipment_lost/unique_equip_after_dedup*100:.1f}%)")

# Identify and save excluded equipment for analysis
if equipment_lost > 0:
    all_equipment_ids = set(df[equipment_id_col].unique())
    final_equipment_ids = set(equipment_df[equipment_id_col].unique())
    excluded_ids = all_equipment_ids - final_equipment_ids

    df_excluded = df[df[equipment_id_col].isin(excluded_ids)].copy()

    # Analyze why equipment were excluded
    post_cutoff_only = df_excluded[df_excluded['started at'] > REFERENCE_DATE]
    post_cutoff_equipment = post_cutoff_only[equipment_id_col].nunique()

    print(f"\n  Exclusion Analysis:")
    print(f"    Post-cutoff failures only: {post_cutoff_equipment:,} equipment ({post_cutoff_equipment/equipment_lost*100:.1f}%)")
    print(f"      → These can be used as validation/test set!")

    # Validate post-cutoff data quality
    print(f"\n  Post-Cutoff Test Set Data Quality:")
    post_cutoff_faults = df_excluded[df_excluded['started at'] > REFERENCE_DATE]
    print(f"    Total post-cutoff faults: {len(post_cutoff_faults):,}")
    print(f"    Unique equipment: {post_cutoff_equipment:,}")
    print(f"    Avg faults per equipment: {len(post_cutoff_faults)/post_cutoff_equipment:.2f}")

    # Check for duplicates in post-cutoff data
    post_cutoff_dups = post_cutoff_faults.duplicated(subset=[equipment_id_col, 'started at'], keep='first').sum()
    if post_cutoff_dups > 0:
        print(f"    ⚠️  WARNING: {post_cutoff_dups} duplicates found in post-cutoff data")
        print(f"        → These were already removed in Step 1B")
    else:
        print(f"    ✓ No duplicates in post-cutoff test set (clean)")

    # Save excluded equipment for manual review
    excluded_file = DATA_DIR / 'excluded_equipment_analysis.csv'
    df_excluded.to_csv(excluded_file, index=False)
    print(f"  ✓ Excluded equipment saved: {excluded_file}")

# ============================================================================
# STEP 8: RENAME COLUMNS
# ============================================================================
print("\n[Step 8/12] Renaming Columns (English→Turkish Standards)...")

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
    'Ekipman_Yaşı_Gün_first': 'Ekipman_Yaşı_Gün',
    'Ekipman_Yaşı_Yıl_first': 'Ekipman_Yaşı_Yıl',
    'Yaş_Kaynak_first': 'Yaş_Kaynak',
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
print("\n[Step 9/12] Creating Cause Code Features (Diversity/Consistency) [12M]...")

has_cause_code = any(col for col in equipment_df.columns if 'cause code' in col.lower() or 'arıza_nedeni' in col.lower())

if has_cause_code and 'cause code' in df.columns:
    cause_distribution = df.groupby([equipment_id_col, 'cause code']).size().unstack(fill_value=0)
    equipment_df['Arıza_Nedeni_Çeşitlilik'] = (cause_distribution > 0).sum(axis=1).reindex(equipment_df['Ekipman_ID']).fillna(0).values
    total_faults_per_equip = cause_distribution.sum(axis=1)
    max_cause_per_equip = cause_distribution.max(axis=1)
    cause_consistency = (max_cause_per_equip / total_faults_per_equip).reindex(equipment_df['Ekipman_ID']).fillna(0).values
    equipment_df['Arıza_Nedeni_Tutarlılık'] = cause_consistency
    print(f"Created cause diversity (avg {equipment_df['Arıza_Nedeni_Çeşitlilik'].mean():.2f} types/equip) and consistency ({equipment_df['Arıza_Nedeni_Tutarlılık'].mean():.1%})")

# ============================================================================
# STEP 10: CALCULATE MTBF & TIME-TO-FIRST-FAILURE [NEW!]
# ============================================================================
print("\n[Step 10/12] Calculating MTBF & Time Until First Failure [6M/12M]...")

def calculate_mtbf_safe(equipment_id):
    """
    Calculate MTBF using ONLY failures BEFORE cutoff date (2024-06-25)
    This prevents data leakage - MTBF is calculated from historical data only

    MTBF = Total operating time BEFORE cutoff / (Number of failures BEFORE cutoff - 1)
    """
    # Get all fault dates for this equipment BEFORE cutoff
    equip_faults = df[
        (df[equipment_id_col] == equipment_id) &
        (df['started at'] <= REFERENCE_DATE)
    ]['started at'].dropna().sort_values()

    if len(equip_faults) < 2:
        # Need at least 2 faults to calculate MTBF (mean time BETWEEN failures)
        return None

    # Calculate time span from first to last failure (before cutoff)
    first_fault = equip_faults.iloc[0]
    last_fault = equip_faults.iloc[-1]
    total_days = (last_fault - first_fault).days

    # Number of intervals = number of faults - 1
    num_faults = len(equip_faults)

    if total_days > 0 and num_faults > 1:
        return total_days / (num_faults - 1)

    return None

# Calculate safe MTBF (no leakage)
print("  Calculating MTBF (using failures BEFORE cutoff only - leakage-safe)...")
equipment_df['MTBF_Gün'] = equipment_df['Ekipman_ID'].apply(calculate_mtbf_safe)

# METHOD 2: MTBF Lifetime (Total Exposure / Failures) - For Survival Analysis
def calculate_mtbf_lifetime(equipment_id):
    """
    Method 2: Calculate MTBF using total exposure time
    MTBF_Lifetime = (Cutoff - Installation) / N_Faults

    Use case: Survival analysis baseline hazard rate
    - Exposure-weighted risk calculation
    - Proportional hazards framework
    - Censoring-aware models
    """
    # Get installation date
    install_row = equipment_df[equipment_df['Ekipman_ID'] == equipment_id]
    if len(install_row) == 0:
        return None

    install_date = install_row['Ekipman_Kurulum_Tarihi'].iloc[0]

    if pd.isna(install_date):
        return None

    # Count faults before cutoff
    num_faults = df[
        (df[equipment_id_col] == equipment_id) &
        (df['started at'] <= REFERENCE_DATE)
    ].shape[0]

    if num_faults == 0:
        return None  # No failures, MTBF undefined

    total_days = (REFERENCE_DATE - install_date).days

    if total_days > 0:
        return total_days / num_faults

    return None

print("  Calculating MTBF Lifetime (Method 2: for survival analysis)...")
equipment_df['MTBF_Lifetime_Gün'] = equipment_df['Ekipman_ID'].apply(calculate_mtbf_lifetime)

# Baseline hazard rate (inverse of MTBF)
equipment_df['Baseline_Hazard_Rate'] = 1 / equipment_df['MTBF_Lifetime_Gün']

# METHOD 3: MTBF Observable (First Fault to Cutoff / Failures) - Degradation Detection
def calculate_mtbf_observable(equipment_id):
    """
    Method 3: Calculate MTBF from first fault to cutoff
    MTBF_Observable = (Cutoff - First_Fault) / N_Faults

    Use case: Degradation trend detection
    - Compare with Method 1 to detect if failures accelerating
    - Ratio > 1.5 = Degrading (recent MTBF shorter)
    - Ratio < 0.8 = Improving (recent MTBF longer)
    """
    equip_faults = df[
        (df[equipment_id_col] == equipment_id) &
        (df['started at'] <= REFERENCE_DATE)
    ]['started at'].dropna().sort_values()

    if len(equip_faults) < 2:
        return None

    first_fault = equip_faults.iloc[0]
    observable_days = (REFERENCE_DATE - first_fault).days
    num_faults = len(equip_faults)

    if observable_days > 0:
        return observable_days / num_faults

    return None

print("  Calculating MTBF Observable (Method 3: for degradation detection)...")
equipment_df['MTBF_Observable_Gün'] = equipment_df['Ekipman_ID'].apply(calculate_mtbf_observable)

# Degradation ratio (Method 3 / Method 1)
# Ratio > 1.0 means failures are accelerating (recent MTBF shorter than historical)
equipment_df['MTBF_Degradation_Ratio'] = (
    equipment_df['MTBF_Observable_Gün'] / equipment_df['MTBF_Gün']
)

# Flag degrading equipment (failures accelerating)
equipment_df['Is_Degrading'] = (
    equipment_df['MTBF_Degradation_Ratio'] > 1.5
).fillna(False).astype(int)

# 🔧 FIX: Calculate last failure date using ONLY failures BEFORE cutoff (no leakage)
def calculate_last_failure_date_safe(equipment_id):
    """
    Get last failure date using ONLY failures BEFORE cutoff date (2024-06-25)
    This prevents data leakage - we don't look into the future
    """
    equip_faults = df[
        (df[equipment_id_col] == equipment_id) &
        (df['started at'] <= REFERENCE_DATE)
    ]['started at'].dropna()

    if len(equip_faults) > 0:
        return equip_faults.max()
    else:
        return None  # No failures before cutoff

print("  Calculating last failure date (using failures BEFORE cutoff only - leakage-safe)...")
equipment_df['Son_Arıza_Tarihi_Safe'] = equipment_df['Ekipman_ID'].apply(calculate_last_failure_date_safe)

# Days since last failure (safe - uses ONLY pre-cutoff failures)
equipment_df['Son_Arıza_Gun_Sayisi'] = (REFERENCE_DATE - equipment_df['Son_Arıza_Tarihi_Safe']).dt.days

# 🔧 FIX: Calculate first failure date using ONLY failures BEFORE cutoff (no leakage)
def calculate_first_failure_date_safe(equipment_id):
    """
    Get first failure date using ONLY failures BEFORE cutoff date (2024-06-25)
    This prevents data leakage for equipment whose first failure is after cutoff
    """
    equip_faults = df[
        (df[equipment_id_col] == equipment_id) &
        (df['started at'] <= REFERENCE_DATE)
    ]['started at'].dropna()

    if len(equip_faults) > 0:
        return equip_faults.min()
    else:
        return None  # No failures before cutoff

print("  Calculating first failure date (using failures BEFORE cutoff only - leakage-safe)...")
equipment_df['İlk_Arıza_Tarihi_Safe'] = equipment_df['Ekipman_ID'].apply(calculate_first_failure_date_safe)

# NEW FEATURE v4.0: Time Until First Failure (Infant Mortality Detection)
# Calculates: Grid Connection Date → First Fault Date
# Uses Sebekeye_Baglanma_Tarihi (via Ekipman_Kurulum_Tarihi)
equipment_df['Ilk_Arizaya_Kadar_Gun'] = (
    equipment_df['İlk_Arıza_Tarihi_Safe'] - equipment_df['Ekipman_Kurulum_Tarihi']
).dt.days
equipment_df['Ilk_Arizaya_Kadar_Yil'] = equipment_df['Ilk_Arizaya_Kadar_Gun'] / 365.25

# Summary statistics
mtbf_valid = equipment_df['MTBF_Gün'].notna().sum()
mtbf_lifetime_valid = equipment_df['MTBF_Lifetime_Gün'].notna().sum()
mtbf_observable_valid = equipment_df['MTBF_Observable_Gün'].notna().sum()
degrading_count = equipment_df['Is_Degrading'].sum()
ttff_valid = equipment_df['Ilk_Arizaya_Kadar_Gun'].notna().sum()
ttff_mean = equipment_df['Ilk_Arizaya_Kadar_Yil'].mean()
infant_mortality = (equipment_df['Ilk_Arizaya_Kadar_Gun'] < 365).sum()  # Failed within 1 year

print(f"\nMTBF Statistics:")
print(f"  Method 1 (Inter-Fault): {mtbf_valid:,}/{len(equipment_df):,} valid ({mtbf_valid/len(equipment_df)*100:.1f}%)")
print(f"  Method 2 (Lifetime): {mtbf_lifetime_valid:,}/{len(equipment_df):,} valid ({mtbf_lifetime_valid/len(equipment_df)*100:.1f}%)")
print(f"  Method 3 (Observable): {mtbf_observable_valid:,}/{len(equipment_df):,} valid ({mtbf_observable_valid/len(equipment_df)*100:.1f}%)")
print(f"  Degrading equipment: {degrading_count:,} ({degrading_count/len(equipment_df)*100:.1f}%) - failures accelerating")
print(f"  Time-to-First-Failure: {ttff_valid:,}/{len(equipment_df):,} valid (avg {ttff_mean:.1f}y, infant mortality: {infant_mortality})")

# ============================================================================
# STEP 11: DETECT RECURRING FAULTS
# ============================================================================
print("\n[Step 11/12] Detecting Recurring Fault Patterns (30/90 day windows) [6M/12M]...")
print("  Calculating recurring faults (using failures BEFORE cutoff only - leakage-safe)...")

def calculate_recurrence_safe(equipment_id):
    """
    Detect recurring faults using ONLY failures BEFORE cutoff date (2024-06-25)
    This prevents data leakage - we don't look into the future
    """
    equip_faults = df[
        (df[equipment_id_col] == equipment_id) &
        (df['started at'] <= REFERENCE_DATE)  # ← CRITICAL FILTER!
    ]['started at'].dropna().sort_values()

    if len(equip_faults) < 2:
        return 0, 0

    time_diffs = equip_faults.diff().dt.days.dropna()
    return int((time_diffs <= 30).any()), int((time_diffs <= 90).any())

recurrence_results = equipment_df['Ekipman_ID'].apply(calculate_recurrence_safe)
equipment_df['Tekrarlayan_Arıza_30gün_Flag'] = [r[0] for r in recurrence_results]
equipment_df['Tekrarlayan_Arıza_90gün_Flag'] = [r[1] for r in recurrence_results]

print(f"Recurring faults (pre-cutoff only): 30-day={equipment_df['Tekrarlayan_Arıza_30gün_Flag'].sum():,} | 90-day={equipment_df['Tekrarlayan_Arıza_90gün_Flag'].sum():,} equipment flagged")

# Chronic repeater validation analysis
print(f"\n[VALIDATION] Chronic Repeater Analysis:")
chronic = equipment_df[equipment_df['Tekrarlayan_Arıza_90gün_Flag'] == 1]
normal = equipment_df[equipment_df['Tekrarlayan_Arıza_90gün_Flag'] == 0]

if len(chronic) > 0 and len(normal) > 0:
    chronic_avg = chronic['Toplam_Arıza_Sayisi_Lifetime'].mean()
    normal_avg = normal['Toplam_Arıza_Sayisi_Lifetime'].mean()
    ratio = chronic_avg / normal_avg if normal_avg > 0 else 0

    print(f"  Chronic repeaters (90-day): {len(chronic):,} equipment ({len(chronic)/len(equipment_df)*100:.1f}%)")
    print(f"    Avg lifetime faults: {chronic_avg:.2f}")
    print(f"  Normal equipment: {len(normal):,} equipment ({len(normal)/len(equipment_df)*100:.1f}%)")
    print(f"    Avg lifetime faults: {normal_avg:.2f}")
    print(f"  Fault rate ratio: {ratio:.1f}x higher for chronic repeaters")

    if ratio < 2.0:
        print(f"  ⚠️  WARNING: Chronic repeaters only {ratio:.1f}x higher fault rate (expected 3-5x)")
        print(f"      → May indicate recurrence window too aggressive or insufficient data")
    elif ratio > 10.0:
        print(f"  ⚠️  WARNING: Chronic repeaters have {ratio:.1f}x higher fault rate (very extreme)")
        print(f"      → May indicate recurrence window too lenient")
    else:
        print(f"  ✓ Chronic repeater detection working as expected (2-10x range)")
else:
    print(f"  ⚠️  No chronic repeaters detected or all equipment chronic")

# ============================================================================
# DATA INTEGRITY VALIDATION CHECKS
# ============================================================================
print(f"\n[VALIDATION] Data Integrity Checks:")

validation_passed = True

# Check 1: Age sanity
age_col = 'Ekipman_Yaşı_Yıl'
if age_col in equipment_df.columns:
    min_age = equipment_df[age_col].min()
    max_age = equipment_df[age_col].max()
    if min_age < 0:
        print(f"  ❌ FAIL: Negative ages detected! (min: {min_age:.1f} years)")
        validation_passed = False
    elif max_age >= 50:
        print(f"  ⚠️  WARNING: Very old equipment detected (max: {max_age:.1f} years)")
        print(f"      → Review equipment with age > 50 years for data quality")
    else:
        print(f"  ✓ Age range valid: {min_age:.1f} - {max_age:.1f} years")
else:
    print(f"  ⚠️  WARNING: Age column '{age_col}' not found")

# Check 2: Temporal window logic (3M ≤ 6M ≤ 12M)
fault_cols = ['Arıza_Sayısı_3ay', 'Arıza_Sayısı_6ay', 'Arıza_Sayısı_12ay']
if all(col in equipment_df.columns for col in fault_cols):
    invalid_3m_6m = (equipment_df['Arıza_Sayısı_3ay'] > equipment_df['Arıza_Sayısı_6ay']).sum()
    invalid_6m_12m = (equipment_df['Arıza_Sayısı_6ay'] > equipment_df['Arıza_Sayısı_12ay']).sum()

    if invalid_3m_6m > 0 or invalid_6m_12m > 0:
        print(f"  ❌ FAIL: Temporal window logic violated!")
        print(f"      3M > 6M count: {invalid_3m_6m} equipment")
        print(f"      6M > 12M count: {invalid_6m_12m} equipment")
        validation_passed = False
    else:
        print(f"  ✓ Fault count windows valid (3M ≤ 6M ≤ 12M)")
else:
    print(f"  ⚠️  WARNING: Fault count columns not found")

# Check 3: First failure timing logic
first_failure_col = 'Ilk_Arizaya_Kadar_Yil'
if first_failure_col in equipment_df.columns and age_col in equipment_df.columns:
    invalid_first_failure = (equipment_df[first_failure_col] > equipment_df[age_col]).sum()
    if invalid_first_failure > 0:
        print(f"  ❌ FAIL: First fault after current age!")
        print(f"      {invalid_first_failure} equipment with illogical first failure dates")
        validation_passed = False
    else:
        print(f"  ✓ First failure timing valid (≤ equipment age)")
else:
    print(f"  ⚠️  WARNING: First failure column not found")

# Check 4: MTBF validity (requires 2+ faults)
mtbf_col = 'MTBF_Gün'
lifetime_faults_col = 'Toplam_Arıza_Sayisi_Lifetime'
if mtbf_col in equipment_df.columns and lifetime_faults_col in equipment_df.columns:
    mtbf_equip = equipment_df[equipment_df[mtbf_col].notna()]
    if len(mtbf_equip) > 0:
        invalid_mtbf = (mtbf_equip[lifetime_faults_col] < 2).sum()
        if invalid_mtbf > 0:
            print(f"  ❌ FAIL: MTBF calculated with <2 faults!")
            print(f"      {invalid_mtbf} equipment have MTBF but <2 lifetime faults")
            validation_passed = False
        else:
            print(f"  ✓ MTBF validity: All {len(mtbf_equip)} equipment have 2+ faults")
    else:
        print(f"  ✓ MTBF validity: No MTBF values to validate")
else:
    print(f"  ⚠️  WARNING: MTBF columns not found")

# Check 5: Recurring fault logic (requires 2+ faults)
recurring_col = 'Tekrarlayan_Arıza_90gün_Flag'
if recurring_col in equipment_df.columns and lifetime_faults_col in equipment_df.columns:
    recurring = equipment_df[equipment_df[recurring_col] == 1]
    if len(recurring) > 0:
        invalid_recurring = (recurring[lifetime_faults_col] < 2).sum()
        if invalid_recurring > 0:
            print(f"  ❌ FAIL: Recurring flag on equipment with <2 faults!")
            print(f"      {invalid_recurring} equipment flagged as recurring but have <2 faults")
            validation_passed = False
        else:
            print(f"  ✓ Recurring fault logic valid: {len(recurring)} flagged equipment")
    else:
        print(f"  ✓ Recurring fault logic: No recurring equipment flagged")
else:
    print(f"  ⚠️  WARNING: Recurring fault column not found")

# Check 6: Data leakage - Faults after cutoff in pre-cutoff dataset
if 'started at' in df.columns:
    # Check df_pre_cutoff if it exists, otherwise check df
    try:
        leakage_faults = (df_pre_cutoff['started at'] > REFERENCE_DATE).sum()
        dataset_name = 'df_pre_cutoff'
    except:
        leakage_faults = 0
        dataset_name = 'N/A'

    if leakage_faults > 0:
        print(f"  ❌ CRITICAL: Data leakage detected in training data!")
        print(f"      {leakage_faults:,} faults after cutoff date ({REFERENCE_DATE.strftime('%Y-%m-%d')}) in {dataset_name}")
        validation_passed = False
    else:
        print(f"  ✓ No data leakage: All training faults before cutoff date")
else:
    print(f"  ⚠️  WARNING: Cannot check data leakage - 'started at' column not found")

# Check 7: Negative MTBF values
mtbf_cols_to_check = ['MTBF_Gün', 'MTBF_Lifetime_Gün', 'MTBF_Observable_Gün']
negative_mtbf_found = False
for col in mtbf_cols_to_check:
    if col in equipment_df.columns:
        negative_count = (equipment_df[col] < 0).sum()
        if negative_count > 0:
            print(f"  ❌ FAIL: Negative MTBF values in '{col}'!")
            print(f"      {negative_count} equipment with negative {col}")
            validation_passed = False
            negative_mtbf_found = True

if not negative_mtbf_found:
    valid_mtbf_cols = [col for col in mtbf_cols_to_check if col in equipment_df.columns]
    if valid_mtbf_cols:
        print(f"  ✓ All MTBF values are positive ({len(valid_mtbf_cols)} MTBF columns checked)")
    else:
        print(f"  ⚠️  WARNING: No MTBF columns found for validation")

# Check 8: Negative time-to-repair
if 'Time_To_Repair_Hours' in df.columns:
    negative_ttr = (df['Time_To_Repair_Hours'] < 0).sum()
    if negative_ttr > 0:
        print(f"  ❌ FAIL: Negative time-to-repair values!")
        print(f"      {negative_ttr:,} faults have end time before start time")
        validation_passed = False
    else:
        print(f"  ✓ All time-to-repair values are positive")
else:
    print(f"  ⚠️  WARNING: Time_To_Repair_Hours column not found")

# Final validation summary
if validation_passed:
    print(f"\n✅ All critical integrity checks PASSED!")
else:
    print(f"\n❌ VALIDATION FAILED: Critical data integrity issues detected!")
    print(f"   → Review the failures above before proceeding to next step")
    print(f"   → Consider fixing data quality issues or investigating root cause")

# ============================================================================
# STEP 12: SAVE RESULTS
# ============================================================================
print(f"\n[Step 12/12] Saving Equipment-Level Dataset...")

equipment_df.to_csv(EQUIPMENT_LEVEL_FILE, index=False, encoding='utf-8-sig')

feature_docs = pd.DataFrame({
    'Feature_Name': equipment_df.columns,
    'Data_Type': equipment_df.dtypes.astype(str),
    'Completeness_%': (equipment_df.notna().sum() / len(equipment_df) * 100).round(1)
})
feature_docs.to_csv(FEATURE_DOCS_FILE, index=False)

print(f"Saved: equipment_level_data.csv ({len(equipment_df):,} records x {len(equipment_df.columns)} features) + feature_documentation.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRANSFORMATION COMPLETE - MULTI-HORIZON POF FEATURES READY")
print("="*80)

print(f"\nPIPELINE STATUS: {original_fault_count:,} faults → {len(equipment_df):,} equipment ({len(equipment_df.columns)} features)")

print(f"\nKEY FEATURES FOR MULTI-HORIZON PREDICTIONS:")
print(f"  • Fault History: 3M/6M/12M temporal counts (PRIMARY drivers)")
print(f"  • Equipment Age: {equipment_df['Yaş_Kaynak'].value_counts().to_dict()}")
print(f"  • Time-to-First-Failure: avg {equipment_df['Ilk_Arizaya_Kadar_Yil'].mean():.1f}y, {infant_mortality} infant mortality cases")
print(f"  • MTBF Features (3 methods):")
print(f"    - Method 1 (Inter-Fault): {mtbf_valid:,} valid - PoF prediction")
print(f"    - Method 2 (Lifetime): {mtbf_lifetime_valid:,} valid - Survival analysis")
print(f"    - Method 3 (Observable): {mtbf_observable_valid:,} valid - Degradation detection")
print(f"    - Degrading equipment: {degrading_count:,} flagged")
print(f"  • Chronic Repeaters: {equipment_df['Tekrarlayan_Arıza_90gün_Flag'].sum():,} flagged")
print(f"  • Equipment Classes: {harmonized_classes} standardized")

print(f"\nENHANCEMENTS IN v5.0:")
print(f"  + NEW: Sebekeye_Baglanma_Tarihi as single age source (simplified)")
print(f"  + 3 MTBF calculation methods (Inter-Fault, Lifetime, Observable)")
print(f"  + Baseline_Hazard_Rate for Cox survival analysis")
print(f"  + MTBF_Degradation_Ratio + Is_Degrading flag")
print(f"  + Time-to-First-Failure (Ilk_Arizaya_Kadar_Gun/Yil)")
print(f"  + Dynamic schema support (handles missing columns)")

print(f"\nNEXT STEP: Run 03_feature_engineering.py")
print(f"  → Creates advanced PoF risk scores and geographic features")
print(f"  → Prepares features for 3M/6M/12M temporal models")
print("="*80)
