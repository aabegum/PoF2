"""
DETECT AND FIX DATE FORMAT ISSUES
Turkish EDAŞ PoF Prediction Project

Purpose:
- Detect various date/time formats in 'started at' and 'ended at' columns
- Identify why pandas can't parse certain dates
- Provide unified parsing solution

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

print("="*100)
print(" "*30 + "DATE FORMAT DETECTION & FIX")
print("="*100)

# ============================================================================
# STEP 1: LOAD RAW DATA (without parsing dates)
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING RAW DATA")
print("="*100)

data_file = 'data/combined_data.xlsx'
if not Path(data_file).exists():
    print(f"\n❌ ERROR: Could not find {data_file}")
    exit(1)

print(f"\n✓ Loading: {data_file}")
# Load WITHOUT parsing dates - keep as raw strings/numbers
df = pd.read_excel(data_file)
print(f"✓ Loaded: {len(df):,} records")

# ============================================================================
# STEP 2: ANALYZE RAW DATE COLUMN TYPES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: ANALYZING RAW DATE COLUMN TYPES")
print("="*100)

date_cols = ['started at', 'ended at']

for col in date_cols:
    if col not in df.columns:
        print(f"\n❌ Column '{col}' not found!")
        continue

    print(f"\n📊 Column: '{col}'")
    print(f"   Data type: {df[col].dtype}")

    # Check for actual nulls
    actual_nulls = df[col].isna().sum()
    print(f"   Actual NULL values: {actual_nulls} ({actual_nulls/len(df)*100:.1f}%)")

    # Get non-null values
    non_null = df[col].dropna()
    print(f"   Non-NULL values: {len(non_null)} ({len(non_null)/len(df)*100:.1f}%)")

    if len(non_null) > 0:
        # Sample raw values
        print(f"\n   Sample RAW values (first 10 non-null):")
        for idx, val in enumerate(non_null.head(10)):
            val_type = type(val).__name__
            print(f"      [{idx+1}] Type: {val_type:20s} Value: {val}")

        # Check data type distribution
        type_counts = non_null.apply(lambda x: type(x).__name__).value_counts()
        print(f"\n   Data type distribution:")
        for dtype, count in type_counts.items():
            print(f"      {dtype:20s}: {count:,} ({count/len(non_null)*100:.1f}%)")

# ============================================================================
# STEP 3: DETECT DATE FORMATS
# ============================================================================
print("\n" + "="*100)
print("STEP 3: DETECTING DATE FORMATS")
print("="*100)

def detect_date_format(value):
    """Detect the format of a date value"""

    # Handle pandas Timestamp (already parsed by Excel reader)
    if isinstance(value, pd.Timestamp):
        return 'pandas_timestamp'

    # Handle datetime
    if isinstance(value, datetime):
        return 'datetime'

    # Handle numeric (Excel serial date)
    if isinstance(value, (int, float)):
        # Excel dates are typically between 1 (1900-01-01) and 60000 (2164)
        if 1 <= value <= 100000:
            return 'excel_serial_date'
        else:
            return 'numeric_unknown'

    # Handle string formats
    if isinstance(value, str):
        value = value.strip()

        # Empty string
        if not value:
            return 'empty_string'

        # Common patterns
        patterns = {
            'iso_datetime': r'^\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}',  # 2021-01-15 12:30:45
            'iso_date': r'^\d{4}-\d{2}-\d{2}$',  # 2021-01-15
            'dmy_slash': r'^\d{1,2}/\d{1,2}/\d{4}',  # 15/01/2021
            'mdy_slash': r'^\d{1,2}/\d{1,2}/\d{4}',  # 01/15/2021 (same pattern as dmy!)
            'dmy_dot': r'^\d{1,2}\.\d{1,2}\.\d{4}',  # 15.01.2021
            'dmy_dash': r'^\d{1,2}-\d{1,2}-\d{4}',  # 15-01-2021
            'turkish_text': r'[A-Za-zğüşıöçĞÜŞİÖÇ]+',  # Contains Turkish characters
        }

        for pattern_name, pattern in patterns.items():
            if re.search(pattern, value):
                return pattern_name

        return 'string_unknown'

    return 'unknown_type'

for col in date_cols:
    if col not in df.columns:
        continue

    print(f"\n📊 Format detection for: '{col}'")

    # Apply format detection
    df[f'{col}_format'] = df[col].apply(detect_date_format)

    # Count formats
    format_counts = df[f'{col}_format'].value_counts()

    print(f"\n   Format distribution:")
    for fmt, count in format_counts.items():
        pct = count / len(df) * 100
        print(f"      {fmt:25s}: {count:4,} ({pct:5.1f}%)")

    # Show examples of each format
    print(f"\n   Examples by format:")
    for fmt in format_counts.index[:5]:  # Top 5 formats
        examples = df[df[f'{col}_format'] == fmt][col].head(3)
        print(f"\n      {fmt}:")
        for ex in examples:
            print(f"         • {ex}")

# ============================================================================
# STEP 4: UNIFIED DATE PARSING
# ============================================================================
print("\n" + "="*100)
print("STEP 4: UNIFIED DATE PARSING")
print("="*100)

def parse_date_flexible(value):
    """
    Parse date with multiple format attempts
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
                # But Excel has a leap year bug for 1900
                return pd.Timestamp('1899-12-30') + pd.Timedelta(days=value)
            except:
                return pd.NaT
        else:
            return pd.NaT

    # String parsing
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return pd.NaT

        # Try multiple formats
        formats = [
            '%Y-%m-%d %H:%M:%S',     # 2021-01-15 12:30:45
            '%Y-%m-%d',              # 2021-01-15
            '%d/%m/%Y %H:%M:%S',     # 15/01/2021 12:30:45
            '%d/%m/%Y',              # 15/01/2021
            '%m/%d/%Y %H:%M:%S',     # 01/15/2021 12:30:45 (US format)
            '%m/%d/%Y',              # 01/15/2021
            '%d.%m.%Y %H:%M:%S',     # 15.01.2021 12:30:45 (Turkish)
            '%d.%m.%Y',              # 15.01.2021
            '%d-%m-%Y %H:%M:%S',     # 15-01-2021 12:30:45
            '%d-%m-%Y',              # 15-01-2021
            '%Y/%m/%d %H:%M:%S',     # 2021/01/15 12:30:45
            '%Y/%m/%d',              # 2021/01/15
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except:
                continue

        # Last resort: let pandas infer
        try:
            return pd.to_datetime(value, infer_datetime_format=True)
        except:
            return pd.NaT

    return pd.NaT

print("\n🔧 Applying flexible date parser...")

for col in date_cols:
    if col not in df.columns:
        continue

    print(f"\n   Processing: '{col}'")

    # Apply flexible parser
    df[f'{col}_parsed'] = df[col].apply(parse_date_flexible)

    # Count results
    before_nulls = df[col].apply(lambda x: pd.isna(x) or (isinstance(x, str) and not x.strip())).sum()
    after_nulls = df[f'{col}_parsed'].isna().sum()
    successfully_parsed = len(df) - after_nulls

    print(f"      Before parsing: {before_nulls:,} nulls/empty")
    print(f"      After parsing:  {after_nulls:,} nulls (NaT)")
    print(f"      Successfully parsed: {successfully_parsed:,} ({successfully_parsed/len(df)*100:.1f}%)")

    if after_nulls < before_nulls:
        recovered = before_nulls - after_nulls
        print(f"      ✅ RECOVERED: {recovered:,} dates that were unparseable before!")

    # Show examples of recovered dates
    was_null = df[col].apply(lambda x: pd.isna(x) or (isinstance(x, str) and not x.strip()))
    now_valid = df[f'{col}_parsed'].notna()
    recovered_mask = was_null & now_valid

    if recovered_mask.sum() > 0:
        print(f"\n      Examples of RECOVERED dates:")
        recovered_examples = df[recovered_mask][[col, f'{col}_parsed']].head(5)
        for idx, row in recovered_examples.iterrows():
            print(f"         Original: {row[col]} → Parsed: {row[f'{col}_parsed']}")

# ============================================================================
# STEP 5: VALIDATION
# ============================================================================
print("\n" + "="*100)
print("STEP 5: VALIDATION")
print("="*100)

print("\n📊 FINAL COVERAGE COMPARISON:")

for col in date_cols:
    if col not in df.columns:
        continue

    # Original pandas parsing
    original_parsed = pd.to_datetime(df[col], errors='coerce').notna().sum()

    # Our flexible parsing
    flexible_parsed = df[f'{col}_parsed'].notna().sum()

    improvement = flexible_parsed - original_parsed

    print(f"\n   '{col}':")
    print(f"      Original pandas parsing: {original_parsed:,} ({original_parsed/len(df)*100:.1f}%)")
    print(f"      Flexible parsing:        {flexible_parsed:,} ({flexible_parsed/len(df)*100:.1f}%)")
    if improvement > 0:
        print(f"      ✅ IMPROVEMENT: +{improvement:,} records ({improvement/len(df)*100:.1f}%)")
    else:
        print(f"      No improvement")

# ============================================================================
# STEP 6: SAVE FIXED DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 6: RECOMMENDATIONS")
print("="*100)

if df['started at_parsed'].notna().sum() > pd.to_datetime(df['started at'], errors='coerce').notna().sum():
    print("\n✅ DATE PARSING IMPROVED!")
    print("\n💡 RECOMMENDED ACTIONS:")
    print("   1. Update 02_data_transformation.py to use flexible date parser")
    print("   2. Or save fixed data to new file")
    print("   3. Re-run temporal diagnostic with improved dates")

    # Option to save
    print("\n💾 Save fixed data? (Y/N)")
    print("   This will create: data/combined_data_FIXED_DATES.xlsx")
    print("   With columns: 'started at_parsed' and 'ended at_parsed'")

else:
    print("\n✅ NO FORMAT ISSUES DETECTED")
    print("   Pandas already parsing all dates correctly")
    print("   The 'missing' timestamps are truly NULL in the source data")

print("\n" + "="*100)
print(f"{'DATE FORMAT DETECTION COMPLETE':^100}")
print("="*100)

# Ask if user wants to save
save_option = input("\nDo you want to save fixed data? (y/n): ").lower().strip()

if save_option == 'y':
    # Replace original columns with parsed versions
    df['started at'] = df['started at_parsed']
    df['ended at'] = df['ended at_parsed']

    # Drop helper columns
    df = df.drop(columns=[c for c in df.columns if '_format' in c or '_parsed' in c])

    # Save
    output_file = 'data/combined_data_FIXED_DATES.xlsx'
    print(f"\n💾 Saving to: {output_file}")
    df.to_excel(output_file, index=False)
    print(f"✓ Saved: {len(df):,} records")

    print("\n📋 NEXT STEPS:")
    print("   1. Verify the fixed file: data/combined_data_FIXED_DATES.xlsx")
    print("   2. If satisfied, rename it to: data/combined_data.xlsx")
    print("   3. Re-run: python check_fault_timestamps.py")
else:
    print("\n✓ Data not saved - review results first")
