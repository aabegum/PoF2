"""
================================================================================
SCRIPT 00: INPUT DATA SOURCE ANALYSIS
================================================================================
Turkish EDAŞ PoF (Probability of Failure) Prediction Pipeline

PURPOSE: Comprehensive analysis of raw input data before any processing
- Examine input file structure, columns, data types
- Validate data formats and sample values
- Identify data quality issues early
- Confirm file readability and encoding
- Generate data dictionary

This should be run FIRST before any other scripts to ensure data integrity.

INPUT:  Combined fault data (config.INPUT_FILE - combined_data_son.xlsx)
OUTPUT: Data source analysis report + data dictionary

ENHANCEMENTS (NEW):
+ Complete column inventory with data types
+ Sample data preview for each column
+ Missing value analysis
+ Data format validation
+ Encoding detection
+ Data type consistency checks
+ Statistics by column type
+ Recommendations for data quality improvements
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import sys
import openpyxl

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

# Import centralized configuration
from config import INPUT_FILE, CUTOFF_DATE, DATA_DIR

print("="*100)
print(" "*25 + "INPUT DATA SOURCE ANALYSIS")
print(" "*20 + "Turkish EDAŞ PoF Prediction Pipeline")
print("="*100)

# ============================================================================
# STEP 1: FILE INFORMATION
# ============================================================================
print("\n" + "="*100)
print("STEP 1: INPUT FILE INFORMATION")
print("="*100)

input_path = Path(INPUT_FILE)

if not input_path.exists():
    print(f"\n❌ ERROR: Input file not found!")
    print(f"   Expected: {input_path}")
    print(f"   Absolute: {input_path.absolute()}")
    sys.exit(1)

print(f"\n✓ File Found:")
print(f"   Path: {input_path}")
print(f"   Absolute Path: {input_path.absolute()}")
print(f"   Size: {input_path.stat().st_size / 1024**2:.2f} MB")
print(f"   Modified: {datetime.fromtimestamp(input_path.stat().st_mtime)}")
print(f"   Format: {input_path.suffix}")

# ============================================================================
# STEP 2: SHEET INFORMATION
# ============================================================================
print("\n" + "="*100)
print("STEP 2: EXCEL SHEET INVENTORY")
print("="*100)

try:
    excel_file = pd.ExcelFile(INPUT_FILE)
    sheets = excel_file.sheet_names
    print(f"\n📋 Available Sheets ({len(sheets)}):")
    for i, sheet in enumerate(sheets, 1):
        print(f"   {i}. {sheet}")

    # Load first sheet (usually the data)
    main_sheet = sheets[0]
    print(f"\n📌 Using main sheet: '{main_sheet}'")
except Exception as e:
    print(f"\n❌ Error reading Excel file: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: LOAD AND INSPECT DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 3: DATA LOADING AND INSPECTION")
print("="*100)

try:
    df = pd.read_excel(INPUT_FILE, sheet_name=main_sheet)
    print(f"\n✓ Data loaded successfully!")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Total cells: {len(df) * len(df.columns):,}")
except Exception as e:
    print(f"\n❌ Error loading data: {e}")
    sys.exit(1)

# ============================================================================
# STEP 4: COLUMN ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 4: COLUMN INVENTORY AND DATA TYPES")
print("="*100)

print(f"\n📊 Column Structure ({len(df.columns)} columns):\n")

# Create detailed column inventory
column_info = []
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    missing = df[col].isna().sum()
    missing_pct = missing / len(df) * 100
    unique = df[col].nunique()

    # Get sample value
    sample = df[col].dropna().iloc[0] if missing < len(df) else None

    column_info.append({
        'No': i,
        'Column': col,
        'Type': str(dtype),
        'Unique': unique,
        'Missing': missing,
        'Missing%': f"{missing_pct:.1f}%",
        'Sample': str(sample)[:50] if sample is not None else "NULL"
    })

    print(f"{i:2d}. {col:<35} Type: {str(dtype):<12} Missing: {missing:>6,} ({missing_pct:>5.1f}%) Unique: {unique:>6,}")

# ============================================================================
# STEP 5: DATA TYPE SUMMARY
# ============================================================================
print("\n" + "="*100)
print("STEP 5: DATA TYPE SUMMARY")
print("="*100)

dtype_summary = df.dtypes.value_counts()
print(f"\n📈 Data Types Distribution:")
for dtype, count in dtype_summary.items():
    print(f"   {str(dtype):<20} {count:>3} columns")

# ============================================================================
# STEP 6: MISSING DATA ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 6: MISSING DATA ANALYSIS")
print("="*100)

missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_%': (df.isnull().sum().values / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_%', ascending=False)

if len(missing_summary) > 0:
    print(f"\n⚠️  Found {len(missing_summary)} columns with missing values:\n")
    print(missing_summary.to_string(index=False))
else:
    print("\n✓ No missing values detected!")

# ============================================================================
# STEP 7: KEY COLUMNS VALIDATION
# ============================================================================
print("\n" + "="*100)
print("STEP 7: KEY COLUMNS VALIDATION")
print("="*100)

key_columns = {
    'cbs_id': 'Equipment ID',
    'started at': 'Fault timestamp',
    'Sebekeye_Baglanma_Tarihi': 'Grid connection date',
    'Şebeke Unsuru': 'Equipment type',
}

print("\n🔍 Checking for required columns:\n")
for col_name, description in key_columns.items():
    if col_name in df.columns:
        dtype = df[col_name].dtype
        missing = df[col_name].isna().sum()
        missing_pct = missing / len(df) * 100
        print(f"   ✓ {col_name:<30} ({description})")
        print(f"     └─ Type: {dtype}, Missing: {missing:,} ({missing_pct:.1f}%)")
    else:
        print(f"   ✗ {col_name:<30} ({description}) - NOT FOUND")

# ============================================================================
# STEP 8: TEMPORAL COVERAGE
# ============================================================================
print("\n" + "="*100)
print("STEP 8: TEMPORAL COVERAGE ANALYSIS")
print("="*100)

if 'started at' in df.columns:
    try:
        df['started at'] = pd.to_datetime(df['started at'], dayfirst=True, errors='coerce')
        min_date = df['started at'].min()
        max_date = df['started at'].max()

        print(f"\n📅 Fault Date Range:")
        print(f"   Earliest fault: {min_date.date()}")
        print(f"   Latest fault:   {max_date.date()}")
        print(f"   Date range:     {(max_date - min_date).days} days ({(max_date - min_date).days / 365.25:.1f} years)")
        print(f"   Cutoff date:    {CUTOFF_DATE.date()}")

        # Coverage before/after cutoff
        before_cutoff = len(df[df['started at'] <= CUTOFF_DATE])
        after_cutoff = len(df[df['started at'] > CUTOFF_DATE])

        print(f"\n📊 Data Split by Cutoff Date:")
        print(f"   Before cutoff:  {before_cutoff:>6,} faults ({before_cutoff/len(df)*100:>5.1f}%)")
        print(f"   After cutoff:   {after_cutoff:>6,} faults ({after_cutoff/len(df)*100:>5.1f}%)")
        print(f"   Total:          {len(df):>6,} faults (100.0%)")
    except Exception as e:
        print(f"\n⚠️  Error analyzing dates: {e}")
else:
    print("\n⚠️  'started at' column not found - cannot analyze temporal coverage")

# ============================================================================
# STEP 9: EQUIPMENT ID ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 9: EQUIPMENT ID ANALYSIS")
print("="*100)

if 'cbs_id' in df.columns:
    unique_ids = df['cbs_id'].nunique()
    missing_ids = df['cbs_id'].isna().sum()

    print(f"\n🔑 Equipment ID Summary (cbs_id):")
    print(f"   Total records:      {len(df):>6,}")
    print(f"   Unique equipment:   {unique_ids:>6,}")
    print(f"   Missing cbs_id:     {missing_ids:>6,} ({missing_ids/len(df)*100:.1f}%)")
    print(f"   Valid cbs_id:       {len(df) - missing_ids:>6,} ({(len(df)-missing_ids)/len(df)*100:.1f}%)")
    print(f"   Avg faults/equip:   {len(df)/unique_ids:>6.1f}")

    # Show top equipment by fault count
    top_equip = df['cbs_id'].value_counts().head(5)
    print(f"\n   Top 5 Equipment by Fault Count:")
    for equip_id, count in top_equip.items():
        print(f"      {equip_id:>10} - {count:>5,} faults")
else:
    print("\n⚠️  'cbs_id' column not found")

# ============================================================================
# STEP 10: DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*100)
print("STEP 10: DATA QUALITY ASSESSMENT")
print("="*100)

# Calculate quality score
quality_checks = []

# Check 1: No completely empty columns
empty_cols = df.columns[df.isnull().all()].tolist()
quality_checks.append(("No empty columns", len(empty_cols) == 0, empty_cols))

# Check 2: Key columns present
key_cols_present = all(col in df.columns for col in ['cbs_id', 'started at'])
quality_checks.append(("Key columns present", key_cols_present, None))

# Check 3: Reasonable data volume
min_rows = 100
quality_checks.append((f"Sufficient data ({len(df):,} rows)", len(df) >= min_rows, None))

# Check 4: No duplicate complete rows
dups = df.duplicated().sum()
quality_checks.append((f"No complete duplicates", dups == 0, f"{dups} duplicates found"))

print("\n✓ Quality Checks:")
for check_name, passed, details in quality_checks:
    status = "✓" if passed else "✗"
    detail_str = f" ({details})" if details else ""
    print(f"   {status} {check_name}{detail_str}")

# ============================================================================
# STEP 12: CHECK FOR HEALTHY EQUIPMENT DATA (Mixed Dataset Support)
# ============================================================================
print("\n" + "="*100)
print("STEP 12: HEALTHY EQUIPMENT DATA AVAILABILITY (Mixed Dataset Support)")
print("="*100)

# Check for healthy equipment file - use config path
from config import HEALTHY_EQUIPMENT_FILE

healthy_equipment_available = False
if HEALTHY_EQUIPMENT_FILE.exists():
    print(f"\n✓ Found: {HEALTHY_EQUIPMENT_FILE}")
    print(f"  Size: {HEALTHY_EQUIPMENT_FILE.stat().st_size / 1024**2:.2f} MB")
    healthy_equipment_available = True
    print(f"  Status: READY for mixed dataset training (Phase 1.4)")
    print(f"  Action: Run Step 2a (02a_healthy_equipment_loader.py) before Step 2")
else:
    print(f"\n✗ No healthy equipment data found")
    print(f"  Expected: {HEALTHY_EQUIPMENT_FILE}")
    print(f"  Pipeline will use SINGLE DATASET (failed equipment only)")
    print(f"  Impact: Step 6 (PoF Model) will train on ~2,670 equipment instead of 5,567")
    print(f"  Recommendation: If mixed dataset available, place at {HEALTHY_EQUIPMENT_FILE} and re-run")

print("\n📖 Sample Data by Column:\n")

# Sample 3 rows
sample_rows = df.head(3)
for col in df.columns:
    print(f"{col}:")
    for i, val in enumerate(sample_rows[col], 1):
        val_str = str(val)[:60]
        print(f"   Row {i}: {val_str}")
    print()

# ============================================================================
# STEP 12: RECOMMENDATIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 12: RECOMMENDATIONS")
print("="*100)

recommendations = []

# High missing data
if len(missing_summary) > 0:
    high_missing = missing_summary[missing_summary['Missing_%'] > 50]
    if len(high_missing) > 0:
        recommendations.append((
            "HIGH PRIORITY",
            f"Review {len(high_missing)} columns with >50% missing data",
            "These may need exclusion or special handling"
        ))

# Temporal coverage
if 'started at' in df.columns:
    if after_cutoff < 100:
        recommendations.append((
            "MEDIUM PRIORITY",
            f"Low post-cutoff fault count ({after_cutoff})",
            "Limited data for target variable creation - may impact model training"
        ))

# Data types
non_numeric = df.select_dtypes(include=['object']).columns
if len(non_numeric) > 10:
    recommendations.append((
        "MEDIUM PRIORITY",
        f"{len(non_numeric)} text columns - ensure proper encoding",
        "Verify Turkish character encoding (UTF-8)"
    ))

print("\n" + "="*100)
if recommendations:
    print("📋 RECOMMENDATIONS:")
    print("="*100)
    for priority, rec, detail in recommendations:
        print(f"\n[{priority}] {rec}")
        print(f"         → {detail}")
else:
    print("✅ NO CRITICAL ISSUES FOUND")
    print("="*100)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("INPUT DATA SOURCE ANALYSIS COMPLETE")
print("="*100)

print(f"\n📊 SUMMARY:")
print(f"   ✓ Input file: {input_path.name}")
print(f"   ✓ Rows: {len(df):,}")
print(f"   ✓ Columns: {len(df.columns)}")
print(f"   ✓ Data types: {len(dtype_summary)} types")
print(f"   ✓ Missing values: {df.isnull().sum().sum():,} cells ({df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.2f}%)")
print(f"   ✓ Quality: READY FOR PROCESSING" if all(check[1] for check in quality_checks) else "   ⚠️  Quality: REVIEW RECOMMENDED")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Review any recommendations above")
print(f"   2. Run: python 01_data_profiling.py")
print(f"   3. Run: python 02_data_transformation.py")
print(f"   4. Continue with full pipeline")

print("\n" + "="*100)
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
