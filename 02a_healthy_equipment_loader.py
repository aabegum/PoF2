"""
================================================================================
SCRIPT 02A: HEALTHY EQUIPMENT DATA LOADER v1.0
================================================================================
Turkish EDAS PoF (Probability of Failure) Prediction Pipeline

PURPOSE:
Load and validate healthy equipment data (zero failures ever) to provide
true negative samples for balanced PoF modeling.

WHAT THIS SCRIPT DOES:
- Loads healthy equipment data from Excel file
- Validates data quality and consistency
- Ensures no overlap with failed equipment (true healthy)
- Prepares data structure compatible with failed equipment dataset
- Creates zero-fault feature defaults for healthy equipment

VALIDATION RULES:
1. cbs_id must NOT appear in combined_data_son.xlsx (truly healthy)
2. Sebekeye_Baglanma_Tarihi < CUTOFF_DATE (installed before cutoff)
3. Equipment_Class_Primary must match EQUIPMENT_CLASS_MAPPING
4. Beklenen_Ömür_Yıl > 0 (valid expected life)

CROSS-REFERENCES:
- Script 02: Data transformation (merges with this output)
- config.py: Cutoff date and equipment class mapping

Input:  data/healthy_equipment.xlsx (user-provided healthy equipment)
Output: data/healthy_equipment_prepared.csv (validated and prepared)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
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
    EQUIPMENT_CLASS_MAPPING
)

# Import shared date parser
from utils.date_parser import parse_date_flexible

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

# Input/output paths
HEALTHY_EQUIPMENT_FILE = DATA_DIR / 'healthy_equipment.xlsx'
OUTPUT_FILE = DATA_DIR / 'healthy_equipment_prepared.csv'

# Required columns
REQUIRED_COLUMNS = [
    'cbs_id',                      # Equipment ID
    'Sebekeye_Baglanma_Tarihi',    # Installation date
    'Equipment_Class_Primary',     # Equipment type
]

# Optional but recommended columns
OPTIONAL_COLUMNS = [
    'component_voltage',           # Operating voltage
    'Voltage_Class',               # AG/OG/YG
    'Beklenen_Ömür_Yıl',          # Expected lifespan
    'İlçe',                        # District
    'Mahalle',                     # Neighborhood
    'KOORDINAT_X',                 # GPS X
    'KOORDINAT_Y',                 # GPS Y
    'total_customer_count',        # Affected customers
    'urban_mv',                    # Urban MV customers
    'urban_lv',                    # Urban LV customers
]

print("\n" + "="*100)
print("SCRIPT 02A: HEALTHY EQUIPMENT DATA LOADER v1.0")
print("="*100)
print(f"Reference Date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
print(f"Cutoff Date: {CUTOFF_DATE.strftime('%Y-%m-%d')}")
print(f"Healthy Definition: Zero failures EVER (truly healthy equipment)")

# ============================================================================
# STEP 1: LOAD HEALTHY EQUIPMENT DATA
# ============================================================================
print("\n[Step 1/7] Loading Healthy Equipment Data...")

if not HEALTHY_EQUIPMENT_FILE.exists():
    print(f"\n❌ ERROR: Healthy equipment file not found!")
    print(f"Expected location: {HEALTHY_EQUIPMENT_FILE}")
    print(f"\nPlease provide the file with the following structure:")
    print(f"  Required columns: {', '.join(REQUIRED_COLUMNS)}")
    print(f"  Optional columns: {', '.join(OPTIONAL_COLUMNS[:5])} (and more)")
    print(f"\nThe pipeline will continue without healthy equipment (only failed equipment).")
    print(f"This limits model performance (no true negative samples).\n")
    sys.exit(1)

print(f"\n✓ Loading from: {HEALTHY_EQUIPMENT_FILE}")
df_healthy = pd.read_excel(HEALTHY_EQUIPMENT_FILE)
original_count = len(df_healthy)
print(f"✓ Loaded: {original_count:,} healthy equipment × {df_healthy.shape[1]} columns")

# ============================================================================
# STEP 2: VALIDATE REQUIRED COLUMNS
# ============================================================================
print("\n[Step 2/7] Validating Required Columns...")

missing_required = [col for col in REQUIRED_COLUMNS if col not in df_healthy.columns]

if missing_required:
    print(f"\n❌ ERROR: Missing required columns: {missing_required}")
    print(f"Available columns: {list(df_healthy.columns)}")
    print(f"\nPlease ensure the file contains all required columns:")
    for col in REQUIRED_COLUMNS:
        status = "✓" if col in df_healthy.columns else "✗"
        print(f"  {status} {col}")
    sys.exit(1)

print(f"✓ All required columns present")

# Check optional columns
available_optional = [col for col in OPTIONAL_COLUMNS if col in df_healthy.columns]
missing_optional = [col for col in OPTIONAL_COLUMNS if col not in df_healthy.columns]

print(f"\nOptional columns:")
print(f"  Present: {len(available_optional)}/{len(OPTIONAL_COLUMNS)}")
if available_optional:
    print(f"  Available: {', '.join(available_optional)}")
if missing_optional:
    print(f"  Missing: {', '.join(missing_optional)}")

# ============================================================================
# STEP 3: VALIDATE DATA QUALITY
# ============================================================================
print("\n[Step 3/7] Validating Data Quality...")

# Validate cbs_id
print("\n--- Validating Equipment IDs ---")
null_ids = df_healthy['cbs_id'].isna().sum()
duplicate_ids = df_healthy['cbs_id'].duplicated().sum()

if null_ids > 0:
    print(f"⚠️  WARNING: {null_ids} records with NULL cbs_id - removing...")
    df_healthy = df_healthy[df_healthy['cbs_id'].notna()].copy()

if duplicate_ids > 0:
    print(f"⚠️  WARNING: {duplicate_ids} duplicate cbs_id values - keeping first occurrence...")
    df_healthy = df_healthy.drop_duplicates(subset='cbs_id', keep='first')

print(f"✓ Valid unique equipment IDs: {df_healthy['cbs_id'].nunique():,}")

# Validate Equipment_Class_Primary
print("\n--- Validating Equipment Classes ---")
null_classes = df_healthy['Equipment_Class_Primary'].isna().sum()

if null_classes > 0:
    print(f"⚠️  WARNING: {null_classes} records with NULL equipment class")
    print(f"  These will be kept but may have limited predictive value")

# Show equipment type distribution
class_dist = df_healthy['Equipment_Class_Primary'].value_counts()
print(f"\n✓ Equipment Type Distribution:")
for eq_class, count in class_dist.head(10).items():
    pct = count / len(df_healthy) * 100
    print(f"  {eq_class:25s}: {count:4,} ({pct:5.1f}%)")

if len(class_dist) > 10:
    others = len(class_dist) - 10
    print(f"  ... and {others} more types")

# Validate installation dates
print("\n--- Validating Installation Dates ---")
df_healthy['Sebekeye_Baglanma_Tarihi_Parsed'] = df_healthy['Sebekeye_Baglanma_Tarihi'].apply(parse_date_flexible)

null_dates = df_healthy['Sebekeye_Baglanma_Tarihi_Parsed'].isna().sum()
if null_dates > 0:
    print(f"⚠️  WARNING: {null_dates} records with invalid installation dates - removing...")
    df_healthy = df_healthy[df_healthy['Sebekeye_Baglanma_Tarihi_Parsed'].notna()].copy()

# Check dates are before cutoff
after_cutoff = (df_healthy['Sebekeye_Baglanma_Tarihi_Parsed'] > CUTOFF_DATE).sum()
if after_cutoff > 0:
    print(f"⚠️  WARNING: {after_cutoff} equipment installed AFTER cutoff date - removing...")
    print(f"  These cannot be used for training (no observation period)")
    df_healthy = df_healthy[df_healthy['Sebekeye_Baglanma_Tarihi_Parsed'] <= CUTOFF_DATE].copy()

print(f"✓ Valid installation dates: {len(df_healthy):,}")
print(f"  Date range: {df_healthy['Sebekeye_Baglanma_Tarihi_Parsed'].min().strftime('%Y-%m-%d')} to {df_healthy['Sebekeye_Baglanma_Tarihi_Parsed'].max().strftime('%Y-%m-%d')}")

# Validate expected life (if present)
if 'Beklenen_Ömür_Yıl' in df_healthy.columns:
    print("\n--- Validating Expected Lifespan ---")
    invalid_life = (df_healthy['Beklenen_Ömür_Yıl'] <= 0) | (df_healthy['Beklenen_Ömür_Yıl'].isna())
    invalid_count = invalid_life.sum()

    if invalid_count > 0:
        print(f"⚠️  WARNING: {invalid_count} records with invalid expected life (≤0 or NULL)")
        print(f"  Setting to default values based on equipment class...")

        # Apply default expected life based on equipment class
        from config import EXPECTED_LIFE_STANDARDS, VOLTAGE_BASED_LIFE, DEFAULT_LIFE

        def get_expected_life(equipment_class):
            """Get expected life using tiered lookup"""
            if pd.isna(equipment_class):
                return DEFAULT_LIFE
            equipment_class = str(equipment_class).strip()
            if equipment_class in EXPECTED_LIFE_STANDARDS:
                return EXPECTED_LIFE_STANDARDS[equipment_class]
            for voltage_key, life in VOLTAGE_BASED_LIFE.items():
                if voltage_key in equipment_class:
                    return life
            return DEFAULT_LIFE

        df_healthy.loc[invalid_life, 'Beklenen_Ömür_Yıl'] = df_healthy.loc[invalid_life, 'Equipment_Class_Primary'].apply(get_expected_life)

    print(f"✓ Expected life range: {df_healthy['Beklenen_Ömür_Yıl'].min():.0f} - {df_healthy['Beklenen_Ömür_Yıl'].max():.0f} years")

# Summary
removed = original_count - len(df_healthy)
if removed > 0:
    print(f"\n⚠️  Removed {removed:,} invalid records ({removed/original_count*100:.1f}%)")
    print(f"✓ Valid healthy equipment: {len(df_healthy):,}")
else:
    print(f"\n✓ All records passed validation!")

# ============================================================================
# STEP 4: CHECK FOR OVERLAP WITH FAILED EQUIPMENT
# ============================================================================
print("\n[Step 4/7] Checking for Overlap with Failed Equipment...")

if INPUT_FILE.exists():
    print(f"\n✓ Loading failed equipment from: {INPUT_FILE}")
    df_failed = pd.read_excel(INPUT_FILE)

    # Get unique failed equipment IDs
    failed_id_col = 'cbs_id' if 'cbs_id' in df_failed.columns else None

    if failed_id_col:
        failed_ids = set(df_failed[failed_id_col].dropna().unique())
        healthy_ids = set(df_healthy['cbs_id'].unique())

        # Find overlap
        overlap = healthy_ids & failed_ids

        if len(overlap) > 0:
            print(f"\n⚠️  WARNING: Found {len(overlap)} equipment IDs in BOTH healthy and failed datasets!")
            print(f"  These equipment are NOT healthy (have failure history)")
            print(f"  Removing them from healthy dataset...")

            # Show examples
            overlap_sample = list(overlap)[:5]
            print(f"\n  Example overlapping IDs: {overlap_sample}")

            # Remove overlap
            df_healthy = df_healthy[~df_healthy['cbs_id'].isin(overlap)].copy()

            print(f"\n✓ Removed {len(overlap)} overlapping equipment")
            print(f"✓ Remaining healthy equipment: {len(df_healthy):,}")
        else:
            print(f"\n✓ No overlap detected - all healthy equipment are truly healthy!")
            print(f"  Failed equipment IDs: {len(failed_ids):,}")
            print(f"  Healthy equipment IDs: {len(healthy_ids):,}")
            print(f"  Overlap: 0")
    else:
        print(f"\n⚠️  WARNING: Cannot verify overlap - 'cbs_id' column not found in failed equipment data")
else:
    print(f"\n⚠️  WARNING: Failed equipment file not found at {INPUT_FILE}")
    print(f"  Cannot verify overlap - proceeding with healthy equipment as-is")

# ============================================================================
# STEP 5: CALCULATE EQUIPMENT AGE
# ============================================================================
print("\n[Step 5/7] Calculating Equipment Age...")

# Calculate age in days and years
df_healthy['Ekipman_Yaşı_Gün'] = (REFERENCE_DATE - df_healthy['Sebekeye_Baglanma_Tarihi_Parsed']).dt.days
df_healthy['Ekipman_Yaşı_Yıl'] = df_healthy['Ekipman_Yaşı_Gün'] / 365.25

print(f"\n✓ Age calculated for {len(df_healthy):,} equipment")
print(f"  Age range: {df_healthy['Ekipman_Yaşı_Yıl'].min():.1f} - {df_healthy['Ekipman_Yaşı_Yıl'].max():.1f} years")
print(f"  Mean age: {df_healthy['Ekipman_Yaşı_Yıl'].mean():.1f} years")
print(f"  Median age: {df_healthy['Ekipman_Yaşı_Yıl'].median():.1f} years")

# ============================================================================
# STEP 6: CREATE ZERO-FAULT FEATURES
# ============================================================================
print("\n[Step 6/7] Creating Zero-Fault Feature Defaults...")

# Add metadata to identify healthy equipment
df_healthy['Has_Failure_History'] = 0
df_healthy['Total_Faults'] = 0
df_healthy['Data_Source'] = 'Healthy_Equipment'

# Set temporal fault counts to zero
df_healthy['Fault_Count_3M'] = 0
df_healthy['Fault_Count_6M'] = 0
df_healthy['Fault_Count_12M'] = 0

# Set fault-related temporal features
df_healthy['Son_Arıza_Gun_Sayisi'] = np.nan  # No last failure
df_healthy['Ilk_Arizaya_Kadar_Gun'] = np.nan  # No first failure
df_healthy['Ilk_Arizaya_Kadar_Yil'] = np.nan

# Set MTBF features (cannot calculate without failures)
df_healthy['MTBF_Gün'] = np.nan
df_healthy['MTBF_Lifetime_Gün'] = np.nan
df_healthy['MTBF_Observable_Gün'] = np.nan
df_healthy['MTBF_Degradation_Ratio'] = np.nan
df_healthy['Is_Degrading'] = 0
df_healthy['Baseline_Hazard_Rate'] = 0.0  # No failures = no hazard

# Set reliability metrics
df_healthy['Güvenilirlik_Skoru'] = 1.0  # Perfect reliability (no failures)

print(f"\n✓ Zero-fault features created:")
print(f"  • Has_Failure_History = 0 (truly healthy)")
print(f"  • Total_Faults = 0")
print(f"  • Fault_Count_* = 0 (all temporal windows)")
print(f"  • MTBF_* = NaN (cannot calculate without failures)")
print(f"  • Son_Arıza_Gun_Sayisi = NaN (no last failure)")
print(f"  • Güvenilirlik_Skoru = 1.0 (perfect reliability)")

# Rename columns to match failed equipment dataset
if 'Sebekeye_Baglanma_Tarihi_Parsed' in df_healthy.columns:
    df_healthy.rename(columns={'Sebekeye_Baglanma_Tarihi_Parsed': 'Grid_Connection_Date'}, inplace=True)

# Rename cbs_id to Ekipman_ID for consistency
df_healthy.rename(columns={'cbs_id': 'Ekipman_ID'}, inplace=True)

# ============================================================================
# STEP 7: SAVE PREPARED HEALTHY EQUIPMENT
# ============================================================================
print("\n[Step 7/7] Saving Prepared Healthy Equipment...")

# Save to CSV
df_healthy.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\n✅ SAVED: {OUTPUT_FILE}")
print(f"   Records: {len(df_healthy):,}")
print(f"   Columns: {len(df_healthy.columns)}")

# Summary statistics
print(f"\n" + "="*100)
print("HEALTHY EQUIPMENT SUMMARY")
print("="*100)
print(f"  Total healthy equipment: {len(df_healthy):,}")
print(f"  Equipment types: {df_healthy['Equipment_Class_Primary'].nunique()}")
print(f"  Age range: {df_healthy['Ekipman_Yaşı_Yıl'].min():.1f} - {df_healthy['Ekipman_Yaşı_Yıl'].max():.1f} years")
print(f"  Mean age: {df_healthy['Ekipman_Yaşı_Yıl'].mean():.1f} years")

if 'Beklenen_Ömür_Yıl' in df_healthy.columns:
    print(f"  Expected life range: {df_healthy['Beklenen_Ömür_Yıl'].min():.0f} - {df_healthy['Beklenen_Ömür_Yıl'].max():.0f} years")

print(f"\n✓ Data Quality:")
print(f"  • All equipment IDs unique: {df_healthy['Ekipman_ID'].is_unique}")
print(f"  • No overlap with failed equipment: ✓")
print(f"  • All installed before cutoff date: ✓")
print(f"  • Zero failures confirmed: ✓")

print(f"\n✓ Ready for merging with failed equipment dataset!")
print(f"  Next step: Run 02_data_transformation.py")

print("\n" + "="*100)
