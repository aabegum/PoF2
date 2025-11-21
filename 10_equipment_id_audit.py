"""
EQUIPMENT ID MISMATCH DIAGNOSTIC (CRITICAL)
============================================
Checks for ID inconsistencies across pipeline that could cause:
- Targets assigned to wrong equipment
- Features and targets misaligned
- Performance metrics meaningless

CHECKS:
1. cbs_id vs Ekipman_ID consistency
2. ID overlap between fault data and feature data
3. Missing/orphaned equipment IDs
4. ID normalization requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("EQUIPMENT ID MISMATCH DIAGNOSTIC")
print("="*100)

# ============================================================================
# STEP 1: LOAD ALL DATA SOURCES
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING DATA FROM ALL SOURCES")
print("="*100)

# Raw fault data (source of truth)
print("\n1. Raw Fault Data (combined_data.xlsx):")
faults = pd.read_excel('data/combined_data.xlsx')
print(f"   Total faults: {len(faults):,}")
print(f"   Columns with 'ID' or 'id': {[col for col in faults.columns if 'id' in col.lower() or 'ID' in col]}")

# Equipment-level data (after Step 2)
print("\n2. Equipment-Level Data (equipment_level_data.csv):")
equipment = pd.read_csv('data/equipment_level_data.csv')
print(f"   Total equipment: {len(equipment):,}")
print(f"   ID column: {[col for col in equipment.columns if 'ID' in col or col == 'Ekipman_ID']}")

# Reduced features (after Step 5)
print("\n3. Reduced Features (features_reduced.csv):")
try:
    features = pd.read_csv('data/features_reduced.csv')
    print(f"   Total equipment: {len(features):,}")
    print(f"   ID column: {[col for col in features.columns if 'ID' in col or col == 'Ekipman_ID']}")
except FileNotFoundError:
    print("   ⚠️  File not found (run feature selection first)")
    features = None

# ============================================================================
# STEP 2: CHECK cbs_id AVAILABILITY
# ============================================================================
print("\n" + "="*100)
print("STEP 2: CBS_ID AVAILABILITY IN RAW DATA")
print("="*100)

if 'cbs_id' in faults.columns:
    total_faults = len(faults)
    has_cbs_id = faults['cbs_id'].notna().sum()
    missing_cbs_id = faults['cbs_id'].isna().sum()
    unique_cbs_ids = faults['cbs_id'].nunique()

    print(f"\n📊 cbs_id Statistics:")
    print(f"   Total faults: {total_faults:,}")
    print(f"   Has cbs_id: {has_cbs_id:,} ({has_cbs_id/total_faults*100:.1f}%)")
    print(f"   Missing cbs_id: {missing_cbs_id:,} ({missing_cbs_id/total_faults*100:.1f}%)")
    print(f"   Unique cbs_ids: {unique_cbs_ids:,}")

    if missing_cbs_id > 0:
        print(f"\n   ⚠️  {missing_cbs_id} faults missing cbs_id!")
        print(f"      These may have been assigned fallback IDs in Step 2")
else:
    print("\n   ❌ CRITICAL: No 'cbs_id' column found in raw data!")
    print("      Check column names in combined_data.xlsx")

# ============================================================================
# STEP 3: CHECK Ekipman_ID IN PROCESSED DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 3: EKIPMAN_ID IN PROCESSED DATA")
print("="*100)

if 'Ekipman_ID' in equipment.columns:
    unique_ekipman_ids = equipment['Ekipman_ID'].nunique()
    print(f"\n📊 Ekipman_ID Statistics:")
    print(f"   Total equipment records: {len(equipment):,}")
    print(f"   Unique Ekipman_IDs: {unique_ekipman_ids:,}")

    # Check for duplicates
    if unique_ekipman_ids < len(equipment):
        print(f"   ⚠️  WARNING: {len(equipment) - unique_ekipman_ids} duplicate Ekipman_IDs!")
    else:
        print(f"   ✓ No duplicates (each equipment has unique ID)")

    # Check ID format
    sample_ids = equipment['Ekipman_ID'].head(10).tolist()
    print(f"\n   Sample Ekipman_IDs: {sample_ids[:5]}")

    # Check for generated IDs (fallback)
    generated_ids = equipment[equipment['Ekipman_ID'].astype(str).str.contains('UNKNOWN', na=False)]
    if len(generated_ids) > 0:
        print(f"\n   ⚠️  {len(generated_ids)} equipment have GENERATED IDs (UNKNOWN_XXX)")
        print(f"      These had missing cbs_id and fallback IDs")
else:
    print("\n   ❌ CRITICAL: No 'Ekipman_ID' column found!")

# ============================================================================
# STEP 4: CHECK ID OVERLAP
# ============================================================================
print("\n" + "="*100)
print("STEP 4: ID OVERLAP CHECK (cbs_id vs Ekipman_ID)")
print("="*100)

if 'cbs_id' in faults.columns and 'Ekipman_ID' in equipment.columns:
    # Get unique IDs from each source
    cbs_ids = set(faults['cbs_id'].dropna().unique())
    ekipman_ids = set(equipment['Ekipman_ID'].unique())

    # Calculate overlap
    overlap = cbs_ids & ekipman_ids
    cbs_only = cbs_ids - ekipman_ids
    ekipman_only = ekipman_ids - cbs_ids

    print(f"\n📊 ID Overlap Analysis:")
    print(f"   cbs_ids (from faults):      {len(cbs_ids):,}")
    print(f"   Ekipman_IDs (from equipment): {len(ekipman_ids):,}")
    print(f"   Overlap (in both):          {len(overlap):,}")
    print(f"   Only in faults (cbs_id):    {len(cbs_only):,}")
    print(f"   Only in equipment:          {len(ekipman_only):,}")

    # Calculate match percentage
    if len(cbs_ids) > 0:
        match_pct = len(overlap) / len(cbs_ids) * 100
        print(f"\n   Match Rate: {match_pct:.1f}% of cbs_ids found in Ekipman_IDs")

        if match_pct < 95:
            print(f"   ❌ CRITICAL: Only {match_pct:.1f}% match!")
            print(f"      Target creation uses cbs_id, but features use Ekipman_ID")
            print(f"      This means targets are assigned to WRONG equipment!")
        elif match_pct < 100:
            print(f"   ⚠️  WARNING: {100-match_pct:.1f}% mismatch")
        else:
            print(f"   ✓ Perfect match (100%)")

    # Check if Ekipman_only are UNKNOWN IDs
    if len(ekipman_only) > 0:
        ekipman_only_list = list(ekipman_only)[:20]
        unknown_count = sum(1 for id in ekipman_only_list if 'UNKNOWN' in str(id))
        print(f"\n   Equipment-only IDs analysis:")
        print(f"      Total equipment-only IDs: {len(ekipman_only):,}")
        print(f"      UNKNOWN (generated) IDs: {unknown_count} of first 20")
        print(f"      Sample equipment-only IDs: {ekipman_only_list[:5]}")

# ============================================================================
# STEP 5: CHECK TARGET CREATION ALIGNMENT
# ============================================================================
print("\n" + "="*100)
print("STEP 5: TARGET CREATION ALIGNMENT CHECK")
print("="*100)

print("\n🔍 Checking 06_temporal_pof_model.py logic:")

# Simulate what target creation does
from config import CUTOFF_DATE
faults['started at'] = pd.to_datetime(faults['started at'], dayfirst=True, errors='coerce')

FUTURE_6M_END = CUTOFF_DATE + pd.DateOffset(months=6)

# This is what the current code does (WRONG)
future_faults_6M_cbs = faults[
    (faults['started at'] > CUTOFF_DATE) &
    (faults['started at'] <= FUTURE_6M_END)
]['cbs_id'].dropna().unique()

print(f"\n   Current Logic (uses cbs_id):")
print(f"      Equipment that will fail in 6M: {len(future_faults_6M_cbs):,}")

# Check how many of these cbs_ids exist in Ekipman_ID
if 'Ekipman_ID' in equipment.columns:
    valid_in_equipment = set(future_faults_6M_cbs) & set(equipment['Ekipman_ID'].unique())
    print(f"      Found in Ekipman_ID: {len(valid_in_equipment):,} ({len(valid_in_equipment)/len(future_faults_6M_cbs)*100:.1f}%)")

    missing = len(future_faults_6M_cbs) - len(valid_in_equipment)
    if missing > 0:
        print(f"      ❌ MISSING: {missing} equipment IDs NOT in feature data!")
        print(f"         These will have targets=0 (wrong!) because .isin() returns False")

# ============================================================================
# STEP 6: RECOMMENDATIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 6: RECOMMENDATIONS")
print("="*100)

print("\n💡 SOLUTIONS:")

if 'cbs_id' in faults.columns and 'Ekipman_ID' in equipment.columns:
    match_pct = len(overlap) / len(cbs_ids) * 100 if len(cbs_ids) > 0 else 0

    if match_pct < 95:
        print("\n❌ CRITICAL MISMATCH DETECTED!")
        print("\n1. IMMEDIATE FIX REQUIRED:")
        print("   Current: Target creation uses 'cbs_id' from raw fault data")
        print("   Problem: Feature data uses 'Ekipman_ID' (may be different)")
        print("   Fix: Update 06_temporal_pof_model.py to:")
        print("        - Load equipment_level_data.csv to get Ekipman_ID → cbs_id mapping")
        print("        - Use Ekipman_ID consistently throughout target creation")
        print("\n2. ADD ID MAPPING:")
        print("   Create mapping table: cbs_id → Ekipman_ID")
        print("   Include in equipment_level_data.csv output")
        print("\n3. VALIDATE:")
        print("   Re-run this script after fix to confirm 100% match")

    elif match_pct < 100:
        print("\n⚠️  MINOR MISMATCH DETECTED")
        print(f"   {100-match_pct:.1f}% of IDs don't match")
        print("   These are likely equipment with missing cbs_id (assigned UNKNOWN_XXX)")
        print("   Action: Document that these equipment are excluded from modeling")

    else:
        print("\n✅ NO MISMATCH DETECTED")
        print("   cbs_id and Ekipman_ID have 100% overlap")
        print("   Target creation is correctly aligned")
        print("   No fixes needed")

print("\n" + "="*100)
print("EQUIPMENT ID AUDIT COMPLETE")
print("="*100)

# Save mapping table if IDs match
if 'cbs_id' in faults.columns and 'Ekipman_ID' in equipment.columns:
    if len(overlap) > 0:
        # Create mapping table
        mapping_df = equipment[['Ekipman_ID']].copy()
        mapping_df['cbs_id'] = mapping_df['Ekipman_ID']  # They're the same if they overlap

        Path('data').mkdir(exist_ok=True)
        mapping_df.to_csv('data/equipment_id_mapping.csv', index=False)
        print("\n💾 Saved ID mapping to: data/equipment_id_mapping.csv")
