"""
DIAGNOSTIC SCRIPT - Investigate Data Issues
============================================
This script investigates why we have 4,290 equipment and 0 targets.
"""

import pandas as pd
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("="*80)
print("DATA DIAGNOSTIC - Investigating 4,290 Equipment Issue")
print("="*80)

# ============================================================================
# CHECK 1: Equipment File Analysis
# ============================================================================
print("\n[CHECK 1] Equipment File Analysis")
print("-"*80)

try:
    equip = pd.read_csv('data/equipment_level_data.csv')

    print(f"Total rows: {len(equip):,}")
    print(f"Unique Ekipman_ID: {equip['Ekipman_ID'].nunique():,}")
    print(f"Has duplicate IDs: {equip['Ekipman_ID'].duplicated().any()}")

    if equip['Ekipman_ID'].duplicated().any():
        dup_count = equip['Ekipman_ID'].duplicated().sum()
        print(f"  [!] WARNING: {dup_count:,} duplicate Ekipman_IDs found!")
        print("\n  Sample duplicate IDs:")
        dups = equip[equip['Ekipman_ID'].duplicated(keep=False)]['Ekipman_ID'].value_counts().head()
        for id_val, count in dups.items():
            print(f"    ID {id_val}: appears {count} times")

    print(f"\nSample IDs (first 10):")
    for i, id_val in enumerate(equip['Ekipman_ID'].head(10), 1):
        print(f"  {i}. {id_val} (type: {type(id_val).__name__})")

    # Check for generated IDs
    generated = equip[equip['Ekipman_ID'].astype(str).str.contains('UNKNOWN', na=False)]
    print(f"\nGenerated UNKNOWN_XXX IDs: {len(generated):,}")

    if len(generated) > 0:
        print("  Sample generated IDs:")
        for id_val in generated['Ekipman_ID'].head(5):
            print(f"    {id_val}")

    # Fault history
    print(f"\nFault Count Distribution:")
    if 'Toplam_Arıza_Sayisi_Lifetime' in equip.columns:
        fault_counts = equip['Toplam_Arıza_Sayisi_Lifetime']
        print(f"  Min: {fault_counts.min()}")
        print(f"  Max: {fault_counts.max()}")
        print(f"  Mean: {fault_counts.mean():.2f}")
        print(f"  Median: {fault_counts.median():.1f}")

        # Equipment with 0 faults
        zero_faults = (fault_counts == 0).sum()
        if zero_faults > 0:
            print(f"\n  [!] PROBLEM: {zero_faults:,} equipment have ZERO faults!")
            print(f"     This explains the high equipment count (4,290)")
            print(f"     Equipment file should only contain equipment WITH faults")
    else:
        print("  [X] Column 'Toplam_Arıza_Sayisi_Lifetime' not found!")

except FileNotFoundError:
    print("[X] File not found: data/equipment_level_data.csv")
except Exception as e:
    print(f"[X] Error: {e}")

# ============================================================================
# CHECK 2: Input File Analysis
# ============================================================================
print("\n\n[CHECK 2] Input File Analysis")
print("-"*80)

try:
    from config import INPUT_FILE

    print(f"Reading: {INPUT_FILE}")
    faults = pd.read_excel(INPUT_FILE)

    print(f"Total rows: {len(faults):,}")
    print(f"Total columns: {len(faults.columns)}")

    # Check for cbs_id
    if 'cbs_id' in faults.columns:
        print(f"\ncbs_id column:")
        print(f"  Total values: {faults['cbs_id'].notna().sum():,}")
        print(f"  Missing (NaN): {faults['cbs_id'].isna().sum():,}")
        print(f"  Unique IDs: {faults['cbs_id'].nunique():,}")

        print(f"\n  Sample cbs_id values (first 10):")
        for i, id_val in enumerate(faults['cbs_id'].dropna().head(10), 1):
            print(f"    {i}. {id_val} (type: {type(id_val).__name__})")
    else:
        print("[X] No 'cbs_id' column found!")
        print("   Available ID columns:")
        id_cols = [col for col in faults.columns if 'id' in col.lower() or 'ID' in col]
        for col in id_cols[:10]:
            print(f"     - {col}")

    # Check for date columns
    print(f"\nDate columns check:")
    if 'Sebekeye_Baglanma_Tarihi' in faults.columns:
        print(f"  [OK] Sebekeye_Baglanma_Tarihi: {faults['Sebekeye_Baglanma_Tarihi'].notna().sum():,} values")
    else:
        print(f"  [X] Sebekeye_Baglanma_Tarihi: NOT FOUND")
        # Look for alternatives
        date_cols = [col for col in faults.columns if any(
            keyword in col.upper() for keyword in ['TARIH', 'DATE', 'BAGLAN', 'KURULUM', 'TESIS']
        )]
        if date_cols:
            print(f"     Potential date columns found:")
            for col in date_cols[:5]:
                print(f"       - {col}")

    if 'started at' in faults.columns:
        started = pd.to_datetime(faults['started at'], errors='coerce')
        print(f"  [OK] started at: {started.notna().sum():,} valid dates")
        print(f"       Date range: {started.min()} to {started.max()}")
    else:
        print(f"  [X] started at: NOT FOUND")

except Exception as e:
    print(f"[X] Error reading input file: {e}")

# ============================================================================
# CHECK 3: ID Type Comparison
# ============================================================================
print("\n\n[CHECK 3] ID Type and Format Comparison")
print("-"*80)

try:
    # Compare ID types
    equip_ids = equip['Ekipman_ID'].dropna()
    fault_ids = faults['cbs_id'].dropna() if 'cbs_id' in faults.columns else pd.Series([])

    print(f"Equipment IDs:")
    print(f"  Count: {len(equip_ids):,}")
    print(f"  Dtype: {equip_ids.dtype}")
    print(f"  Sample: {equip_ids.iloc[0]} (type: {type(equip_ids.iloc[0]).__name__})")

    if len(fault_ids) > 0:
        print(f"\nFault cbs_id:")
        print(f"  Count: {len(fault_ids):,}")
        print(f"  Dtype: {fault_ids.dtype}")
        print(f"  Sample: {fault_ids.iloc[0]} (type: {type(fault_ids.iloc[0]).__name__})")

        # Check overlap
        equip_set = set(equip_ids.astype(float))
        fault_set = set(fault_ids.astype(float))

        overlap = len(equip_set & fault_set)
        match_pct = (overlap / len(equip_set)) * 100 if len(equip_set) > 0 else 0

        print(f"\nID Overlap:")
        print(f"  Equipment IDs: {len(equip_set):,}")
        print(f"  Fault IDs: {len(fault_set):,}")
        print(f"  Matching: {overlap:,} ({match_pct:.1f}%)")

        if match_pct < 95:
            print(f"\n  [!] WARNING: Low match rate!")
            print(f"      This will cause target creation to fail")

except Exception as e:
    print(f"[X] Error in ID comparison: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

print("\nPROBLEMS FOUND:")
problems = []

try:
    if len(equip) > 2000:
        problems.append(f"1. Too many equipment records: {len(equip):,} (expected ~700-1,500)")
        problems.append("   -> Likely cause: Input file contains equipment WITHOUT faults")

    if 'Toplam_Arıza_Sayisi_Lifetime' in equip.columns:
        zero_faults = (equip['Toplam_Arıza_Sayisi_Lifetime'] == 0).sum()
        if zero_faults > 0:
            problems.append(f"2. Equipment with 0 faults: {zero_faults:,}")
            problems.append("   -> These should be filtered out in Step 2")

    if 'Sebekeye_Baglanma_Tarihi' not in faults.columns:
        problems.append("3. Missing required column: Sebekeye_Baglanma_Tarihi")
        problems.append("   -> Step 2 will fail without this column")

    if match_pct < 95:
        problems.append(f"4. Low ID match rate: {match_pct:.1f}% (expected >95%)")
        problems.append("   -> Target creation will assign 0 targets (all equipment = class 0)")
        problems.append("   -> XGBoost will fail with base_score=0 error")

    if len(problems) == 0:
        print("  [OK] No major problems detected!")
    else:
        for problem in problems:
            print(f"  {problem}")

    print("\n" + "="*80)
    print("RECOMMENDED ACTIONS:")
    print("="*80)

    if len(equip) > 2000:
        print("\n1. CHECK YOUR INPUT FILE:")
        print("   - Open data/combined_data_son.xlsx")
        print("   - Verify it contains ONLY fault records")
        print("   - Should NOT contain equipment master list")
        print("   - Each row = one fault event")

    if 'Sebekeye_Baglanma_Tarihi' not in faults.columns:
        print("\n2. ADD REQUIRED COLUMN:")
        print("   - Add 'Sebekeye_Baglanma_Tarihi' column to input file")
        print("   - OR tell me the actual column name for grid connection date")

    if match_pct < 95:
        print("\n3. FIX ID MISMATCH:")
        print("   - Verify cbs_id column is populated")
        print("   - Check if IDs are in same format (all integers, no strings)")
        print("   - Ensure no data type mismatches")

except Exception as e:
    print(f"[X] Error in summary: {e}")

print("\n" + "="*80)
