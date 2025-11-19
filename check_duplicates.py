"""
Quick script to check for duplicate faults in combined_data.xlsx

Since we're combining TESIS, EDBS, and WORKORDER sources, the same fault
might appear multiple times, causing fault count inflation.
"""

import pandas as pd
import numpy as np

print("="*80)
print("DUPLICATE DETECTION CHECK - combined_data.xlsx")
print("="*80)

# Load data
df = pd.read_excel('data/combined_data.xlsx')
print(f"\nTotal records: {len(df):,}")

# Try to identify equipment ID column
equip_id_cols = ['cbs_id', 'Ekipman Kodu', 'Ekipman ID', 'HEPSI_ID']
equip_id_col = next((col for col in equip_id_cols if col in df.columns), None)

if not equip_id_col:
    print("\n⚠️  WARNING: No equipment ID column found!")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

print(f"\nUsing equipment ID column: {equip_id_col}")

# Parse dates
from datetime import datetime

def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    try:
        return pd.to_datetime(value, format='%d-%m-%Y %H:%M:%S', errors='coerce')
    except:
        try:
            return pd.to_datetime(value, errors='coerce')
        except:
            return pd.NaT

df['started_at_parsed'] = df['started at'].apply(parse_date)
df['ended_at_parsed'] = df['ended at'].apply(parse_date)

print(f"Parsed dates: {df['started_at_parsed'].notna().sum():,} start times, {df['ended_at_parsed'].notna().sum():,} end times")

# ============================================================================
# CHECK 1: Exact duplicates (all columns identical)
# ============================================================================
print("\n" + "-"*80)
print("CHECK 1: Exact Duplicates (All Columns Identical)")
print("-"*80)

exact_dups = df.duplicated(keep=False)
exact_dup_count = exact_dups.sum()

if exact_dup_count > 0:
    print(f"❌ FOUND {exact_dup_count:,} exact duplicate records ({exact_dup_count/len(df)*100:.1f}%)")
    print("\nSample exact duplicates:")
    print(df[exact_dups].head(3)[[equip_id_col, 'started at', 'ended at']].to_string())
else:
    print("✅ No exact duplicates found")

# ============================================================================
# CHECK 2: Same equipment + same start time (likely same fault)
# ============================================================================
print("\n" + "-"*80)
print("CHECK 2: Same Equipment + Same Start Time")
print("-"*80)

df_with_time = df[df['started_at_parsed'].notna()].copy()
time_dups = df_with_time.duplicated(subset=[equip_id_col, 'started_at_parsed'], keep=False)
time_dup_count = time_dups.sum()

if time_dup_count > 0:
    print(f"❌ FOUND {time_dup_count:,} records with same equipment+start time ({time_dup_count/len(df)*100:.1f}%)")
    print("\nSample same-time duplicates:")
    sample_dups = df_with_time[time_dups].sort_values([equip_id_col, 'started_at_parsed']).head(6)
    print(sample_dups[[equip_id_col, 'started at', 'ended at', 'Fault Description' if 'Fault Description' in df.columns else 'Neden Kodu']].to_string())
else:
    print("✅ No same-time duplicates found")

# ============================================================================
# CHECK 3: Same equipment + overlapping time windows
# ============================================================================
print("\n" + "-"*80)
print("CHECK 3: Same Equipment + Overlapping Time Windows")
print("-"*80)

df_with_both = df[(df['started_at_parsed'].notna()) & (df['ended_at_parsed'].notna())].copy()
df_with_both = df_with_both.sort_values([equip_id_col, 'started_at_parsed'])

overlap_count = 0
overlap_pairs = []

for equip_id, group in df_with_both.groupby(equip_id_col):
    if len(group) < 2:
        continue

    group_sorted = group.sort_values('started_at_parsed')
    for i in range(len(group_sorted) - 1):
        row1 = group_sorted.iloc[i]
        row2 = group_sorted.iloc[i+1]

        # Check if overlapping: fault1 end > fault2 start
        if row1['ended_at_parsed'] > row2['started_at_parsed']:
            overlap_count += 1
            if len(overlap_pairs) < 3:  # Save first 3 examples
                overlap_pairs.append((row1, row2))

if overlap_count > 0:
    print(f"⚠️  FOUND {overlap_count:,} overlapping time windows (might be same fault reported in multiple sources)")
    print("\nSample overlapping faults:")
    for i, (r1, r2) in enumerate(overlap_pairs, 1):
        print(f"\n  Pair {i}: {r1[equip_id_col]}")
        print(f"    Fault 1: {r1['started at']} → {r1['ended at']}")
        print(f"    Fault 2: {r2['started at']} → {r2['ended at']}")
else:
    print("✅ No overlapping time windows found")

# ============================================================================
# CHECK 4: Per-equipment fault distribution
# ============================================================================
print("\n" + "-"*80)
print("CHECK 4: Equipment with Suspiciously High Fault Counts")
print("-"*80)

fault_counts = df.groupby(equip_id_col).size().sort_values(ascending=False)

print(f"Total equipment: {len(fault_counts):,}")
print(f"Mean faults per equipment: {fault_counts.mean():.1f}")
print(f"Median faults per equipment: {fault_counts.median():.1f}")
print(f"\nTop 10 equipment by fault count:")
print(fault_counts.head(10).to_string())

# Flag suspicious (>3 standard deviations above mean)
mean_faults = fault_counts.mean()
std_faults = fault_counts.std()
threshold = mean_faults + 3 * std_faults
suspicious = fault_counts[fault_counts > threshold]

if len(suspicious) > 0:
    print(f"\n⚠️  {len(suspicious)} equipment with >3 std above mean (>{threshold:.1f} faults):")
    print(suspicious.to_string())
else:
    print(f"\n✅ No equipment with abnormally high fault counts (threshold: {threshold:.1f})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

issues_found = 0
if exact_dup_count > 0:
    print(f"❌ Exact duplicates: {exact_dup_count:,} records ({exact_dup_count/len(df)*100:.1f}%)")
    issues_found += 1

if time_dup_count > 0:
    print(f"❌ Same equipment+time duplicates: {time_dup_count:,} records ({time_dup_count/len(df)*100:.1f}%)")
    issues_found += 1

if overlap_count > 0:
    print(f"⚠️  Overlapping time windows: {overlap_count:,} pairs")
    issues_found += 1

if len(suspicious) > 0:
    print(f"⚠️  Suspicious high fault counts: {len(suspicious)} equipment")
    issues_found += 1

if issues_found == 0:
    print("✅ No obvious duplicate issues detected")
else:
    print(f"\n⚠️  Total issues found: {issues_found}")
    print("\n💡 RECOMMENDATION: Add duplicate detection to 02_data_transformation.py")
    print("   Suggested approach:")
    print("   1. Remove exact duplicates: df.drop_duplicates()")
    print("   2. Remove same equipment+time: df.drop_duplicates(subset=[equipment_id, 'started at'])")
    print("   3. Consider merging overlapping faults (same equipment, overlapping times)")

print("\n" + "="*80)
