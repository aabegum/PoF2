"""
FIX FUTURE DATES IN FAULT DATA
Turkish EDAŞ PoF Prediction Project

Purpose:
- Detect and fix dates that are in the future
- Provide options to handle future dates (cap, remove, or adjust)

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

print("="*100)
print(" "*35 + "FUTURE DATE FIX UTILITY")
print("="*100)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING DATA")
print("="*100)

data_file = 'data/combined_data.xlsx'
if not Path(data_file).exists():
    print(f"\n❌ ERROR: Could not find {data_file}")
    exit(1)

print(f"\n✓ Loading: {data_file}")
df = pd.read_excel(data_file)
print(f"✓ Loaded: {len(df):,} records")

# ============================================================================
# STEP 2: IDENTIFY DATE COLUMNS
# ============================================================================
print("\n" + "="*100)
print("STEP 2: IDENTIFYING DATE COLUMNS")
print("="*100)

date_columns = ['started at', 'ended at', 'planned started at']
found_date_cols = [col for col in date_columns if col in df.columns]

if not found_date_cols:
    print("\n❌ ERROR: No date columns found!")
    exit(1)

print(f"\n✓ Found {len(found_date_cols)} date columns:")
for col in found_date_cols:
    print(f"   • {col}")

# ============================================================================
# STEP 3: PARSE AND DIAGNOSE DATES
# ============================================================================
print("\n" + "="*100)
print("STEP 3: DIAGNOSING DATE ISSUES")
print("="*100)

# Define "today" - use a reasonable cutoff date
# Option 1: Use actual today
today = datetime.now()
print(f"\n📅 Reference date (today): {today.strftime('%Y-%m-%d')}")

# Option 2: Use max reasonable date (uncomment if you want to use this)
# today = datetime(2024, 12, 31)  # Adjust as needed
# print(f"\n📅 Reference date (manual): {today.strftime('%Y-%m-%d')}")

issues_found = False
date_info = {}

for col in found_date_cols:
    print(f"\n📊 Analyzing: {col}")

    # Parse dates
    df[col] = pd.to_datetime(df[col], errors='coerce')

    # Get statistics
    valid_dates = df[col].notna()
    min_date = df.loc[valid_dates, col].min()
    max_date = df.loc[valid_dates, col].max()

    # Check for future dates
    future_mask = df[col] > today
    future_count = future_mask.sum()

    print(f"   Valid dates:   {valid_dates.sum():,} / {len(df):,}")
    print(f"   Date range:    {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    if future_count > 0:
        issues_found = True
        print(f"   ⚠️  FUTURE DATES: {future_count:,} records ({future_count/len(df)*100:.1f}%)")

        # Show examples
        future_examples = df.loc[future_mask, col].head(5)
        print(f"   Examples:")
        for date in future_examples:
            days_ahead = (date - today).days
            print(f"      • {date.strftime('%Y-%m-%d')} ({days_ahead} days in future)")

        date_info[col] = {
            'future_count': future_count,
            'max_date': max_date,
            'future_mask': future_mask
        }
    else:
        print(f"   ✓ No future dates")

# ============================================================================
# STEP 4: FIX FUTURE DATES
# ============================================================================
if issues_found:
    print("\n" + "="*100)
    print("STEP 4: FIXING FUTURE DATES")
    print("="*100)

    print("\n🔧 Choose fix strategy:")
    print("   1. CAP to today's date (future dates → today)")
    print("   2. SHIFT back by 1 year (2025-12-02 → 2024-12-02)")
    print("   3. REMOVE rows with future dates")
    print("   4. MANUAL: Assume it's a typo and auto-correct")

    # Auto-detect likely issue
    print("\n🤖 Auto-detecting issue...")

    # Check if all future dates are in 2025 but should be 2024
    all_future_dates = []
    for col, info in date_info.items():
        future_dates = df.loc[info['future_mask'], col]
        all_future_dates.extend(future_dates.tolist())

    if all_future_dates:
        future_years = [d.year for d in all_future_dates if pd.notna(d)]
        if future_years:
            most_common_year = max(set(future_years), key=future_years.count)
            print(f"   Most future dates are in year: {most_common_year}")

            # If most are in 2025, likely should be 2024
            if most_common_year == 2025:
                print(f"   💡 Likely typo: 2025 should be 2024")
                recommended_fix = 2
            else:
                print(f"   💡 Recommend capping to today")
                recommended_fix = 1

    # Apply fix (using recommended strategy)
    fix_strategy = recommended_fix  # Change this to 1, 2, or 3 as needed

    print(f"\n⚙️  Applying fix strategy {fix_strategy}...")

    df_fixed = df.copy()

    for col, info in date_info.items():
        future_mask = info['future_mask']

        if fix_strategy == 1:
            # Cap to today
            df_fixed.loc[future_mask, col] = today
            print(f"   ✓ {col}: Capped {future_mask.sum()} dates to {today.strftime('%Y-%m-%d')}")

        elif fix_strategy == 2:
            # Shift back 1 year
            df_fixed.loc[future_mask, col] = df_fixed.loc[future_mask, col] - pd.DateOffset(years=1)
            print(f"   ✓ {col}: Shifted {future_mask.sum()} dates back by 1 year")

        elif fix_strategy == 3:
            # Remove rows
            df_fixed = df_fixed[~future_mask]
            print(f"   ✓ {col}: Removed {future_mask.sum()} rows")

        elif fix_strategy == 4:
            # Auto-correct based on pattern
            # If year is 2025, change to 2024
            for idx in df_fixed[future_mask].index:
                old_date = df_fixed.loc[idx, col]
                if pd.notna(old_date) and old_date.year == 2025:
                    new_date = old_date.replace(year=2024)
                    df_fixed.loc[idx, col] = new_date
            print(f"   ✓ {col}: Auto-corrected {future_mask.sum()} dates (2025→2024)")

    # ============================================================================
    # STEP 5: VALIDATE FIX
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 5: VALIDATING FIX")
    print("="*100)

    all_fixed = True
    for col in found_date_cols:
        future_after_fix = (df_fixed[col] > today).sum()
        print(f"\n✓ {col}:")
        print(f"   Before: {date_info.get(col, {}).get('future_count', 0)} future dates")
        print(f"   After:  {future_after_fix} future dates")

        if future_after_fix > 0:
            all_fixed = False

    if all_fixed:
        print("\n✅ All future dates successfully fixed!")
    else:
        print("\n⚠️  Some future dates remain - may need manual review")

    # ============================================================================
    # STEP 6: SAVE FIXED DATA
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 6: SAVING FIXED DATA")
    print("="*100)

    # Create backup
    backup_file = 'data/combined_data_BACKUP.xlsx'
    print(f"\n💾 Creating backup: {backup_file}")
    df.to_excel(backup_file, index=False)
    print(f"✓ Backup saved: {len(df):,} records")

    # Save fixed data
    fixed_file = 'data/combined_data_FIXED.xlsx'
    print(f"\n💾 Saving fixed data: {fixed_file}")
    df_fixed.to_excel(fixed_file, index=False)
    print(f"✓ Fixed data saved: {len(df_fixed):,} records")

    # Optionally overwrite original (uncomment to enable)
    # print(f"\n💾 Overwriting original: {data_file}")
    # df_fixed.to_excel(data_file, index=False)
    # print(f"✓ Original file updated")

    print("\n" + "="*100)
    print("📋 SUMMARY")
    print("="*100)
    print(f"\nOriginal records:  {len(df):,}")
    print(f"Fixed records:     {len(df_fixed):,}")
    print(f"Records removed:   {len(df) - len(df_fixed):,}")
    print(f"\n✓ Backup saved to: {backup_file}")
    print(f"✓ Fixed data at:   {fixed_file}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the fixed data: {fixed_file}")
    print(f"   2. If satisfied, manually rename {fixed_file} to {data_file}")
    print(f"   3. Re-run 00_temporal_diagnostic.py to verify")

else:
    print("\n" + "="*100)
    print("✅ NO ISSUES FOUND - All dates are valid!")
    print("="*100)

print("\n" + "="*100)
print(f"{'FUTURE DATE FIX COMPLETE':^100}")
print("="*100)
