"""
FAULT TIMESTAMP QUALITY CHECK
Turkish EDAŞ PoF Prediction Project

Purpose:
- Deep dive analysis of 'started at' and 'ended at' columns
- Check data quality, coverage, and usability for temporal PoF modeling
- Identify repair duration patterns and anomalies

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

print("="*100)
print(" "*30 + "FAULT TIMESTAMP QUALITY CHECK")
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
print(f"✓ Loaded: {len(df):,} fault records")

# ============================================================================
# STEP 2: PARSE TIMESTAMP COLUMNS
# ============================================================================
print("\n" + "="*100)
print("STEP 2: PARSING TIMESTAMP COLUMNS")
print("="*100)

# Check if columns exist
if 'started at' not in df.columns or 'ended at' not in df.columns:
    print("\n❌ ERROR: 'started at' or 'ended at' columns not found!")
    print(f"Available columns: {list(df.columns[:20])}")
    exit(1)

print("\n✓ Found both columns:")
print(f"   • started at")
print(f"   • ended at")

# Parse dates
df['started at'] = pd.to_datetime(df['started at'], errors='coerce')
df['ended at'] = pd.to_datetime(df['ended at'], errors='coerce')

# ============================================================================
# STEP 3: COVERAGE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 3: TIMESTAMP COVERAGE ANALYSIS")
print("="*100)

total_records = len(df)

has_started = df['started at'].notna()
has_ended = df['ended at'].notna()
has_both = has_started & has_ended
has_neither = ~has_started & ~has_ended
has_only_started = has_started & ~has_ended
has_only_ended = ~has_started & has_ended

print(f"\n📊 COVERAGE BREAKDOWN:")
print(f"   Total records:           {total_records:,}")
print(f"   ")
print(f"   ✅ Has BOTH timestamps:   {has_both.sum():,} ({has_both.sum()/total_records*100:5.1f}%)")
print(f"   ⚠  Has ONLY 'started at': {has_only_started.sum():,} ({has_only_started.sum()/total_records*100:5.1f}%)")
print(f"   ⚠  Has ONLY 'ended at':   {has_only_ended.sum():,} ({has_only_ended.sum()/total_records*100:5.1f}%)")
print(f"   ❌ Has NEITHER:           {has_neither.sum():,} ({has_neither.sum()/total_records*100:5.1f}%)")

# ============================================================================
# STEP 4: TEMPORAL COVERAGE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 4: TEMPORAL RANGE ANALYSIS")
print("="*100)

if has_started.sum() > 0:
    started_min = df.loc[has_started, 'started at'].min()
    started_max = df.loc[has_started, 'started at'].max()
    started_span = (started_max - started_min).days

    print(f"\n📅 'started at' RANGE:")
    print(f"   Earliest: {started_min.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Latest:   {started_max.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Span:     {started_span:,} days ({started_span/365:.1f} years)")

if has_ended.sum() > 0:
    ended_min = df.loc[has_ended, 'ended at'].min()
    ended_max = df.loc[has_ended, 'ended at'].max()
    ended_span = (ended_max - ended_min).days

    print(f"\n📅 'ended at' RANGE:")
    print(f"   Earliest: {ended_min.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Latest:   {ended_max.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Span:     {ended_span:,} days ({ended_span/365:.1f} years)")

# ============================================================================
# STEP 5: REPAIR DURATION ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 5: REPAIR DURATION ANALYSIS")
print("="*100)

if has_both.sum() > 0:
    df.loc[has_both, 'repair_duration_hours'] = (
        (df.loc[has_both, 'ended at'] - df.loc[has_both, 'started at']).dt.total_seconds() / 3600
    )

    durations = df.loc[has_both, 'repair_duration_hours']

    print(f"\n⏱️  REPAIR DURATION STATISTICS (for {has_both.sum():,} faults with both timestamps):")
    print(f"   Mean:     {durations.mean():8.2f} hours")
    print(f"   Median:   {durations.median():8.2f} hours")
    print(f"   Min:      {durations.min():8.2f} hours")
    print(f"   Max:      {durations.max():8.2f} hours")
    print(f"   Std Dev:  {durations.std():8.2f} hours")

    # Check for anomalies
    negative_duration = durations < 0
    zero_duration = durations == 0
    very_short = (durations > 0) & (durations < 0.1)  # Less than 6 minutes
    very_long = durations > 24 * 7  # More than 7 days

    print(f"\n⚠️  DURATION ANOMALIES:")
    print(f"   Negative (ended before started): {negative_duration.sum():,} ({negative_duration.sum()/has_both.sum()*100:5.1f}%)")
    print(f"   Zero (instant repair):           {zero_duration.sum():,} ({zero_duration.sum()/has_both.sum()*100:5.1f}%)")
    print(f"   Very short (<6 min):             {very_short.sum():,} ({very_short.sum()/has_both.sum()*100:5.1f}%)")
    print(f"   Very long (>7 days):             {very_long.sum():,} ({very_long.sum()/has_both.sum()*100:5.1f}%)")

    # Show examples of anomalies
    if negative_duration.sum() > 0:
        print(f"\n   Examples of NEGATIVE durations:")
        neg_examples = df.loc[has_both & negative_duration, ['id', 'started at', 'ended at', 'repair_duration_hours']].head(3)
        for idx, row in neg_examples.iterrows():
            print(f"      ID {row['id']}: Started {row['started at']} → Ended {row['ended at']} ({row['repair_duration_hours']:.2f} hrs)")

    if very_long.sum() > 0:
        print(f"\n   Examples of VERY LONG durations (>7 days):")
        long_examples = df.loc[has_both & very_long, ['id', 'started at', 'ended at', 'repair_duration_hours']].head(3)
        for idx, row in long_examples.iterrows():
            print(f"      ID {row['id']}: {row['repair_duration_hours']:.1f} hours ({row['repair_duration_hours']/24:.1f} days)")

    # Duration distribution
    print(f"\n📊 DURATION DISTRIBUTION:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = durations.quantile(p/100)
        print(f"   {p:2d}th percentile: {val:8.2f} hours ({val/24:6.2f} days)")

else:
    print("\n⚠️  Cannot analyze repair duration - no records with both timestamps")

# ============================================================================
# STEP 6: FUTURE DATE CHECK
# ============================================================================
print("\n" + "="*100)
print("STEP 6: FUTURE DATE CHECK")
print("="*100)

today = datetime.now()
print(f"\n📅 Reference date (today): {today.strftime('%Y-%m-%d')}")

if has_started.sum() > 0:
    future_started = df.loc[has_started, 'started at'] > today
    print(f"\n'started at' FUTURE DATES:")
    print(f"   Future dates: {future_started.sum():,} ({future_started.sum()/has_started.sum()*100:5.1f}%)")

    if future_started.sum() > 0:
        future_years = df.loc[has_started & future_started, 'started at'].dt.year.value_counts().sort_index()
        print(f"   Future years:")
        for year, count in future_years.items():
            print(f"      {year}: {count:,} records")

if has_ended.sum() > 0:
    future_ended = df.loc[has_ended, 'ended at'] > today
    print(f"\n'ended at' FUTURE DATES:")
    print(f"   Future dates: {future_ended.sum():,} ({future_ended.sum()/has_ended.sum()*100:5.1f}%)")

    if future_ended.sum() > 0:
        future_years = df.loc[has_ended & future_ended, 'ended at'].dt.year.value_counts().sort_index()
        print(f"   Future years:")
        for year, count in future_years.items():
            print(f"      {year}: {count:,} records")

# ============================================================================
# STEP 7: MISSING DATA PATTERNS
# ============================================================================
print("\n" + "="*100)
print("STEP 7: MISSING DATA PATTERNS")
print("="*100)

# Check if missing data is systematic
if 'Equipment_Type' in df.columns:
    print(f"\n📊 MISSING TIMESTAMPS BY EQUIPMENT TYPE:")

    for eq_type in df['Equipment_Type'].value_counts().head(5).index:
        eq_mask = df['Equipment_Type'] == eq_type
        eq_count = eq_mask.sum()
        eq_has_started = (eq_mask & has_started).sum()
        eq_has_ended = (eq_mask & has_ended).sum()
        eq_has_both = (eq_mask & has_both).sum()

        print(f"\n   {eq_type}:")
        print(f"      Total:         {eq_count:,}")
        print(f"      Has 'started': {eq_has_started:,} ({eq_has_started/eq_count*100:5.1f}%)")
        print(f"      Has 'ended':   {eq_has_ended:,} ({eq_has_ended/eq_count*100:5.1f}%)")
        print(f"      Has BOTH:      {eq_has_both:,} ({eq_has_both/eq_count*100:5.1f}%)")

# Check by year
if has_started.sum() > 0:
    print(f"\n📊 TIMESTAMP COVERAGE BY YEAR:")
    df['year'] = df['started at'].dt.year

    for year in sorted(df['year'].dropna().unique()):
        year_mask = df['year'] == year
        year_count = year_mask.sum()
        year_has_both = (year_mask & has_both).sum()

        print(f"   {int(year)}: {year_has_both:,}/{year_count:,} ({year_has_both/year_count*100:5.1f}% complete)")

# ============================================================================
# STEP 8: USABILITY ASSESSMENT
# ============================================================================
print("\n" + "="*100)
print("STEP 8: USABILITY ASSESSMENT FOR POF MODELING")
print("="*100)

usable_started = has_started.sum()
usable_ended = has_ended.sum()
usable_both = has_both.sum()

print(f"\n🎯 PRIMARY USE CASE: Temporal PoF Modeling")
print(f"   Requirement: 'started at' for temporal features")
print(f"   ")
print(f"   ✅ Usable records: {usable_started:,} / {total_records:,} ({usable_started/total_records*100:.1f}%)")
print(f"   ❌ Unusable:       {total_records - usable_started:,} ({(total_records - usable_started)/total_records*100:.1f}%)")

if usable_started / total_records >= 0.80:
    print(f"\n   ✅ EXCELLENT: >80% coverage - sufficient for temporal modeling")
elif usable_started / total_records >= 0.60:
    print(f"\n   ✅ GOOD: 60-80% coverage - acceptable for temporal modeling")
elif usable_started / total_records >= 0.40:
    print(f"\n   ⚠️  FAIR: 40-60% coverage - may introduce bias")
else:
    print(f"\n   ❌ POOR: <40% coverage - temporal modeling not recommended")

print(f"\n🎯 SECONDARY USE CASE: Repair Duration Features")
print(f"   Requirement: Both 'started at' AND 'ended at'")
print(f"   ")
print(f"   ✅ Usable records: {usable_both:,} / {total_records:,} ({usable_both/total_records*100:.1f}%)")
print(f"   ❌ Unusable:       {total_records - usable_both:,} ({(total_records - usable_both)/total_records*100:.1f}%)")

if usable_both / total_records >= 0.80:
    print(f"\n   ✅ EXCELLENT: >80% coverage - can use repair duration features")
elif usable_both / total_records >= 0.60:
    print(f"\n   ✅ GOOD: 60-80% coverage - can use repair duration with caution")
else:
    print(f"\n   ❌ LIMITED: <60% coverage - repair duration features may not be reliable")

# ============================================================================
# STEP 9: RECOMMENDATIONS
# ============================================================================
print("\n" + "="*100)
print("RECOMMENDATIONS")
print("="*100)

print(f"\n💡 DATA HANDLING STRATEGY:")

if usable_started / total_records >= 0.70:
    print(f"   1. ✅ USE 'started at' as primary temporal reference")
    print(f"      - {usable_started:,} records available")
    print(f"      - Sufficient for temporal PoF modeling")
else:
    print(f"   1. ⚠️  'started at' has limited coverage ({usable_started/total_records*100:.1f}%)")
    print(f"      - Consider investigating why {total_records - usable_started:,} records are missing timestamps")

if usable_both / total_records >= 0.60:
    print(f"\n   2. ✅ CREATE repair duration features")
    print(f"      - Feature: repair_duration_hours = ended_at - started_at")
    print(f"      - {usable_both:,} records available")
else:
    print(f"\n   2. ⚠️  SKIP repair duration features (only {usable_both/total_records*100:.1f}% coverage)")

print(f"\n   3. 🔧 HANDLE MISSING TIMESTAMPS:")
print(f"      Option A: Exclude faults with missing 'started at' ({total_records - usable_started:,} faults)")
print(f"               → Reduces dataset to {usable_started:,} records")
print(f"      ")
print(f"      Option B: Impute using 'ended at' - N hours (if pattern exists)")
print(f"               → Requires domain knowledge of typical repair duration")
print(f"      ")
print(f"      Option C: Keep all faults but set temporal features to NULL for missing timestamps")
print(f"               → Model must handle missing values")

if future_started.sum() > 0 or future_ended.sum() > 0:
    print(f"\n   4. 🔧 FIX FUTURE DATES:")
    print(f"      → Run: python fix_future_dates.py")
    print(f"      → Strategy: Shift 2025 dates → 2024 dates")

print("\n" + "="*100)
print(f"{'TIMESTAMP QUALITY CHECK COMPLETE':^100}")
print("="*100)
