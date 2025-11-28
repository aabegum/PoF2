"""
Quick check: Does our data have faults beyond the 12M window?
This will explain why 12M and 24M targets are identical.
"""

import pandas as pd
from config import INPUT_FILE, CUTOFF_DATE

print("="*80)
print("DATA AVAILABILITY CHECK FOR 24M HORIZON")
print("="*80)

# Load fault data
print(f"\nLoading: {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE)

# Parse dates
df['started at'] = pd.to_datetime(df['started at'], dayfirst=True, errors='coerce')

# Get date range
print(f"\n📅 Data Date Range:")
print(f"   Earliest fault: {df['started at'].min()}")
print(f"   Latest fault:   {df['started at'].max()}")
print(f"   Cutoff date:    {CUTOFF_DATE}")
print(f"   Total faults:   {len(df):,}")

# Calculate future windows
FUTURE_3M = CUTOFF_DATE + pd.DateOffset(months=3)
FUTURE_6M = CUTOFF_DATE + pd.DateOffset(months=6)
FUTURE_12M = CUTOFF_DATE + pd.DateOffset(months=12)
FUTURE_24M = CUTOFF_DATE + pd.DateOffset(months=24)

print(f"\n🎯 Prediction Windows:")
print(f"   3M window:  {CUTOFF_DATE.date()} → {FUTURE_3M.date()}")
print(f"   6M window:  {CUTOFF_DATE.date()} → {FUTURE_6M.date()}")
print(f"   12M window: {CUTOFF_DATE.date()} → {FUTURE_12M.date()}")
print(f"   24M window: {CUTOFF_DATE.date()} → {FUTURE_24M.date()}")

# Count faults in each window
post_cutoff = df[df['started at'] > CUTOFF_DATE]
window_3M = df[(df['started at'] > CUTOFF_DATE) & (df['started at'] <= FUTURE_3M)]
window_6M = df[(df['started at'] > CUTOFF_DATE) & (df['started at'] <= FUTURE_6M)]
window_12M = df[(df['started at'] > CUTOFF_DATE) & (df['started at'] <= FUTURE_12M)]
window_24M = df[(df['started at'] > CUTOFF_DATE) & (df['started at'] <= FUTURE_24M)]
window_beyond_12M = df[(df['started at'] > FUTURE_12M) & (df['started at'] <= FUTURE_24M)]

print(f"\n📊 Fault Counts by Window:")
print(f"   Total post-cutoff:  {len(post_cutoff):,} faults")
print(f"   Within 3M window:   {len(window_3M):,} faults")
print(f"   Within 6M window:   {len(window_6M):,} faults")
print(f"   Within 12M window:  {len(window_12M):,} faults")
print(f"   Within 24M window:  {len(window_24M):,} faults")
print(f"   Between 12M-24M:    {len(window_beyond_12M):,} faults  ← KEY!")

# Check equipment counts
print(f"\n🔧 Unique Equipment by Window:")
print(f"   3M window:  {window_3M['cbs_id'].nunique():,} equipment")
print(f"   6M window:  {window_6M['cbs_id'].nunique():,} equipment")
print(f"   12M window: {window_12M['cbs_id'].nunique():,} equipment")
print(f"   24M window: {window_24M['cbs_id'].nunique():,} equipment")
print(f"   Between 12M-24M: {window_beyond_12M['cbs_id'].nunique():,} equipment  ← KEY!")

print(f"\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if len(window_beyond_12M) == 0:
    print("❌ NO DATA BEYOND 12M WINDOW!")
    print(f"   Latest fault: {df['started at'].max().date()}")
    print(f"   12M window ends: {FUTURE_12M.date()}")
    print(f"   24M window ends: {FUTURE_24M.date()}")
    print(f"\n   This explains why 12M and 24M targets are identical.")
    print(f"   The dataset does not contain faults for the 24M prediction horizon.")
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Option A: Remove 24M horizon from analysis (no data available)")
    print(f"   Option B: Collect additional data up to {FUTURE_24M.date()}")
    print(f"   Option C: Reduce max horizon to 12M (current data limitation)")
elif window_12M['cbs_id'].nunique() == window_24M['cbs_id'].nunique():
    print("⚠️  DATA EXISTS BEYOND 12M, BUT SAME EQUIPMENT FAILED")
    print(f"   This is rare but possible if:")
    print(f"   - All equipment that will fail in 24M already failed by 12M")
    print(f"   - No 'new' failures between month 12-24")
else:
    print("✓ DATA LOOKS GOOD - 24M has more equipment than 12M")
    print(f"   This is expected and correct!")
