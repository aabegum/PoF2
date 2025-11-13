"""
TEMPORAL DATA DIAGNOSTIC
Turkish EDAŞ PoF Prediction Project

Purpose:
- Analyze temporal distribution of fault data
- Recommend optimal prediction cutoff date
- Validate data sufficiency for temporal PoF modeling

This diagnostic helps determine:
1. Date range of fault data
2. Distribution of failures over time
3. Optimal cutoff date for train/test temporal split
4. Expected positive rates for 6M/12M targets

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
print(" "*30 + "TEMPORAL DATA DIAGNOSTIC")
print(" "*25 + "PoF Probability Prediction Analysis")
print("="*100)

# ============================================================================
# STEP 1: LOAD FAULT-LEVEL DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING FAULT-LEVEL DATA")
print("="*100)

# Try to find the fault-level data file
possible_paths = [
    'data/combined_data.xlsx',
    'combined_data.xlsx',
    'data/faults.csv',
    'data/transformed_data.csv'
]

df = None
for path in possible_paths:
    if Path(path).exists():
        print(f"\n✓ Found data at: {path}")
        if path.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        break

if df is None:
    print("\n❌ ERROR: Could not find fault-level data!")
    print("Please ensure one of these files exists:")
    for path in possible_paths:
        print(f"  • {path}")
    exit(1)

print(f"✓ Loaded: {len(df):,} fault records")
print(f"✓ Columns: {len(df.columns)}")

# ============================================================================
# STEP 2: PARSE TEMPORAL COLUMNS
# ============================================================================
print("\n" + "="*100)
print("STEP 2: PARSING TEMPORAL COLUMNS")
print("="*100)

# Identify temporal column
temporal_cols = ['started at', 'Arıza_Tarihi', 'Fault_Date', 'date', 'Date']
fault_date_col = None

for col in temporal_cols:
    if col in df.columns:
        fault_date_col = col
        print(f"\n✓ Found temporal column: {col}")
        break

if fault_date_col is None:
    print("\n❌ ERROR: Could not find fault date column!")
    print("Available columns:", list(df.columns[:20]))
    exit(1)

# Parse dates
df[fault_date_col] = pd.to_datetime(df[fault_date_col], errors='coerce')

# Remove invalid dates
df = df.dropna(subset=[fault_date_col])
print(f"✓ Valid dates: {len(df):,} fault records")

# Identify equipment ID column
equip_id_cols = ['Ekipman_ID', 'Equipment_ID', 'equipment_id', 'ID', 'Asset_ID']
equip_id_col = None

for col in equip_id_cols:
    if col in df.columns:
        equip_id_col = col
        print(f"✓ Found equipment ID column: {col}")
        break

if equip_id_col is None:
    print("\n⚠ WARNING: Could not find equipment ID column!")
    print("Available columns:", list(df.columns[:20]))
    # Try to infer from first few columns
    equip_id_col = df.columns[0]
    print(f"  Using first column as ID: {equip_id_col}")

# ============================================================================
# STEP 3: TEMPORAL DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 3: TEMPORAL DISTRIBUTION ANALYSIS")
print("="*100)

print("\n📊 Date Range:")
earliest_date = df[fault_date_col].min()
latest_date = df[fault_date_col].max()
date_span_days = (latest_date - earliest_date).days

print(f"   Earliest failure: {earliest_date.strftime('%Y-%m-%d')}")
print(f"   Latest failure:   {latest_date.strftime('%Y-%m-%d')}")
print(f"   Total span:       {date_span_days} days ({date_span_days/365:.1f} years)")

# Count unique equipment
unique_equipment = df[equip_id_col].nunique()
print(f"\n📊 Equipment:")
print(f"   Unique equipment: {unique_equipment:,}")
print(f"   Total faults:     {len(df):,}")
print(f"   Avg faults/equip: {len(df)/unique_equipment:.1f}")

# Failures by year
print("\n📊 Failures by Year:")
df['Year'] = df[fault_date_col].dt.year
yearly_counts = df.groupby('Year').size().sort_index()

for year, count in yearly_counts.items():
    pct = count / len(df) * 100
    print(f"   {year}: {count:4,} faults ({pct:5.1f}%)")

# ============================================================================
# STEP 4: RECOMMEND TEMPORAL CUTOFF DATES
# ============================================================================
print("\n" + "="*100)
print("STEP 4: RECOMMENDING TEMPORAL CUTOFF DATES")
print("="*100)

print("\n💡 Temporal Cutoff Strategy:")
print("   To implement true PoF prediction, we need to split data temporally:")
print("   • Historical period: Use for feature calculation")
print("   • Future period: Use for target creation (did equipment fail?)")
print("   • Cutoff date: The 'prediction point' that splits historical vs future")

# Calculate potential cutoff dates
# Strategy: Reserve last 12 months for target window
cutoff_12m = latest_date - timedelta(days=365)
cutoff_6m = latest_date - timedelta(days=180)

print(f"\n📅 Recommended Cutoff Dates:")
print(f"\n   OPTION A: 12-month prediction window")
print(f"   ├─ Cutoff date:     {cutoff_12m.strftime('%Y-%m-%d')}")
print(f"   ├─ Historical:      {earliest_date.strftime('%Y-%m-%d')} to {cutoff_12m.strftime('%Y-%m-%d')}")
print(f"   ├─ Future (target): {cutoff_12m.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
print(f"   └─ Historical span: {(cutoff_12m - earliest_date).days} days ({(cutoff_12m - earliest_date).days/365:.1f} years)")

print(f"\n   OPTION B: 6-month prediction window")
print(f"   ├─ Cutoff date:     {cutoff_6m.strftime('%Y-%m-%d')}")
print(f"   ├─ Historical:      {earliest_date.strftime('%Y-%m-%d')} to {cutoff_6m.strftime('%Y-%m-%d')}")
print(f"   ├─ Future (target): {cutoff_6m.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
print(f"   └─ Historical span: {(cutoff_6m - earliest_date).days} days ({(cutoff_6m - earliest_date).days/365:.1f} years)")

# ============================================================================
# STEP 5: ANALYZE TARGET DISTRIBUTIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 5: ANALYZING FUTURE TARGET DISTRIBUTIONS")
print("="*100)

print("\nFor each cutoff option, calculating how many equipment will fail in future windows...")

for option_name, cutoff_date in [('OPTION A (12M window)', cutoff_12m), ('OPTION B (6M window)', cutoff_6m)]:
    print(f"\n{option_name}:")
    print(f"Cutoff: {cutoff_date.strftime('%Y-%m-%d')}")

    # Split data
    historical = df[df[fault_date_col] < cutoff_date]
    future = df[df[fault_date_col] >= cutoff_date]

    print(f"  Historical faults: {len(historical):,} ({len(historical)/len(df)*100:.1f}%)")
    print(f"  Future faults:     {len(future):,} ({len(future)/len(df)*100:.1f}%)")

    # Get unique equipment in historical data
    historical_equipment = historical[equip_id_col].unique()
    print(f"  Equipment in historical data: {len(historical_equipment):,}")

    # Calculate target distributions
    # 6M Target: Equipment that fails within 180 days after cutoff
    cutoff_6m_end = cutoff_date + timedelta(days=180)
    future_6m = future[future[fault_date_col] <= cutoff_6m_end]
    equipment_fail_6m = future_6m[equip_id_col].unique()
    equipment_no_fail_6m = set(historical_equipment) - set(equipment_fail_6m)

    pos_rate_6m = len(equipment_fail_6m) / len(historical_equipment) * 100 if len(historical_equipment) > 0 else 0

    print(f"\n  6M Target (failures in [{cutoff_date.strftime('%Y-%m-%d')}, {cutoff_6m_end.strftime('%Y-%m-%d')}]):")
    print(f"    Will fail (1):     {len(equipment_fail_6m):,} equipment ({pos_rate_6m:.1f}%)")
    print(f"    Will not fail (0): {len(equipment_no_fail_6m):,} equipment ({100-pos_rate_6m:.1f}%)")

    # 12M Target: Equipment that fails within 365 days after cutoff
    cutoff_12m_end = cutoff_date + timedelta(days=365)
    future_12m = future[future[fault_date_col] <= cutoff_12m_end]
    equipment_fail_12m = future_12m[equip_id_col].unique()
    equipment_no_fail_12m = set(historical_equipment) - set(equipment_fail_12m)

    pos_rate_12m = len(equipment_fail_12m) / len(historical_equipment) * 100 if len(historical_equipment) > 0 else 0

    print(f"\n  12M Target (failures in [{cutoff_date.strftime('%Y-%m-%d')}, {cutoff_12m_end.strftime('%Y-%m-%d')}]):")
    print(f"    Will fail (1):     {len(equipment_fail_12m):,} equipment ({pos_rate_12m:.1f}%)")
    print(f"    Will not fail (0): {len(equipment_no_fail_12m):,} equipment ({100-pos_rate_12m:.1f}%)")

    # Validate class balance
    if pos_rate_6m < 5 or pos_rate_6m > 95:
        print(f"  ⚠️  WARNING: 6M target has severe class imbalance ({pos_rate_6m:.1f}%)")
    if pos_rate_12m < 5 or pos_rate_12m > 95:
        print(f"  ⚠️  WARNING: 12M target has severe class imbalance ({pos_rate_12m:.1f}%)")

# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================
print("\n" + "="*100)
print("STEP 6: CREATING TEMPORAL VISUALIZATIONS")
print("="*100)

output_dir = Path('outputs/temporal_diagnostic')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Failures over time
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Monthly failure counts
df['YearMonth'] = df[fault_date_col].dt.to_period('M')
monthly_counts = df.groupby('YearMonth').size()
monthly_counts.index = monthly_counts.index.to_timestamp()

axes[0].plot(monthly_counts.index, monthly_counts.values, linewidth=2, color='steelblue')
axes[0].axvline(cutoff_12m, color='red', linestyle='--', linewidth=2, label=f'Cutoff 12M ({cutoff_12m.strftime("%Y-%m-%d")})')
axes[0].axvline(cutoff_6m, color='orange', linestyle='--', linewidth=2, label=f'Cutoff 6M ({cutoff_6m.strftime("%Y-%m-%d")})')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Number of Failures', fontsize=12)
axes[0].set_title('Temporal Distribution of Failures (Monthly)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Cumulative failures
df_sorted = df.sort_values(fault_date_col)
df_sorted['Cumulative'] = range(1, len(df_sorted) + 1)

axes[1].plot(df_sorted[fault_date_col], df_sorted['Cumulative'], linewidth=2, color='darkgreen')
axes[1].axvline(cutoff_12m, color='red', linestyle='--', linewidth=2, label=f'Cutoff 12M')
axes[1].axvline(cutoff_6m, color='orange', linestyle='--', linewidth=2, label=f'Cutoff 6M')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Cumulative Failures', fontsize=12)
axes[1].set_title('Cumulative Failure Count Over Time', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'temporal_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Temporal distribution plot saved: {output_dir / 'temporal_distribution.png'}")

# ============================================================================
# STEP 7: RECOMMENDATIONS
# ============================================================================
print("\n" + "="*100)
print("RECOMMENDATIONS FOR POF IMPLEMENTATION")
print("="*100)

print("\n🎯 RECOMMENDED APPROACH:")
print("\n1. Choose OPTION A (12M window cutoff) if:")
print("   • You want both 6M and 12M predictions")
print("   • Historical span is sufficient (>= 2 years)")
print("   • Both 6M and 12M targets have reasonable positive rates (10-40%)")

print("\n2. Choose OPTION B (6M window cutoff) if:")
print("   • You only need 6M predictions")
print("   • You want maximum historical data for training")
print("   • 6M target has better class balance")

print("\n3. Implementation Steps:")
print("   STEP 1: Run 02b_temporal_transformation.py")
print("           → Calculates features using ONLY historical faults (before cutoff)")
print("   ")
print("   STEP 2: Run 06d_temporal_model_training.py")
print("           → Creates temporal targets (failures in future windows)")
print("           → Trains PoF prediction models")
print("           → Outputs: P(failure in next 6M), P(failure in next 12M)")

print("\n4. Key Differences from Current Approach:")
print("   ❌ OLD: Target = Total lifetime failures >= 2 (static classification)")
print("   ✅ NEW: Target = Did equipment fail in [cutoff, cutoff+N months]? (temporal PoF)")
print("   ")
print("   ❌ OLD: Same target for 6M and 12M (identical predictions)")
print("   ✅ NEW: Different targets for 6M and 12M (true probability forecasting)")

print("\n💡 NEXT STEPS:")
print("   1. Review the diagnostic plots in: outputs/temporal_diagnostic/")
print("   2. Decide on cutoff date (Option A or B)")
print("   3. Run 02b_temporal_transformation.py with chosen cutoff")
print("   4. Run 06d_temporal_model_training.py to train PoF models")

print("\n" + "="*100)
print(f"{'TEMPORAL DIAGNOSTIC COMPLETE':^100}")
print("="*100)
