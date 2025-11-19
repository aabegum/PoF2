"""
CONSEQUENCE OF FAILURE (CoF) & RISK SCORING
Turkish EDAŞ PoF Prediction Project

Purpose:
- Calculate Consequence of Failure (CoF) based on customer impact
- Combine with Probability of Failure (PoF) to create Risk scores
- Generate risk-based prioritization for CAPEX/Maintenance decisions

Risk Formula:
    CoF = Outage_Duration × Customer_Count × Critical_Multiplier
    Risk = PoF × CoF
    Risk_Score = normalize(Risk, 0, 100)

Methodology:
- CoF Components:
  • Customer Impact: Total customers affected (urban + suburban + rural, MV + LV)
  • Outage Duration: Average outage minutes per equipment
  • Critical Factor: Multiplier for critical infrastructure (default 1.0)

- Risk Calculation:
  • Raw_Risk = PoF_Probability × CoF
  • Risk_Score = (Raw_Risk - min) / (max - min) × 100
  • Risk_Category: Low (0-40), Medium (40-70), High (70-90), Critical (90-100)

Input:
- predictions/pof_multi_horizon_predictions.csv (PoF from Model 1 - Survival Analysis)
- data/equipment_level_data.csv (customer impact metrics)
- data/combined_data.xlsx (outage durations from fault records)

Output:
- results/risk_assessment_3M.csv (Equipment ID, PoF, CoF, Risk, Priority)
- results/risk_assessment_12M.csv
- results/risk_assessment_24M.csv
- results/capex_priority_list.csv (Top 100 equipment for replacement)
- outputs/risk_analysis/risk_matrix_*.png (PoF vs CoF quadrant charts)
- outputs/risk_analysis/risk_distribution_by_class_*.png

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

# Import centralized configuration
from config import (
    INPUT_FILE,
    EQUIPMENT_LEVEL_FILE,
    PREDICTION_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    CUTOFF_DATE
)
# Fix Unicode encoding for Windows console (Turkish cp1254 issue)
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
warnings.filterwarnings('ignore')

# Configuration (from config.py): RANDOM_STATE, CUTOFF_DATE
HORIZONS = ['3M', '12M', '24M']  # CoF analysis horizons
REFERENCE_DATE = CUTOFF_DATE

# Critical customer multipliers (can be customized)
CRITICAL_MULTIPLIERS = {
    'default': 1.0,        # Standard equipment
    'hospital': 5.0,       # Hospitals (if critical customer flag available)
    'industrial': 2.0,     # Industrial zones
    'residential': 1.0     # Residential areas
}

# Risk category thresholds
RISK_CATEGORIES = {
    'DÜŞÜK': (0, 40),       # Low risk
    'ORTA': (40, 70),       # Medium risk
    'YÜKSEK': (70, 90),     # High risk
    'KRİTİK': (90, 100)     # Critical risk
}

# Output directories
RESULTS_DIR.mkdir(exist_ok=True)
risk_analysis_dir = OUTPUT_DIR / 'risk_analysis'
risk_analysis_dir.mkdir(parents=True, exist_ok=True)

print("="*100)
print("                    CONSEQUENCE OF FAILURE (CoF) & RISK SCORING")
print("                              Risk = PoF × CoF Analysis")
print("="*100)

print("\n📋 Configuration:")
print(f"   Reference Date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
print(f"   Prediction Horizons: {HORIZONS}")
print(f"   Risk Categories: DÜŞÜK/ORTA/YÜKSEK/KRİTİK")

# ============================================================================
# STEP 1: LOAD POF PREDICTIONS (MODEL 1 - SURVIVAL ANALYSIS)
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING POF PREDICTIONS")
print("="*100)

# Try to load survival analysis predictions first (Model 1)
pof_paths = [
    PREDICTION_DIR / 'pof_multi_horizon_predictions.csv',
    PREDICTION_DIR / 'failure_predictions_12m.csv'  # Fallback to Model 2
]

df_pof = None
model_source = None

for pof_path in pof_paths:
    if Path(pof_path).exists():
        print(f"\n✓ Loading PoF predictions from: {pof_path}")
        df_pof = pd.read_csv(pof_path)
        model_source = pof_path
        print(f"✓ Loaded: {len(df_pof):,} equipment predictions")
        break

if df_pof is None:
    print("\n❌ ERROR: PoF predictions not found!")
    print("Please run either:")
    print("  - 09_survival_analysis.py (recommended) OR")
    print("  - 06_model_training.py")
    exit(1)

# Identify PoF columns
pof_cols = [col for col in df_pof.columns if 'PoF_Probability' in col or 'Failure_Probability' in col]
print(f"\n✓ Found PoF columns: {pof_cols}")

# Standardize column names
if 'Ekipman_Kodu' in df_pof.columns:
    df_pof.rename(columns={'Ekipman_Kodu': 'Ekipman_ID'}, inplace=True)
elif 'Equipment_ID' in df_pof.columns:
    df_pof.rename(columns={'Equipment_ID': 'Ekipman_ID'}, inplace=True)

# ============================================================================
# STEP 2: LOAD EQUIPMENT DATA (CUSTOMER IMPACT METRICS)
# ============================================================================
print("\n" + "="*100)
print("STEP 2: LOADING EQUIPMENT DATA FOR CoF CALCULATION")
print("="*100)

equip_path = EQUIPMENT_LEVEL_FILE
if not equip_path.exists():
    print(f"\n❌ ERROR: {equip_path} not found!")
    print("Please run 02_data_transformation.py first!")
    exit(1)

print(f"\n✓ Loading equipment data from: {equip_path}")
df_equip = pd.read_csv(equip_path)
print(f"✓ Loaded: {len(df_equip):,} equipment records")

# Identify customer impact columns
customer_cols = [col for col in df_equip.columns if 'customer' in col.lower() or 'müşteri' in col.lower()]
print(f"\n✓ Found customer impact columns: {len(customer_cols)}")
for col in customer_cols:
    print(f"  • {col}")

# ============================================================================
# STEP 3: CALCULATE OUTAGE DURATION FROM FAULT RECORDS
# ============================================================================
print("\n" + "="*100)
print("STEP 3: CALCULATING AVERAGE OUTAGE DURATION PER EQUIPMENT")
print("="*100)

# Load fault-level data
fault_paths = [INPUT_FILE, 'combined_data.xlsx']  # Try config path first, then fallback
df_faults = None

for fault_path in fault_paths:
    if Path(fault_path).exists():
        print(f"\n✓ Loading fault records from: {fault_path}")
        df_faults = pd.read_excel(fault_path)
        print(f"✓ Loaded: {len(df_faults):,} fault records")
        break

if df_faults is None:
    print("\n⚠️  Warning: Fault data not found - using default outage duration")
    df_equip['Avg_Outage_Minutes'] = 120  # Default 2 hours
else:
    # Find equipment ID column
    equip_id_col = None
    for col in ['cbs_id', 'Ekipman ID', 'Equipment_ID', 'Ekipman_ID']:
        if col in df_faults.columns:
            equip_id_col = col
            break

    # Strategy 1: Try pre-calculated duration columns first (BEST!)
    duration_col = None
    avg_duration = pd.Series(dtype=float)  # Initialize as empty

    if 'duration time' in df_faults.columns:
        duration_col = 'duration time'
        print(f"\n✓ Found pre-calculated duration column: '{duration_col}'")
    elif 'outage duration by hour' in df_faults.columns:
        duration_col = 'outage duration by hour'
        print(f"\n✓ Found pre-calculated duration column: '{duration_col}' (in hours)")

    if duration_col and equip_id_col:
        print(f"→ Strategy 1: Using pre-calculated durations")

        # Use pre-calculated durations
        if 'hour' in duration_col.lower():
            # Convert hours to minutes
            df_faults['Outage_Duration_Minutes'] = pd.to_numeric(df_faults[duration_col], errors='coerce') * 60
        else:
            # Assume it's already in minutes
            df_faults['Outage_Duration_Minutes'] = pd.to_numeric(df_faults[duration_col], errors='coerce')

        # Filter valid durations (positive, < 1 week)
        valid_durations = df_faults[
            (df_faults['Outage_Duration_Minutes'] > 0) &
            (df_faults['Outage_Duration_Minutes'] < 10080)  # 7 days
        ]

        print(f"  Valid pre-calculated durations: {len(valid_durations):,}/{len(df_faults):,}")

        if len(valid_durations) > 0:
            # Calculate average per equipment
            avg_duration = valid_durations.groupby(equip_id_col)['Outage_Duration_Minutes'].mean()

            print(f"\n✓ Calculated average outage duration for {len(avg_duration):,} equipment")
            print(f"  Mean duration: {avg_duration.mean():.1f} minutes")
            print(f"  Median duration: {avg_duration.median():.1f} minutes")
            print(f"  Range: {avg_duration.min():.1f} - {avg_duration.max():.1f} minutes")
        else:
            print(f"\n⚠️  Warning: No valid pre-calculated durations found!")

    # Strategy 2: Calculate from actual timestamps (FALLBACK) - only if Strategy 1 failed
    if len(avg_duration) == 0 and equip_id_col:
        print(f"\n→ Strategy 2: Calculating from timestamps")

        # Find actual timestamp columns (not planned/estimated)
        start_col = None
        end_col = None

        # First try actual times
        if 'started at' in df_faults.columns and 'ended at' in df_faults.columns:
            start_col = 'started at'
            end_col = 'ended at'
            print(f"✓ Found actual timestamp columns: '{start_col}', '{end_col}'")
        else:
            # Fallback to planned/estimated
            for col in df_faults.columns:
                if 'started at' in col.lower() and not start_col:
                    start_col = col
                if 'ended at' in col.lower() and not end_col:
                    end_col = col

            if start_col and end_col:
                print(f"✓ Found timestamp columns: '{start_col}', '{end_col}'")

        if start_col and end_col:
            # Parse timestamps
            df_faults[start_col] = pd.to_datetime(df_faults[start_col], errors='coerce')
            df_faults[end_col] = pd.to_datetime(df_faults[end_col], errors='coerce')

            # Check how many valid timestamps we have
            valid_start = df_faults[start_col].notna().sum()
            valid_end = df_faults[end_col].notna().sum()
            print(f"  Valid timestamps: {valid_start:,}/{len(df_faults):,} start, {valid_end:,}/{len(df_faults):,} end")

            # Calculate duration in minutes
            df_faults['Outage_Duration_Minutes'] = (
                (df_faults[end_col] - df_faults[start_col]).dt.total_seconds() / 60
            )

            # Filter valid durations (positive, < 1 week)
            valid_durations = df_faults[
                (df_faults['Outage_Duration_Minutes'] > 0) &
                (df_faults['Outage_Duration_Minutes'] < 10080)  # 7 days
            ]

            print(f"  Valid durations (0 < duration < 7 days): {len(valid_durations):,}/{len(df_faults):,}")

            if len(valid_durations) > 0:
                # Calculate average per equipment
                avg_duration = valid_durations.groupby(equip_id_col)['Outage_Duration_Minutes'].mean()

                print(f"\n✓ Calculated average outage duration for {len(avg_duration):,} equipment")
                print(f"  Mean duration: {avg_duration.mean():.1f} minutes")
                print(f"  Median duration: {avg_duration.median():.1f} minutes")
                print(f"  Range: {avg_duration.min():.1f} - {avg_duration.max():.1f} minutes")
            else:
                print(f"\n⚠️  Warning: No valid timestamp-based durations found!")
                print(f"  Possible issues: Invalid timestamps, negative durations, or durations > 7 days")
        else:
            print(f"\n⚠️  Warning: Could not find timestamp columns")

    # Merge with equipment data or use default
    if len(avg_duration) > 0:
        df_equip = df_equip.merge(
            avg_duration.rename('Avg_Outage_Minutes'),
            left_on='Ekipman_ID',
            right_index=True,
            how='left'
        )

        # Fill missing with median
        median_duration = avg_duration.median()
        df_equip['Avg_Outage_Minutes'].fillna(median_duration, inplace=True)
        missing_count = df_equip['Avg_Outage_Minutes'].isna().sum()
        if missing_count > 0:
            print(f"  ✓ Filled {missing_count:,} missing values with median ({median_duration:.1f} min)")
    else:
        # No valid durations from either strategy - use default
        print(f"\n⚠️  Using default outage duration for all equipment: 120 minutes")
        df_equip['Avg_Outage_Minutes'] = 120

# ============================================================================
# STEP 4: CALCULATE TOTAL CUSTOMER IMPACT
# ============================================================================
print("\n" + "="*100)
print("STEP 4: CALCULATING TOTAL CUSTOMER IMPACT")
print("="*100)

# Find customer count columns (multiple categories: urban/suburban/rural × MV/LV)
customer_count_cols = []
for col in df_equip.columns:
    # Look for Max columns (worst-case customer impact)
    if 'customer' in col.lower() and ('max' in col.lower() or 'avg' in col.lower()):
        customer_count_cols.append(col)

if customer_count_cols:
    print(f"\n✓ Found {len(customer_count_cols)} customer count columns:")
    for col in customer_count_cols:
        print(f"  • {col}: Mean = {df_equip[col].mean():.1f} customers")

    # Calculate total customer impact (sum across all categories)
    df_equip['Total_Customers_Affected'] = df_equip[customer_count_cols].sum(axis=1)

    print(f"\n✓ Total customer impact calculated:")
    print(f"  Mean customers per equipment: {df_equip['Total_Customers_Affected'].mean():.1f}")
    print(f"  Median customers per equipment: {df_equip['Total_Customers_Affected'].median():.1f}")
    print(f"  Max customers (single equipment): {df_equip['Total_Customers_Affected'].max():.0f}")
else:
    print("\n⚠️  Warning: No customer count columns found - using default value")
    df_equip['Total_Customers_Affected'] = 100  # Default value

# ============================================================================
# STEP 5: CALCULATE CoF (CONSEQUENCE OF FAILURE)
# ============================================================================
print("\n" + "="*100)
print("STEP 5: CALCULATING CoF (CONSEQUENCE OF FAILURE)")
print("="*100)

print("\n--- CoF Formula ---")
print("CoF = Outage_Duration_Minutes × Total_Customers_Affected × Critical_Multiplier")
print("(Critical_Multiplier = 1.0 for all equipment - can be enhanced with critical customer data)")

# Apply critical multiplier (default 1.0 for all - can be enhanced)
df_equip['Critical_Multiplier'] = CRITICAL_MULTIPLIERS['default']

# Calculate raw CoF
df_equip['CoF_Raw'] = (
    df_equip['Avg_Outage_Minutes'] *
    df_equip['Total_Customers_Affected'] *
    df_equip['Critical_Multiplier']
)

print(f"\n✓ Raw CoF calculated:")
print(f"  Mean CoF: {df_equip['CoF_Raw'].mean():.0f}")
print(f"  Median CoF: {df_equip['CoF_Raw'].median():.0f}")
print(f"  Range: {df_equip['CoF_Raw'].min():.0f} - {df_equip['CoF_Raw'].max():.0f}")

# Convert to percentile-based score (0-100) - robust to outliers
df_equip['CoF_Score'] = df_equip['CoF_Raw'].rank(pct=True) * 100

print(f"\n✓ Percentile-based CoF scores (0-100):")
print(f"  Mean CoF Score: {df_equip['CoF_Score'].mean():.1f}")
print(f"  Median CoF Score: {df_equip['CoF_Score'].median():.1f}")

# Calculate percentile thresholds
p75 = df_equip['CoF_Score'].quantile(0.75)
p90 = df_equip['CoF_Score'].quantile(0.90)
p95 = df_equip['CoF_Score'].quantile(0.95)

print(f"  Percentile Thresholds: 75th={p75:.1f}, 90th={p90:.1f}, 95th={p95:.1f}")

# CoF categories based on percentiles (industry standard)
# DÜŞÜK: 0-75th percentile, ORTA: 75-90th, YÜKSEK: 90-95th, KRİTİK: 95-100th
df_equip['CoF_Category'] = pd.cut(
    df_equip['CoF_Score'],
    bins=[0, 75, 90, 95, 100],
    labels=['DÜŞÜK', 'ORTA', 'YÜKSEK', 'KRİTİK'],
    include_lowest=True
)

print(f"\n✓ CoF Distribution:")
cof_dist = df_equip['CoF_Category'].value_counts().sort_index()
for category, count in cof_dist.items():
    pct = count / len(df_equip) * 100
    print(f"  {category:8s}: {count:4d} equipment ({pct:5.1f}%)")

# ============================================================================
# STEP 6: CALCULATE RISK = PoF × CoF FOR EACH HORIZON
# ============================================================================
print("\n" + "="*100)
print("STEP 6: CALCULATING RISK SCORES (Risk = PoF × CoF)")
print("="*100)

# Merge PoF predictions with equipment CoF data
df_risk_base = df_pof.merge(
    df_equip[['Ekipman_ID', 'Ekipman_Sınıfı', 'İlçe', 'CoF_Raw', 'CoF_Score', 'CoF_Category',
              'Avg_Outage_Minutes', 'Total_Customers_Affected']],
    on='Ekipman_ID',
    how='left'
)

print(f"\n✓ Merged PoF and CoF data: {len(df_risk_base):,} equipment")

# Calculate risk for each horizon
risk_results = {}

for horizon in HORIZONS:
    # Find PoF column for this horizon
    pof_col = None
    for col in df_risk_base.columns:
        if f'PoF_Probability_{horizon}' in col:
            pof_col = col
            break

    if pof_col is None:
        print(f"\n⚠️  Skipping {horizon} - PoF column not found")
        continue

    print(f"\n--- {horizon} Risk Calculation ---")

    # Create horizon-specific dataframe
    df_risk = df_risk_base.copy()

    # Calculate raw risk
    df_risk['Risk_Raw'] = df_risk[pof_col] * df_risk['CoF_Raw']

    # Convert to percentile-based score (0-100) - robust to outliers
    df_risk['Risk_Score'] = df_risk['Risk_Raw'].rank(pct=True) * 100

    # Risk categories based on percentiles (industry standard)
    # DÜŞÜK: 0-75th percentile, ORTA: 75-90th, YÜKSEK: 90-95th, KRİTİK: 95-100th
    df_risk['Risk_Category'] = pd.cut(
        df_risk['Risk_Score'],
        bins=[0, 75, 90, 95, 100],
        labels=['DÜŞÜK', 'ORTA', 'YÜKSEK', 'KRİTİK'],
        include_lowest=True
    )

    # Priority ranking (1 = highest risk)
    df_risk['Priority_Rank'] = df_risk['Risk_Score'].rank(ascending=False, method='min').astype(int)

    print(f"✓ Risk scores calculated:")
    print(f"  Mean Risk: {df_risk['Risk_Score'].mean():.1f}")
    print(f"  Median Risk: {df_risk['Risk_Score'].median():.1f}")

    print(f"\n✓ Risk Distribution:")
    risk_dist = df_risk['Risk_Category'].value_counts().sort_index()
    for category, count in risk_dist.items():
        pct = count / len(df_risk) * 100
        print(f"  {category:8s}: {count:4d} equipment ({pct:5.1f}%)")

    # Prepare output columns
    output_cols = [
        'Ekipman_ID', 'Ekipman_Sınıfı', 'İlçe',
        pof_col, 'CoF_Score', 'Risk_Score',
        'Risk_Category', 'Priority_Rank',
        'Avg_Outage_Minutes', 'Total_Customers_Affected', 'CoF_Raw'
    ]

    # Rename PoF column for clarity
    df_risk_output = df_risk[output_cols].copy()
    df_risk_output.rename(columns={pof_col: 'PoF_Probability'}, inplace=True)

    # Sort by risk score
    df_risk_output = df_risk_output.sort_values('Risk_Score', ascending=False)

    # Save
    output_path = RESULTS_DIR / f'risk_assessment_{horizon}.csv'
    df_risk_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: {output_path}")

    # Store for visualization
    risk_results[horizon] = df_risk

# ============================================================================
# STEP 7: GENERATE CAPEX PRIORITY LIST
# ============================================================================
print("\n" + "="*100)
print("STEP 7: GENERATING CAPEX PRIORITY LIST")
print("="*100)

# Use 12M horizon for CAPEX planning (most common planning horizon)
if '12M' in risk_results:
    df_capex = risk_results['12M'].copy()

    # Filter for high-risk equipment (YÜKSEK or KRİTİK)
    df_capex_priority = df_capex[df_capex['Risk_Category'].isin(['YÜKSEK', 'KRİTİK'])].copy()

    print(f"\n✓ High-Risk Equipment (YÜKSEK + KRİTİK): {len(df_capex_priority):,}")

    # Top 100 for CAPEX
    df_capex_top100 = df_capex.head(100).copy()

    # Add actionable columns
    df_capex_top100['Recommended_Action'] = df_capex_top100['Risk_Category'].map({
        'KRİTİK': 'IMMEDIATE REPLACEMENT',
        'YÜKSEK': 'PRIORITY REPLACEMENT',
        'ORTA': 'PREVENTIVE MAINTENANCE',
        'DÜŞÜK': 'ROUTINE MONITORING'
    })

    print(f"\n✓ Top 100 Equipment for CAPEX:")
    action_dist = df_capex_top100['Recommended_Action'].value_counts()
    for action, count in action_dist.items():
        print(f"  {action}: {count} equipment")

    # Save CAPEX priority list
    capex_cols = [
        'Priority_Rank', 'Ekipman_ID', 'Ekipman_Sınıfı', 'İlçe',
        'Risk_Score', 'Risk_Category', 'PoF_Probability_12M', 'CoF_Score',
        'Recommended_Action', 'Total_Customers_Affected', 'Avg_Outage_Minutes'
    ]

    output_path = RESULTS_DIR / 'capex_priority_list.csv'
    df_capex_top100[capex_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: {output_path}")

    # Print top 10
    print(f"\n🔥 TOP 10 HIGHEST RISK EQUIPMENT:")
    print("-" * 100)
    for idx, row in df_capex_top100.head(10).iterrows():
        print(f"  #{row['Priority_Rank']:3.0f} | {str(row['Ekipman_ID']):>10s} | "
              f"{str(row['Ekipman_Sınıfı']):15s} | Risk={row['Risk_Score']:5.1f} | "
              f"PoF={row['PoF_Probability_12M']:.2f} | CoF={row['CoF_Score']:.1f} | "
              f"{row['Risk_Category']}")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 8: GENERATING RISK VISUALIZATIONS")
print("="*100)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

for horizon, df_risk in risk_results.items():
    print(f"\n--- Creating visualizations for {horizon} ---")

    # 1. Risk Matrix (PoF vs CoF)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get PoF column
    pof_col = [col for col in df_risk.columns if f'PoF_Probability_{horizon}' in col][0]

    # Scatter plot
    scatter = ax.scatter(
        df_risk[pof_col] * 100,  # PoF as percentage
        df_risk['CoF_Score'],
        c=df_risk['Risk_Score'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Add quadrant lines
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Quadrant labels
    ax.text(25, 85, 'Low PoF\nHigh CoF', ha='center', va='center',
            fontsize=10, alpha=0.6, style='italic')
    ax.text(75, 85, 'High PoF\nHigh CoF\n(CRITICAL)', ha='center', va='center',
            fontsize=10, alpha=0.6, weight='bold', color='red')
    ax.text(25, 15, 'Low PoF\nLow CoF', ha='center', va='center',
            fontsize=10, alpha=0.6, style='italic')
    ax.text(75, 15, 'High PoF\nLow CoF', ha='center', va='center',
            fontsize=10, alpha=0.6, style='italic')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Risk Score (0-100)', fontsize=12)

    # Labels
    ax.set_xlabel(f'Probability of Failure - {horizon} (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Consequence of Failure Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Matrix - {horizon} Horizon\n(PoF × CoF)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    # Save
    output_path = risk_analysis_dir / f'risk_matrix_{horizon}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved risk matrix: {output_path}")

    # 2. Risk Distribution by Equipment Class
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top equipment classes
    top_classes = df_risk['Ekipman_Sınıfı'].value_counts().head(8).index
    df_plot = df_risk[df_risk['Ekipman_Sınıfı'].isin(top_classes)]

    # Box plot
    sns.boxplot(
        data=df_plot,
        x='Ekipman_Sınıfı',
        y='Risk_Score',
        ax=ax,
        palette='RdYlGn_r'
    )

    # Add mean markers
    means = df_plot.groupby('Ekipman_Sınıfı')['Risk_Score'].mean()
    positions = range(len(top_classes))
    ax.scatter(positions, means, color='red', s=100, zorder=5, marker='D',
               label='Mean Risk', edgecolors='black', linewidth=1)

    # Risk thresholds
    ax.axhline(40, color='yellow', linestyle='--', alpha=0.5, label='Medium Risk Threshold')
    ax.axhline(70, color='orange', linestyle='--', alpha=0.5, label='High Risk Threshold')
    ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='Critical Risk Threshold')

    # Labels
    ax.set_xlabel('Equipment Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Distribution by Equipment Class - {horizon} Horizon',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Save
    output_path = risk_analysis_dir / f'risk_distribution_by_class_{horizon}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved distribution plot: {output_path}")

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*100)
print("RISK ASSESSMENT COMPLETE - SUMMARY")
print("="*100)

print(f"\n📊 OVERALL STATISTICS:")
print(f"   Total Equipment Assessed: {len(df_risk_base):,}")
print(f"   Horizons Analyzed: {', '.join(risk_results.keys())}")

print(f"\n🎯 CoF METRICS:")
print(f"   Mean CoF Score: {df_equip['CoF_Score'].mean():.1f}")
print(f"   High/Critical CoF Equipment: {(df_equip['CoF_Category'].isin(['YÜKSEK', 'KRİTİK'])).sum():,}")

if '12M' in risk_results:
    df_12m = risk_results['12M']
    print(f"\n🎯 RISK METRICS (12M Horizon):")
    print(f"   Mean Risk Score: {df_12m['Risk_Score'].mean():.1f}")
    print(f"   Critical Risk Equipment: {(df_12m['Risk_Category'] == 'KRİTİK').sum():,}")
    print(f"   High Risk Equipment: {(df_12m['Risk_Category'] == 'YÜKSEK').sum():,}")
    print(f"   Equipment Needing Action: {(df_12m['Risk_Category'].isin(['YÜKSEK', 'KRİTİK'])).sum():,}")

print(f"\n📂 OUTPUT FILES GENERATED:")
for horizon in risk_results.keys():
    print(f"   • results/risk_assessment_{horizon}.csv")
print(f"   • results/capex_priority_list.csv (Top 100 equipment)")
for horizon in risk_results.keys():
    print(f"   • outputs/risk_analysis/risk_matrix_{horizon}.png")
    print(f"   • outputs/risk_analysis/risk_distribution_by_class_{horizon}.png")

print(f"\n✅ KEY ACHIEVEMENTS:")
print(f"   ✓ CoF calculated based on customer impact and outage duration")
print(f"   ✓ Risk scores computed (Risk = PoF × CoF)")
print(f"   ✓ Equipment prioritized for CAPEX/Maintenance decisions")
print(f"   ✓ Risk matrices generated for all horizons")
print(f"   ✓ Risk categories assigned (DÜŞÜK/ORTA/YÜKSEK/KRİTİK)")

print("\n" + "="*100)
print("                         RISK ASSESSMENT PIPELINE COMPLETE")
print("="*100)
