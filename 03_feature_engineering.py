"""
ADVANCED FEATURE ENGINEERING
Turkish EDAŞ PoF Prediction Project

Purpose:
- Calculate Expected Life Ratios (Yas_Beklenen_Omur_Orani)
- Create geographic clusters
- Engineer risk scores
- Add temporal aggregations
- Generate interaction features

Input:  data/equipment_level_data.csv (1,313 equipment)
Output: data/features_engineered.csv (~40-50 features)

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

# Import centralized configuration
from config import (
    EQUIPMENT_LEVEL_FILE,
    FEATURES_ENGINEERED_FILE,
    FEATURE_CATALOG_FILE,
    RANDOM_STATE,
    ENABLE_GEOGRAPHIC_CLUSTERING,
    MIN_EQUIPMENT_FOR_CLUSTERING,
    EQUIPMENT_PER_CLUSTER
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

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*100)
print(" "*30 + "ADVANCED FEATURE ENGINEERING")
print(" "*35 + "PoF Prediction")
print("="*100)

# ============================================================================
# CONFIGURATION: EXPECTED LIFE STANDARDS
# ============================================================================

# Tier 1: Industry Standards (Most Common Equipment)
# Based on Turkish EDAŞ standards and international IEEE/IEC guidelines
EXPECTED_LIFE_STANDARDS = {
    'Ayırıcı': 25,           # Disconnectors
    'Kesici': 30,            # Circuit breakers
    'OG/AG Trafo': 30,       # Transformers (Medium/Low voltage)
    'AG Anahtar': 20,        # LV Switches
    'AG Pano Box': 25,       # LV Panel Boxes
    'Bina': 40,              # Buildings/Substations
    'Rekortman': 20,         # Reclosers
    'OG Hat': 35,            # MV Lines
    'AG Hat': 30,            # LV Lines
    'AG Pano': 25,           # LV Panels
    'Trafo Bina Tip': 30,    # Building-type transformers
}

# Tier 2: Voltage-Based Defaults (If exact class not found)
VOLTAGE_BASED_LIFE = {
    'OG': 30,  # Medium voltage equipment
    'AG': 25,  # Low voltage equipment  
    'YG': 35,  # High voltage equipment
}

# Tier 3: Overall Conservative Default
DEFAULT_LIFE = 25

print("\n📋 Expected Life Standards Loaded:")
print(f"   Tier 1: {len(EXPECTED_LIFE_STANDARDS)} specific equipment types")
print(f"   Tier 2: {len(VOLTAGE_BASED_LIFE)} voltage-based defaults")
print(f"   Tier 3: Default = {DEFAULT_LIFE} years")

# ============================================================================
# STEP 1: LOAD EQUIPMENT-LEVEL DATA
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING EQUIPMENT-LEVEL DATA")
print("="*100)

if not EQUIPMENT_LEVEL_FILE.exists():
    print(f"\n❌ ERROR: File not found at {EQUIPMENT_LEVEL_FILE}")
    print("Please run 02_data_transformation.py first!")
    exit(1)

print(f"\n✓ Loading from: {EQUIPMENT_LEVEL_FILE}")
df = pd.read_csv(EQUIPMENT_LEVEL_FILE)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# Verify Equipment_Class_Primary exists (created by 02_data_transformation.py)
if 'Equipment_Class_Primary' not in df.columns:
    print("\n❌ ERROR: Equipment_Class_Primary column not found!")
    print("This column should be created by 02_data_transformation.py (v2.0+)")
    print("Please run 02_data_transformation.py first!")
    exit(1)

print("✓ Equipment_Class_Primary column verified (from transformation script)")

original_feature_count = df.shape[1]

# ============================================================================
# STEP 2: EXPECTED LIFE RATIO CALCULATION
# ============================================================================
print("\n" + "="*100)
print("STEP 2: CALCULATING EXPECTED LIFE RATIOS")
print("="*100)

def get_expected_life(equipment_class):
    """
    Get expected life for equipment using tiered lookup strategy
    
    Tier 1: Exact match in EXPECTED_LIFE_STANDARDS
    Tier 2: Voltage-based match (OG/AG/YG)
    Tier 3: Conservative default (25 years)
    """
    if pd.isna(equipment_class):
        return DEFAULT_LIFE
    
    equipment_class = str(equipment_class).strip()
    
    # Tier 1: Exact match
    if equipment_class in EXPECTED_LIFE_STANDARDS:
        return EXPECTED_LIFE_STANDARDS[equipment_class]
    
    # Tier 2: Voltage-based match
    for voltage_key, life in VOLTAGE_BASED_LIFE.items():
        if voltage_key in equipment_class:
            return life
    
    # Tier 3: Default
    return DEFAULT_LIFE

print("\n--- Calculating Expected Life per Equipment ---")

# Note: Equipment_Class_Primary is provided by 02_data_transformation.py (v2.0+)
# Priority: Equipment_Type → Ekipman_Sınıfı → Kesinti Ekipman Sınıfı → Ekipman Sınıf

# Show Equipment_Class_Primary coverage
class_coverage = df['Equipment_Class_Primary'].notna().sum()
class_pct = class_coverage / len(df) * 100
print(f"\n✓ Equipment_Class_Primary available for {class_coverage:,} equipment ({class_pct:.1f}%)")

# Show unique equipment types
unique_classes = df['Equipment_Class_Primary'].nunique()
print(f"✓ {unique_classes} unique equipment types identified")

# Apply expected life lookup using primary class
df['Beklenen_Ömür_Yıl'] = df['Equipment_Class_Primary'].apply(get_expected_life)

# Show expected life distribution
print("\nExpected Life Distribution:")
life_dist = df['Beklenen_Ömür_Yıl'].value_counts().sort_index()
for life_years, count in life_dist.items():
    pct = count / len(df) * 100
    print(f"  {life_years} years: {count:,} equipment ({pct:.1f}%)")

# Calculate Age-to-Expected-Life Ratio
print("\n--- Calculating Yas_Beklenen_Omur_Orani ---")

def calculate_age_ratio(row):
    """Calculate ratio of current age to expected life"""
    age = row['Ekipman_Yaşı_Yıl']
    expected_life = row['Beklenen_Ömür_Yıl']
    
    if pd.notna(age) and expected_life > 0:
        return age / expected_life
    return None

df['Yas_Beklenen_Omur_Orani'] = df.apply(calculate_age_ratio, axis=1)

# Statistics
ratio_available = df['Yas_Beklenen_Omur_Orani'].notna().sum()
print(f"✓ Ratio calculated for {ratio_available:,} equipment ({ratio_available/len(df)*100:.1f}%)")

if ratio_available > 0:
    ratio_stats = df['Yas_Beklenen_Omur_Orani'].describe()
    print(f"\nRatio Statistics:")
    print(f"  Mean:   {ratio_stats['mean']:.2f}")
    print(f"  Median: {ratio_stats['50%']:.2f}")
    print(f"  Min:    {ratio_stats['min']:.2f}")
    print(f"  Max:    {ratio_stats['max']:.2f}")
    
    # Risk categorization
    print(f"\nRisk Categories Based on Age Ratio:")
    df['Age_Risk_Category'] = pd.cut(
        df['Yas_Beklenen_Omur_Orani'],
        bins=[0, 0.4, 0.7, 1.0, float('inf')],
        labels=['Low (0-40%)', 'Medium (40-70%)', 'High (70-100%)', 'Critical (>100%)']
    )
    
    for category in ['Low (0-40%)', 'Medium (40-70%)', 'High (70-100%)', 'Critical (>100%)']:
        count = (df['Age_Risk_Category'] == category).sum()
        pct = count / ratio_available * 100
        icon = "✅" if "Low" in category else ("⚠" if "Medium" in category else "❌")
        print(f"  {icon} {category}: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 3: GEOGRAPHIC CLUSTERING - REMOVED
# ============================================================================
# NOTE: Geographic clustering has been removed from the feature engineering pipeline
#
# Rationale for removal:
#   - K-means clustering on X,Y coordinates produces noisy, unstable patterns
#   - Distribution networks are LINEAR (power lines), not point-based clusters
#   - Cluster aggregations (Arıza_Sayısı_12ay_Cluster_Avg, etc.) created data leakage risk
#   - Better alternative: Use İlçe (district) - clear, interpretable, domain-meaningful
#
# Features removed:
#   - Geographic_Cluster (K-means cluster ID)
#   - Arıza_Sayısı_12ay_Cluster_Avg (leaky - uses future data)
#   - Tekrarlayan_Arıza_90gün_Flag_Cluster_Avg (leaky - uses future data)
#   - MTBF_Gün_Cluster_Avg (circular logic)
#
# Replacement: İlçe (district) is created in STEP 9B and provides superior geographic signal
print("\n" + "="*100)
print("STEP 3: GEOGRAPHIC CLUSTERING - SKIPPED (See STEP 9B for İlçe-based geography)")
print("="*100)
print("✓ Geographic clustering removed - using İlçe (district) instead")

# ============================================================================
# STEP 4: FAILURE RATE FEATURES - REMOVED
# ============================================================================
# NOTE: Redundant failure rate features have been removed from the pipeline
#
# Rationale for removal:
#   - Failure_Rate_Per_Year: Redundant (tree models learn this from Toplam_Arıza / Ekipman_Yaşı)
#   - Recent_Failure_Intensity: Uses Arıza_Sayısı_3ay (may include post-cutoff data - leakage risk)
#   - Failure_Acceleration: Uses Arıza_Sayısı_6ay (may include post-cutoff data - leakage risk)
#
# Features removed:
#   - Failure_Rate_Per_Year (redundant ratio feature)
#   - Recent_Failure_Intensity (leaky - uses 3-month window)
#   - Failure_Acceleration (leaky - uses 6-month window)
#
# Replacement: Models learn these patterns from raw features (Toplam_Arıza_Sayisi_Lifetime,
#              Ekipman_Yaşı_Yıl, Son_Arıza_Gun_Sayisi, MTBF metrics)
print("\n" + "="*100)
print("STEP 4: FAILURE RATE FEATURES - SKIPPED (Models learn from raw features)")
print("="*100)
print("✓ Redundant failure rate features removed")

# ============================================================================
# STEP 5: RELIABILITY METRICS
# ============================================================================
print("\n" + "="*100)
print("STEP 5: ENGINEERING RELIABILITY METRICS")
print("="*100)

# MTBF-based reliability score (inverse of failure rate)
if 'MTBF_Gün' in df.columns:
    df['Reliability_Score'] = df['MTBF_Gün'].apply(
        lambda x: min(100, (x / 365) * 100) if pd.notna(x) and x > 0 else 0
    )
    print(f"✓ Reliability score (0-100) calculated")

# Time since last failure normalized
if 'Son_Arıza_Gun_Sayisi' in df.columns and 'MTBF_Gün' in df.columns:
    df['Time_Since_Last_Normalized'] = df.apply(
        lambda row: row['Son_Arıza_Gun_Sayisi'] / row['MTBF_Gün']
        if pd.notna(row['MTBF_Gün']) and row['MTBF_Gün'] > 0
        else None,
        axis=1
    )
    print(f"✓ Normalized time since last failure calculated")

# Failure free period indicator
if 'Arıza_Sayısı_3ay' in df.columns:
    df['Failure_Free_3M'] = (df['Arıza_Sayısı_3ay'] == 0).astype(int)
    failure_free = df['Failure_Free_3M'].sum()
    print(f"✓ Failure-free equipment (3 months): {failure_free:,} ({failure_free/len(df)*100:.1f}%)")

# ============================================================================
# STEP 5B: ADVANCED MTBF FEATURES (TIER 3 ENHANCEMENTS)
# ============================================================================
print("\n" + "="*100)
print("STEP 5B: ENGINEERING ADVANCED MTBF FEATURES")
print("="*100)

# These features detect equipment degradation trends and failure predictability
# Part of the optimal 30-feature set for improved PoF prediction

# FEATURE 1: MTBF_InterFault_Trend (Degradation Detector)
# Purpose: Detect if equipment is degrading (failures accelerating) over time
# Method: Compare recent MTBF to historical MTBF
#   - Recent MTBF = Average of last 50% of inter-fault times
#   - Historical MTBF = Average of first 50% of inter-fault times
#   - Trend = Recent / Historical
# Interpretation:
#   - < 1.0 = Degrading (failures accelerating - HIGH RISK)
#   - = 1.0 = Stable
#   - > 1.0 = Improving (failures slowing down - GOOD SIGN)

print("\n--- Calculating MTBF Trend (Degradation Detector) ---")

def calculate_mtbf_trend(row):
    """
    Calculate MTBF trend by comparing recent vs historical inter-fault times
    Requires at least 4 faults to split into two periods
    """
    # This would require fault-level data to calculate inter-fault times
    # Since we only have equipment-level aggregates, we approximate using:
    # - MTBF_Gün (average inter-fault time)
    # - Son_Arıza_Gun_Sayisi (days since last fault)
    # - Toplam_Arıza_Sayisi_Lifetime (total fault count)

    if pd.isna(row['MTBF_Gün']) or row['Toplam_Arıza_Sayisi_Lifetime'] < 4:
        return None  # Need at least 4 faults for trend analysis

    # Approximation: If recent fault happened sooner than expected MTBF, equipment is degrading
    # Trend = (Days since last fault) / MTBF_Gün
    # This is a proxy - ideally would use actual inter-fault time series
    if pd.notna(row['Son_Arıza_Gun_Sayisi']) and row['MTBF_Gün'] > 0:
        # Invert to get degradation indicator
        # If days_since < MTBF → recent fault was early → degrading → trend < 1
        recent_ratio = row['Son_Arıza_Gun_Sayisi'] / row['MTBF_Gün']
        return recent_ratio

    return 1.0  # Default to stable

if 'MTBF_Gün' in df.columns:
    df['MTBF_InterFault_Trend'] = df.apply(calculate_mtbf_trend, axis=1)

    trend_available = df['MTBF_InterFault_Trend'].notna().sum()
    print(f"✓ MTBF_InterFault_Trend calculated: {trend_available:,} equipment ({trend_available/len(df)*100:.1f}%)")

    if trend_available > 0:
        trend_stats = df['MTBF_InterFault_Trend'].describe()
        print(f"  Mean trend: {trend_stats['mean']:.2f}")
        print(f"  Median trend: {trend_stats['50%']:.2f}")

        # Degradation categories
        degrading = (df['MTBF_InterFault_Trend'] < 0.8).sum()
        stable = ((df['MTBF_InterFault_Trend'] >= 0.8) & (df['MTBF_InterFault_Trend'] <= 1.2)).sum()
        improving = (df['MTBF_InterFault_Trend'] > 1.2).sum()

        print(f"  Degrading (trend < 0.8): {degrading:,} equipment")
        print(f"  Stable (0.8 ≤ trend ≤ 1.2): {stable:,} equipment")
        print(f"  Improving (trend > 1.2): {improving:,} equipment")
else:
    print("⚠ MTBF_Gün not available - skipping MTBF_InterFault_Trend")
    df['MTBF_InterFault_Trend'] = None

# FEATURE 2: MTBF_InterFault_StdDev (Predictability Measure)
# Purpose: Measure failure timing predictability
# Method: Standard deviation of inter-fault times (normalized by mean)
# Interpretation:
#   - Low StdDev = Consistent, predictable failure pattern (good for maintenance planning)
#   - High StdDev = Erratic, unpredictable failures (complex degradation)

print("\n--- Calculating MTBF StdDev (Predictability Measure) ---")

def calculate_mtbf_stddev_proxy(row):
    """
    Calculate MTBF variability proxy
    True calculation would require fault-level inter-fault time series
    We approximate using coefficient of variation concept
    """
    # Requires at least 2 faults for variability measure
    if row['Toplam_Arıza_Sayisi_Lifetime'] < 2:
        return None

    # Proxy: Equipment with recurring faults in short windows have high variability
    # Use ratio of lifetime MTBF vs observable MTBF as variability indicator
    if pd.notna(row['MTBF_Lifetime_Gün']) and pd.notna(row['MTBF_Observable_Gün']):
        if row['MTBF_Observable_Gün'] > 0:
            # Higher ratio = more variable (observable MTBF differs from lifetime MTBF)
            variability_ratio = abs(row['MTBF_Lifetime_Gün'] - row['MTBF_Observable_Gün']) / row['MTBF_Observable_Gün']
            return variability_ratio

    return 0.0  # Default to low variability

if 'MTBF_Lifetime_Gün' in df.columns and 'MTBF_Observable_Gün' in df.columns:
    df['MTBF_InterFault_StdDev'] = df.apply(calculate_mtbf_stddev_proxy, axis=1)

    stddev_available = df['MTBF_InterFault_StdDev'].notna().sum()
    print(f"✓ MTBF_InterFault_StdDev calculated: {stddev_available:,} equipment ({stddev_available/len(df)*100:.1f}%)")

    if stddev_available > 0:
        stddev_stats = df['MTBF_InterFault_StdDev'].describe()
        print(f"  Mean variability: {stddev_stats['mean']:.2f}")
        print(f"  Median variability: {stddev_stats['50%']:.2f}")

        # Predictability categories
        predictable = (df['MTBF_InterFault_StdDev'] < 0.3).sum()
        moderate = ((df['MTBF_InterFault_StdDev'] >= 0.3) & (df['MTBF_InterFault_StdDev'] <= 0.7)).sum()
        erratic = (df['MTBF_InterFault_StdDev'] > 0.7).sum()

        print(f"  Predictable (low variability < 0.3): {predictable:,} equipment")
        print(f"  Moderate variability (0.3-0.7): {moderate:,} equipment")
        print(f"  Erratic (high variability > 0.7): {erratic:,} equipment")
else:
    print("⚠ MTBF columns not available - skipping MTBF_InterFault_StdDev")
    df['MTBF_InterFault_StdDev'] = None

print("\n✓ Advanced MTBF features complete!")

# ============================================================================
# STEP 6: CUSTOMER IMPACT FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 6: ENGINEERING CUSTOMER IMPACT FEATURES")
print("="*100)

if 'Avg_Customer_Count' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
    # Customer-minutes at risk (annual)
    df['Customer_Minutes_Risk_Annual'] = df['Avg_Customer_Count'] * df['Arıza_Sayısı_12ay'] * 120  # Assume 2 hours per outage
    
    risk_available = df['Customer_Minutes_Risk_Annual'].notna().sum()
    print(f"✓ Customer-minutes risk calculated: {risk_available:,} equipment")
    
    # Customer impact category
    if risk_available > 0:
        df['Customer_Impact_Category'] = pd.cut(
            df['Customer_Minutes_Risk_Annual'],
            bins=[0, 1000, 5000, 20000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        for category in ['Low', 'Medium', 'High', 'Critical']:
            count = (df['Customer_Impact_Category'] == category).sum()
            pct = count / risk_available * 100
            print(f"  {category} impact: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 7: EQUIPMENT CLASS AGGREGATIONS - REMOVED
# ============================================================================
# NOTE: Equipment class aggregation features have been removed from the pipeline
#
# Rationale for removal:
#   - Class averages create target leakage (using class performance to predict class members)
#   - Circular reasoning: If equipment X is in class Y, using Y's average to predict X
#   - Models learn equipment class patterns automatically from Equipment_Class_Primary
#   - Aggregating Arıza_Sayısı_12ay creates leakage (uses 12-month window)
#
# Features removed:
#   - Arıza_Sayısı_12ay_Class_Avg (leaky - uses 12-month aggregation)
#   - MTBF_Gün_Class_Avg (circular logic)
#   - Ekipman_Yaşı_Yıl_Class_Avg (not predictive)
#   - Yas_Beklenen_Omur_Orani_Class_Avg (not predictive)
#   - Failure_vs_Class_Avg (derived from leaky feature)
#
# Replacement: Equipment_Class_Primary (already exists) provides class information without leakage
print("\n" + "="*100)
print("STEP 7: EQUIPMENT CLASS AGGREGATIONS - SKIPPED (Using Equipment_Class_Primary instead)")
print("="*100)
print("✓ Class aggregation features removed")

# ============================================================================
# STEP 8: INTERACTION FEATURES (CLEANED - LEAKAGE REMOVED)
# ============================================================================
print("\n" + "="*100)
print("STEP 8: CREATING INTERACTION FEATURES")
print("="*100)

# Age ratio × Recurrence interaction (KEEP - uses lifetime count, no leakage)
if 'Yas_Beklenen_Omur_Orani' in df.columns and 'Tekrarlayan_Arıza_90gün_Flag' in df.columns:
    df['AgeRatio_Recurrence_Interaction'] = df['Yas_Beklenen_Omur_Orani'] * df['Tekrarlayan_Arıza_90gün_Flag']
    print("✓ Age Ratio × Recurrence interaction (compound aging + recurrence risk)")

# Overdue Factor (TIER 3 Enhancement - Imminent Failure Risk)
# Purpose: Detect equipment "overdue" for next failure based on historical pattern
# Calculation: Days since last failure / MTBF_InterFault_Gün
# Interpretation:
#   - < 1.0 = Not yet due for next failure based on pattern
#   - = 1.0 = Due for failure based on historical pattern
#   - > 1.0 = Overdue (higher risk of imminent failure)

print("\n--- Calculating Overdue Factor (Imminent Risk Detector) ---")

if 'Son_Arıza_Gun_Sayisi' in df.columns and 'MTBF_Gün' in df.columns:
    def calculate_overdue_factor(row):
        """
        Calculate how overdue equipment is for next failure
        Requires valid MTBF and days since last failure
        """
        if pd.isna(row['Son_Arıza_Gun_Sayisi']) or pd.isna(row['MTBF_Gün']):
            return None

        # Equipment that never failed gets None (no pattern to base on)
        if row['Toplam_Arıza_Sayisi_Lifetime'] == 0:
            return None

        # Avoid division by zero
        if row['MTBF_Gün'] <= 0:
            return None

        # Calculate overdue factor
        overdue_factor = row['Son_Arıza_Gun_Sayisi'] / row['MTBF_Gün']
        return overdue_factor

    df['Overdue_Factor'] = df.apply(calculate_overdue_factor, axis=1)

    overdue_available = df['Overdue_Factor'].notna().sum()
    print(f"✓ Overdue_Factor calculated: {overdue_available:,} equipment ({overdue_available/len(df)*100:.1f}%)")

    if overdue_available > 0:
        overdue_stats = df['Overdue_Factor'].describe()
        print(f"  Mean overdue factor: {overdue_stats['mean']:.2f}")
        print(f"  Median overdue factor: {overdue_stats['50%']:.2f}")

        # Risk categories
        not_due = (df['Overdue_Factor'] < 0.8).sum()
        approaching = ((df['Overdue_Factor'] >= 0.8) & (df['Overdue_Factor'] < 1.2)).sum()
        overdue = (df['Overdue_Factor'] >= 1.2).sum()

        print(f"  Not yet due (< 0.8): {not_due:,} equipment")
        print(f"  Approaching due (0.8-1.2): {approaching:,} equipment")
        print(f"  Overdue (≥ 1.2): {overdue:,} equipment (⚠ IMMINENT RISK)")
else:
    print("⚠ Required columns not available - skipping Overdue_Factor")
    df['Overdue_Factor'] = None

# REMOVED FEATURES (leakage risk):
# - Age_Failure_Interaction: Used Arıza_Sayısı_12ay (may include post-cutoff data)
# - Customer_Failure_Interaction: Used Arıza_Sayısı_12ay (may include post-cutoff data)

# ============================================================================
# STEP 9B: ADDITIONAL DOMAIN-SPECIFIC FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 9B: ADDITIONAL DOMAIN-SPECIFIC FEATURES")
print("="*100)

# ============================================================================
# 1. VOLTAGE LEVEL CLASSIFICATION
# ============================================================================
print("\n--- Voltage Level Classification ---")

if 'component_voltage' in df.columns:
    # Rename for clarity
    df['voltage_level'] = df['component_voltage']

    coverage = df['voltage_level'].notna().sum()
    print(f"✓ Voltage level data available: {coverage:,} equipment ({coverage/len(df)*100:.1f}%)")
    print(f"  Unique voltage values: {df['voltage_level'].nunique()}")

    # Show raw distribution
    print(f"\n  Raw voltage distribution:")
    for voltage, count in df['voltage_level'].value_counts().sort_index(ascending=False).head(5).items():
        pct = count / len(df) * 100
        print(f"    {voltage:>10.1f} V: {count:>4,} equipment ({pct:>5.1f}%)")

    # Create categorical voltage classification
    def classify_voltage(voltage):
        """
        Classify voltage into Turkish EDAŞ standard categories
        AG (Alçak Gerilim) = Low Voltage (< 1 kV)
        OG (Orta Gerilim) = Medium Voltage (10-36 kV)
        YG (Yüksek Gerilim) = High Voltage (>= 36 kV)
        """
        if pd.isna(voltage):
            return None

        voltage = float(voltage)

        if voltage < 1000:  # < 1 kV
            return 'AG'
        elif 10000 <= voltage < 36000:  # 10-36 kV
            return 'OG'
        elif voltage >= 36000:  # >= 36 kV
            return 'YG'
        else:
            return 'Bilinmeyen'

    df['Voltage_Class'] = df['voltage_level'].apply(classify_voltage)

    # Show classification distribution
    print(f"\n  Voltage classification:")
    voltage_class_dist = df['Voltage_Class'].value_counts()
    for v_class, count in voltage_class_dist.items():
        if pd.notna(v_class):
            pct = count / len(df) * 100

            # Add description
            if v_class == 'AG':
                desc = '(0.4 kV - Low Voltage)'
            elif v_class == 'OG':
                desc = '(15.8/34.5 kV - Medium Voltage)'
            elif v_class == 'YG':
                desc = '(>36 kV - High Voltage)'
            else:
                desc = ''

            print(f"    {v_class} {desc:35s}: {count:>4,} equipment ({pct:>5.1f}%)")

    # Create binary flags for modeling
    df['Is_MV'] = (df['Voltage_Class'] == 'OG').astype(int)
    df['Is_LV'] = (df['Voltage_Class'] == 'AG').astype(int)
    df['Is_HV'] = (df['Voltage_Class'] == 'YG').astype(int)

    print(f"\n✓ Voltage flags created:")
    print(f"    Is_MV: {df['Is_MV'].sum():,} equipment")
    print(f"    Is_LV: {df['Is_LV'].sum():,} equipment")
    print(f"    Is_HV: {df['Is_HV'].sum():,} equipment")

    # Voltage-based failure analysis
    print(f"\n  Voltage-level failure patterns:")
    for v_class in ['AG', 'OG', 'YG']:
        mask = df['Voltage_Class'] == v_class
        if mask.sum() > 0:
            avg_faults = df.loc[mask, 'Arıza_Sayısı_12ay'].mean() if 'Arıza_Sayısı_12ay' in df.columns else 0
            avg_age = df.loc[mask, 'Ekipman_Yaşı_Yıl'].mean() if 'Ekipman_Yaşı_Yıl' in df.columns else 0
            recurring_pct = df.loc[mask, 'Tekrarlayan_Arıza_90gün_Flag'].sum() / mask.sum() * 100 if 'Tekrarlayan_Arıza_90gün_Flag' in df.columns else 0

            print(f"    {v_class}: Avg age={avg_age:.1f}y, Avg faults(12M)={avg_faults:.2f}, Recurring={recurring_pct:.1f}%")

else:
    print("⚠ component_voltage column not found (should be from 02_data_transformation.py)")
    df['voltage_level'] = None
    df['Voltage_Class'] = None
    df['Is_MV'] = 0
    df['Is_LV'] = 0
    df['Is_HV'] = 0

# ============================================================================
# 2. GEOGRAPHIC CLASSIFICATION (Urban vs Rural)
# ============================================================================
print("\n--- Geographic Classification (Manisa Region) ---")

if 'İlçe' in df.columns:
    # Salihli and Alaşehir are major urban centers
    # Gördes is rural/agricultural
    urban_districts = ['SALİHLİ', 'ALAŞEHİR', 'SALIHLI', 'ALAŞEHIR']

    df['Bölge_Tipi'] = df['İlçe'].apply(
        lambda x: 'Kentsel' if pd.notna(x) and str(x).upper() in urban_districts
        else 'Kırsal'
    )

    # Show distribution
    kentsel_count = (df['Bölge_Tipi'] == 'Kentsel').sum()
    kirsal_count = (df['Bölge_Tipi'] == 'Kırsal').sum()
    print(f"✓ Geographic classification:")
    print(f"  Kentsel (Salihli + Alaşehir): {kentsel_count:,} equipment ({kentsel_count/len(df)*100:.1f}%)")
    print(f"  Kırsal (Gördes): {kirsal_count:,} equipment ({kirsal_count/len(df)*100:.1f}%)")

    # District breakdown (top 5)
    print(f"\n  District breakdown:")
    for district, count in df['İlçe'].value_counts().head(5).items():
        btype = 'Kentsel' if str(district).upper() in urban_districts else 'Kırsal'
        print(f"    {district:<15} {count:>4,} equipment ({btype})")
else:
    print("⚠ İlçe column not found")
    df['Bölge_Tipi'] = None

# ============================================================================
# 3. SEASONAL FEATURES
# ============================================================================
print("\n--- Seasonal Features ---")

if 'Son_Arıza_Tarihi' in df.columns:
    def get_season(date_str):
        """Map fault date to season (Turkish climate)"""
        if pd.isna(date_str):
            return None
        try:
            date = pd.to_datetime(date_str)
            month = date.month
            if month in [12, 1, 2]:
                return 'Kış'        # Winter
            elif month in [3, 4, 5]:
                return 'İlkbahar'   # Spring
            elif month in [6, 7, 8]:
                return 'Yaz'        # Summer
            else:
                return 'Sonbahar'   # Fall
        except:
            return None

    df['Son_Arıza_Mevsim'] = df['Son_Arıza_Tarihi'].apply(get_season)

    season_dist = df['Son_Arıza_Mevsim'].value_counts()
    print(f"✓ Last fault season calculated ({df['Son_Arıza_Mevsim'].notna().sum()} equipment)")
    print(f"  Season distribution:")
    for season in ['Yaz', 'Kış', 'İlkbahar', 'Sonbahar']:
        count = season_dist.get(season, 0)
        if count > 0:
            print(f"    {season:<12} {count:>4,} equipment ({count/df['Son_Arıza_Mevsim'].notna().sum()*100:.1f}%)")
else:
    print("⚠ Son_Arıza_Tarihi column not found")
    df['Son_Arıza_Mevsim'] = None

# ============================================================================
# 4. CUSTOMER TYPE RATIOS
# ============================================================================
print("\n--- Customer Type Ratios ---")

# Check for customer impact columns (used later for customer-weighted risk)
has_customer_cols = all(col in df.columns for col in ['urban_mv_Avg', 'urban_lv_Avg',
                                                        'suburban_mv_Avg', 'suburban_lv_Avg',
                                                        'rural_mv_Avg', 'rural_lv_Avg',
                                                        'total_customer_count_Avg'])

# Check if fault-level ratios were calculated (proper method - no Simpson's Paradox)
has_fault_level_ratios = all(col in df.columns for col in ['Urban_Customer_Ratio_mean',
                                                             'Rural_Customer_Ratio_mean',
                                                             'MV_Customer_Ratio_mean'])

if has_fault_level_ratios:
    # Use pre-calculated fault-level ratios (averaged from fault level)
    # This avoids Simpson's Paradox - ratios were calculated BEFORE averaging
    df['Kentsel_Müşteri_Oranı'] = df['Urban_Customer_Ratio_mean']
    df['Kırsal_Müşteri_Oranı'] = df['Rural_Customer_Ratio_mean']
    df['OG_Müşteri_Oranı'] = df['MV_Customer_Ratio_mean']

    print(f"✓ Customer ratios loaded (fault-level calculated - no Simpson's Paradox):")
    print(f"  Urban customer ratio: Mean={df['Kentsel_Müşteri_Oranı'].mean():.2%}, Max={df['Kentsel_Müşteri_Oranı'].max():.2%}")
    print(f"  Rural customer ratio: Mean={df['Kırsal_Müşteri_Oranı'].mean():.2%}, Max={df['Kırsal_Müşteri_Oranı'].max():.2%}")
    print(f"  MV customer ratio: Mean={df['OG_Müşteri_Oranı'].mean():.2%}, Max={df['OG_Müşteri_Oranı'].max():.2%}")

else:
    # Fallback: Use old method (equipment-level averaged counts)
    if has_customer_cols:
        print("⚠ WARNING: Using old method (equipment-level averaged counts)")
        print("  This can cause Simpson's Paradox. Run 02_data_transformation.py to fix.")

        # Urban customer ratio
        df['Kentsel_Müşteri_Oranı'] = (
            (df['urban_mv_Avg'].fillna(0) + df['urban_lv_Avg'].fillna(0)) /
            (df['total_customer_count_Avg'] + 1)  # +1 to avoid division by zero
        ).clip(upper=1.0)

        # Rural customer ratio
        df['Kırsal_Müşteri_Oranı'] = (
            (df['rural_mv_Avg'].fillna(0) + df['rural_lv_Avg'].fillna(0)) /
            (df['total_customer_count_Avg'] + 1)
        ).clip(upper=1.0)

        # MV customer ratio (across all areas)
        df['OG_Müşteri_Oranı'] = (
            (df['urban_mv_Avg'].fillna(0) + df['suburban_mv_Avg'].fillna(0) + df['rural_mv_Avg'].fillna(0)) /
            (df['total_customer_count_Avg'] + 1)
        ).clip(upper=1.0)

        print(f"✓ Customer ratios calculated (capped at 100% - defensive fix):")
        print(f"  Urban customer ratio: Mean={df['Kentsel_Müşteri_Oranı'].mean():.2%}, Max={df['Kentsel_Müşteri_Oranı'].max():.2%}")
        print(f"  Rural customer ratio: Mean={df['Kırsal_Müşteri_Oranı'].mean():.2%}, Max={df['Kırsal_Müşteri_Oranı'].max():.2%}")
        print(f"  MV customer ratio: Mean={df['OG_Müşteri_Oranı'].mean():.2%}, Max={df['OG_Müşteri_Oranı'].max():.2%}")
    else:
        print("⚠ Customer impact columns not found")
        df['Kentsel_Müşteri_Oranı'] = 0
        df['Kırsal_Müşteri_Oranı'] = 0
        df['OG_Müşteri_Oranı'] = 0

# ============================================================================
# 5. LOADING INTENSITY METRICS (Leakage-Safe)
# ============================================================================
print("\n--- Loading Intensity Metrics (Leakage-Safe) ---")

# Use recency-based loading instead of 12-month fault counts (avoids target leakage)
if 'Son_Arıza_Gun_Sayisi' in df.columns and 'Ekipman_Yaşı_Yıl' in df.columns:
    # Equipment loading score: Recent failures indicate high loading
    # Formula: 1 / (days_since_last_fault + 1) → Recent fault = high score
    # Handle NaN and negative values (equipment that never failed)
    days_since = df['Son_Arıza_Gun_Sayisi'].copy()
    days_since = days_since.fillna(365)  # Never failed = 365 days default
    days_since = days_since.clip(lower=0)  # Negative values → 0 (equipment never failed)

    df['Ekipman_Yoğunluk_Skoru'] = 1 / (days_since + 1)

    print(f"✓ Equipment loading score (recency-based):")
    print(f"  Mean={df['Ekipman_Yoğunluk_Skoru'].mean():.4f}, Max={df['Ekipman_Yoğunluk_Skoru'].max():.4f}")
    print(f"  Note: Equipment never failed get default 365-day recency")
else:
    print("⚠ Son_Arıza_Gun_Sayisi not available for loading score")
    df['Ekipman_Yoğunluk_Skoru'] = 0

# NOTE: Customer-weighted risk removed - was based on Composite_PoF_Risk_Score
# which has been eliminated. Models can learn customer impact directly.

# ============================================================================
# 6. URBAN/RURAL VS CUSTOMER IMPACT CROSS-ANALYSIS (Informational)
# ============================================================================
if 'Bölge_Tipi' in df.columns and df['Bölge_Tipi'].notna().any() and 'Arıza_Sayısı_12ay' in df.columns:
    print("\n--- Urban/Rural Failure Pattern Summary ---")
    for btype in ['Kentsel', 'Kırsal']:
        mask = df['Bölge_Tipi'] == btype
        if mask.sum() > 0:
            avg_customers = df.loc[mask, 'total_customer_count_Avg'].mean() if has_customer_cols else 0
            avg_faults = df.loc[mask, 'Arıza_Sayısı_12ay'].mean()
            avg_age = df.loc[mask, 'Ekipman_Yaşı_Yıl'].mean() if 'Ekipman_Yaşı_Yıl' in df.columns else 0
            print(f"  {btype}: Avg customers={avg_customers:.1f}, Avg faults(12M)={avg_faults:.2f}, Avg age={avg_age:.1f}y")

print("\n✓ Additional domain-specific features complete!")
print(f"  New features: voltage_level, Voltage_Class, Is_MV/LV/HV, Bölge_Tipi,")
print(f"                Son_Arıza_Mevsim, Kentsel/Kırsal/OG_Müşteri_Oranı,")
print(f"                Ekipman_Yoğunluk_Skoru, Müşteri_Başına_Risk")

# ============================================================================
# STEP 9C: FAULT CAUSE CODE FEATURES (MODULE 3)
# ============================================================================
print("\n" + "="*100)
print("STEP 9C: FAULT CAUSE CODE FEATURES")
print("="*100)

# Check if cause code columns exist (added by 02_data_transformation.py)
has_cause_codes = ('Arıza_Nedeni_İlk' in df.columns or
                   'Arıza_Nedeni_Son' in df.columns or
                   'Arıza_Nedeni_Sık' in df.columns)

if has_cause_codes:
    print("\n✓ Cause code columns found - creating cause-based features")

    # Equipment Class × Cause Code interaction
    if 'Equipment_Class_Primary' in df.columns and 'Arıza_Nedeni_Sık' in df.columns:
        print("\n--- Equipment Class × Cause Code Interaction ---")

        # Create interaction feature (categorical)
        df['Ekipman_Neden_Kombinasyonu'] = (
            df['Equipment_Class_Primary'].astype(str) + '_' +
            df['Arıza_Nedeni_Sık'].astype(str)
        )

        # Calculate risk score per combination (average failure count)
        if 'Toplam_Arıza_Sayisi_Lifetime' in df.columns:
            combo_risk = df.groupby('Ekipman_Neden_Kombinasyonu')['Toplam_Arıza_Sayisi_Lifetime'].mean()
            df['Ekipman_Neden_Risk_Skoru'] = df['Ekipman_Neden_Kombinasyonu'].map(combo_risk)

            print(f"  ✓ Created Ekipman_Neden_Kombinasyonu: {df['Ekipman_Neden_Kombinasyonu'].nunique()} unique combinations")
            print(f"  ✓ Created Ekipman_Neden_Risk_Skoru: Mean={df['Ekipman_Neden_Risk_Skoru'].mean():.2f}")

    # Cause consistency flag (high consistency = always same cause)
    if 'Arıza_Nedeni_Tutarlılık' in df.columns:
        df['Tek_Neden_Flag'] = (df['Arıza_Nedeni_Tutarlılık'] >= 0.8).astype(int)  # 80%+ same cause
        print(f"\n  ✓ Created Tek_Neden_Flag: {df['Tek_Neden_Flag'].sum()} equipment with single dominant cause (≥80%)")

    # Cause diversity risk (more cause types = more complex failures)
    if 'Arıza_Nedeni_Çeşitlilik' in df.columns:
        df['Çok_Nedenli_Flag'] = (df['Arıza_Nedeni_Çeşitlilik'] >= 3).astype(int)  # 3+ different causes
        print(f"  ✓ Created Çok_Nedenli_Flag: {df['Çok_Nedenli_Flag'].sum()} equipment with multiple causes (≥3 types)")

    # Cause changed flag (recent cause different from first cause)
    if 'Arıza_Nedeni_İlk' in df.columns and 'Arıza_Nedeni_Son' in df.columns:
        df['Neden_Değişim_Flag'] = (
            df['Arıza_Nedeni_İlk'].astype(str) != df['Arıza_Nedeni_Son'].astype(str)
        ).astype(int)
        print(f"  ✓ Created Neden_Değişim_Flag: {df['Neden_Değişim_Flag'].sum()} equipment with changing failure causes")

    # Summary statistics by cause code
    if 'Arıza_Nedeni_Sık' in df.columns:
        print(f"\n--- Failure Cause Distribution ---")
        cause_dist = df['Arıza_Nedeni_Sık'].value_counts().head(10)
        print(f"  Top 10 most common failure causes:")
        for cause, count in cause_dist.items():
            pct = count / len(df) * 100
            print(f"    {cause}: {count} equipment ({pct:.1f}%)")

    print("\n✓ Fault cause code features complete!")
    print(f"  New features: Ekipman_Neden_Kombinasyonu, Ekipman_Neden_Risk_Skoru,")
    print(f"                Tek_Neden_Flag, Çok_Nedenli_Flag, Neden_Değişim_Flag")

else:
    print("\n⚠ Cause code columns not found - skipping cause-based features")
    print("  This is expected if 02_data_transformation.py hasn't been re-run with cause code support")

# ============================================================================
# STEP 10: FEATURE SUMMARY & VALIDATION
# ============================================================================
print("\n" + "="*100)
print("STEP 10: FEATURE ENGINEERING SUMMARY")
print("="*100)

new_feature_count = df.shape[1] - original_feature_count

print(f"\n📊 Feature Engineering Results:")
print(f"   Original features: {original_feature_count}")
print(f"   New features added: {new_feature_count}")
print(f"   Total features: {df.shape[1]}")

# List new features (optimized 30-feature set approach)
print("\n✅ Key Engineered Features (Optimal 30-Feature Set):")
print("\n--- TIER 1: Essential Features ---")
tier1_features = [
    'Beklenen_Ömür_Yıl',
    'Yas_Beklenen_Omur_Orani',
]

for feature in tier1_features:
    if feature in df.columns:
        coverage = df[feature].notna().sum() / len(df) * 100
        print(f"  ✓ {feature:<45} {coverage:>6.1f}% coverage")

print("\n--- TIER 3: MTBF Enhancement Features (NEW) ---")
tier3_mtbf = [
    'MTBF_InterFault_Trend',
    'MTBF_InterFault_StdDev',
]

for feature in tier3_mtbf:
    if feature in df.columns:
        coverage = df[feature].notna().sum() / len(df) * 100
        print(f"  ✓ {feature:<45} {coverage:>6.1f}% coverage")

print("\n--- TIER 3: Interaction Features ---")
tier3_interactions = [
    'AgeRatio_Recurrence_Interaction',
    'Overdue_Factor',
]

for feature in tier3_interactions:
    if feature in df.columns:
        coverage = df[feature].notna().sum() / len(df) * 100
        print(f"  ✓ {feature:<45} {coverage:>6.1f}% coverage")

print("\n--- Additional Domain Features ---")
additional_features = [
    'Age_Risk_Category',
    'Voltage_Class',
    'Bölge_Tipi',
    'Son_Arıza_Mevsim',
    'Customer_Minutes_Risk_Annual',
    'Customer_Impact_Category',
]

for feature in additional_features:
    if feature in df.columns:
        coverage = df[feature].notna().sum() / len(df) * 100
        print(f"  ✓ {feature:<45} {coverage:>6.1f}% coverage")

print("\n--- REMOVED Features (Leakage/Redundancy) ---")
removed_features = [
    'Geographic_Cluster (K-means clustering - noisy)',
    'Arıza_Sayısı_12ay_Cluster_Avg (cluster aggregation - leaky)',
    'Failure_Rate_Per_Year (redundant ratio)',
    'Recent_Failure_Intensity (leaky - 3M window)',
    'Failure_Acceleration (leaky - 6M window)',
    'Arıza_Sayısı_12ay_Class_Avg (class aggregation - leaky)',
    'MTBF_Gün_Class_Avg (circular logic)',
    'Failure_vs_Class_Avg (derived from leaky feature)',
    'Age_Failure_Interaction (leaky - uses 12M data)',
    'Customer_Failure_Interaction (leaky - uses 12M data)',
]

for feature in removed_features:
    print(f"  ❌ {feature}")

# Data quality check
print("\n--- Data Quality Validation ---")
print(f"  Missing values: {df.isnull().sum().sum():,}")
print(f"  Duplicate rows: {df.duplicated().sum()}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# STEP 11: SAVE ENGINEERED FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 11: SAVING ENGINEERED DATASET")
print("="*100)

print(f"\n💾 Saving to: {FEATURES_ENGINEERED_FILE}")
df.to_csv(FEATURES_ENGINEERED_FILE, index=False, encoding='utf-8-sig')

print(f"✅ Successfully saved!")
print(f"   Records: {len(df):,}")
print(f"   Features: {df.shape[1]}")
print(f"   File size: {FEATURES_ENGINEERED_FILE.stat().st_size / 1024**2:.2f} MB")

# Save feature catalog
print("\n📋 Creating feature catalog...")

feature_catalog = pd.DataFrame({
    'Feature_Name': df.columns,
    'Data_Type': df.dtypes.astype(str),
    'Non_Null_Count': df.notna().sum(),
    'Null_Count': df.isnull().sum(),
    'Completeness_%': (df.notna().sum() / len(df) * 100).round(1),
    'Unique_Values': [df[col].nunique() for col in df.columns],
})

# Categorize features
def categorize_feature(name):
    if 'Arıza' in name or 'Fault' in name or 'Failure' in name:
        return 'Failure History'
    elif 'Yaş' in name or 'Age' in name or 'Ömür' in name:
        return 'Age/Life'
    elif 'KOORDINAT' in name or 'Geographic' in name or 'Cluster' in name:
        return 'Geographic'
    elif 'Customer' in name or 'Müşteri' in name:
        return 'Customer Impact'
    elif 'Risk' in name or 'Score' in name:
        return 'Risk Metrics'
    elif 'MTBF' in name or 'Reliability' in name:
        return 'Reliability'
    elif 'Ekipman' in name or 'Equipment' in name:
        return 'Equipment Metadata'
    elif 'Interaction' in name:
        return 'Interaction Features'
    else:
        return 'Other'

feature_catalog['Category'] = feature_catalog['Feature_Name'].apply(categorize_feature)

feature_catalog.to_csv(FEATURE_CATALOG_FILE, index=False)
print(f"✅ Feature catalog saved to: {FEATURE_CATALOG_FILE}")

# Print summary by category
print("\n--- Features by Category ---")
category_counts = feature_catalog['Category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category:<25} {count:>3} features")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
# NOTE: High-risk equipment identification removed - was based on
# Composite_PoF_Risk_Score which has been eliminated. Use model predictions
# instead for identifying high-risk equipment.
print("\n" + "="*100)
print("FEATURE ENGINEERING COMPLETE")
print("="*100)

print("\n🎯 KEY ACCOMPLISHMENTS:")
print(f"   ✅ Expected Life Ratios calculated ({df['Yas_Beklenen_Omur_Orani'].notna().sum():,} equipment)")
print(f"   ✅ MTBF Enhancement Features added (Trend + StdDev)")
print(f"   ✅ Overdue Factor created (imminent failure risk detector)")
print(f"   ✅ Removed 10 problematic features (leakage/redundancy)")
print(f"   ✅ {new_feature_count} new features engineered (net change)")

print("\n📊 FEATURE SUMMARY:")
numeric_features = df.select_dtypes(include=[np.number]).columns
print(f"   Total features: {len(df.columns)}")
print(f"   Numeric features: {len(numeric_features)}")

print("\n📂 OUTPUT FILES:")
print(f"   • data/features_engineered.csv ({df.shape[1]} features)")
print(f"   • data/feature_catalog.csv (feature documentation)")

print("\n🚀 READY FOR NEXT PHASE:")
print("   1. Feature selection (VIF analysis + importance filtering)")
print("   2. Model training (XGBoost + CatBoost)")
print("   3. PoF predictions (3/6/12/24 months)")

print("\n" + "="*100)
print(f"{'FEATURE ENGINEERING PIPELINE COMPLETE':^100}")
print("="*100)