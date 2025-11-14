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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
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

data_path = Path('data/equipment_level_data.csv')

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("Please run 02_data_transformation.py first!")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
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
# STEP 3: GEOGRAPHIC CLUSTERING
# ============================================================================
print("\n" + "="*100)
print("STEP 3: CREATING GEOGRAPHIC CLUSTERS")
print("="*100)

if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    # Get equipment with valid coordinates
    has_coords = df['KOORDINAT_X'].notna() & df['KOORDINAT_Y'].notna()
    coord_count = has_coords.sum()
    
    print(f"\n✓ Equipment with coordinates: {coord_count:,} ({coord_count/len(df)*100:.1f}%)")
    
    if coord_count > 10:  # Need minimum equipment for clustering
        # Prepare coordinate data
        coords = df.loc[has_coords, ['KOORDINAT_X', 'KOORDINAT_Y']].values
        
        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Determine optimal number of clusters (use elbow method heuristic)
        # For ~1,300 equipment, 15-20 clusters is reasonable
        n_clusters = min(20, coord_count // 50)  # ~50 equipment per cluster
        
        print(f"  Creating {n_clusters} geographic clusters...")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df.loc[has_coords, 'Geographic_Cluster'] = kmeans.fit_predict(coords_scaled)
        
        # For equipment without coordinates, assign -1
        df['Geographic_Cluster'] = df['Geographic_Cluster'].fillna(-1).astype(int)
        
        print(f"✓ Geographic clusters created")
        
        # Cluster statistics
        cluster_sizes = df[df['Geographic_Cluster'] >= 0]['Geographic_Cluster'].value_counts()
        print(f"  Average equipment per cluster: {cluster_sizes.mean():.1f}")
        print(f"  Min cluster size: {cluster_sizes.min()}")
        print(f"  Max cluster size: {cluster_sizes.max()}")
        
        # Calculate cluster-level failure rates
        print("\n  Calculating cluster-level failure rates...")
        cluster_stats = df.groupby('Geographic_Cluster').agg({
            'Arıza_Sayısı_12ay': 'mean',
            'Tekrarlayan_Arıza_90gün_Flag': 'mean',
            'MTBF_Gün': 'mean'
        }).add_suffix('_Cluster_Avg')
        
        # Merge back to main dataframe
        df = df.merge(cluster_stats, left_on='Geographic_Cluster', right_index=True, how='left')
        
        print("✓ Cluster-level features added")
    else:
        print("⚠ Too few equipment with coordinates for clustering")
        df['Geographic_Cluster'] = -1
else:
    print("\n⚠ WARNING: Coordinate columns not found, skipping geographic clustering")
    df['Geographic_Cluster'] = -1

# ============================================================================
# STEP 4: FAILURE RATE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 4: ENGINEERING FAILURE RATE FEATURES")
print("="*100)

print("\n--- Calculating Failure Rates ---")

# Annualized failure rate
if 'Toplam_Arıza_Sayisi_Lifetime' in df.columns and 'Ekipman_Yaşı_Yıl' in df.columns:
    df['Failure_Rate_Per_Year'] = df.apply(
        lambda row: row['Toplam_Arıza_Sayisi_Lifetime'] / row['Ekipman_Yaşı_Yıl'] 
        if pd.notna(row['Ekipman_Yaşı_Yıl']) and row['Ekipman_Yaşı_Yıl'] > 0 
        else None, 
        axis=1
    )
    
    rate_available = df['Failure_Rate_Per_Year'].notna().sum()
    print(f"✓ Annual failure rate: {rate_available:,} equipment")
    
    if rate_available > 0:
        print(f"  Mean: {df['Failure_Rate_Per_Year'].mean():.2f} faults/year")
        print(f"  Median: {df['Failure_Rate_Per_Year'].median():.2f} faults/year")

# Recent failure intensity (3-month to 12-month ratio)
if 'Arıza_Sayısı_3ay' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
    df['Recent_Failure_Intensity'] = df.apply(
        lambda row: row['Arıza_Sayısı_3ay'] / row['Arıza_Sayısı_12ay']
        if pd.notna(row['Arıza_Sayısı_12ay']) and row['Arıza_Sayısı_12ay'] > 0
        else 0,
        axis=1
    )
    print(f"✓ Recent failure intensity calculated")

# Failure acceleration (comparing 6-month periods)
if 'Arıza_Sayısı_6ay' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
    df['Failure_Acceleration'] = df.apply(
        lambda row: (row['Arıza_Sayısı_6ay'] * 2) / row['Arıza_Sayısı_12ay']
        if pd.notna(row['Arıza_Sayısı_12ay']) and row['Arıza_Sayısı_12ay'] > 0
        else 1.0,
        axis=1
    )
    print(f"✓ Failure acceleration calculated")

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
# STEP 7: EQUIPMENT CLASS AGGREGATIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 7: CREATING EQUIPMENT CLASS AGGREGATIONS")
print("="*100)

if 'Equipment_Class_Primary' in df.columns:
    print("\n--- Calculating class-level benchmarks ---")

    # Group by equipment class (using unified Equipment_Class_Primary)
    class_stats = df.groupby('Equipment_Class_Primary').agg({
        'Arıza_Sayısı_12ay': 'mean',
        'MTBF_Gün': 'mean',
        'Ekipman_Yaşı_Yıl': 'mean',
        'Yas_Beklenen_Omur_Orani': 'mean'
    }).add_suffix('_Class_Avg')
    
    # Merge back
    df = df.merge(class_stats, left_on='Equipment_Class_Primary', right_index=True, how='left')
    
    print(f"✓ Class-level benchmarks added for {len(class_stats)} equipment types")
    
    # Relative performance vs. class average
    if 'Arıza_Sayısı_12ay_Class_Avg' in df.columns:
        df['Failure_vs_Class_Avg'] = df.apply(
            lambda row: row['Arıza_Sayısı_12ay'] / row['Arıza_Sayısı_12ay_Class_Avg']
            if pd.notna(row['Arıza_Sayısı_12ay_Class_Avg']) and row['Arıza_Sayısı_12ay_Class_Avg'] > 0
            else 1.0,
            axis=1
        )
        
        # Performance categories
        better = (df['Failure_vs_Class_Avg'] < 0.8).sum()
        worse = (df['Failure_vs_Class_Avg'] > 1.2).sum()
        print(f"  Equipment performing better than class average: {better:,}")
        print(f"  Equipment performing worse than class average: {worse:,}")

# ============================================================================
# STEP 8: COMPOSITE RISK SCORES
# ============================================================================
print("\n" + "="*100)
print("STEP 8: CALCULATING COMPOSITE RISK SCORES")
print("="*100)

print("\n--- Building PoF Risk Score (0-100) ---")

# Components of risk score
risk_components = []

# 1. Age risk (30% weight)
if 'Yas_Beklenen_Omur_Orani' in df.columns:
    df['Age_Risk_Score'] = df['Yas_Beklenen_Omur_Orani'].clip(0, 2) * 50  # 0-100 scale
    risk_components.append(('Age_Risk_Score', 0.30))
    print("  ✓ Age risk (30% weight)")

# 2. Recent failure history (40% weight)
if 'Arıza_Sayısı_6ay' in df.columns:
    # Normalize to 0-100 (assume 5+ failures = max risk)
    df['Recent_Failure_Risk_Score'] = df['Arıza_Sayısı_6ay'].clip(0, 5) * 20
    risk_components.append(('Recent_Failure_Risk_Score', 0.40))
    print("  ✓ Recent failure risk (40% weight)")

# 3. Reliability degradation (20% weight)
if 'MTBF_Gün' in df.columns:
    # Inverse MTBF score (lower MTBF = higher risk)
    df['MTBF_Risk_Score'] = df['MTBF_Gün'].apply(
        lambda x: max(0, 100 - (x / 365 * 100)) if pd.notna(x) else 50
    )
    risk_components.append(('MTBF_Risk_Score', 0.20))
    print("  ✓ MTBF risk (20% weight)")

# 4. Recurrence pattern (10% weight)
if 'Tekrarlayan_Arıza_90gün_Flag' in df.columns:
    df['Recurrence_Risk_Score'] = df['Tekrarlayan_Arıza_90gün_Flag'] * 100
    risk_components.append(('Recurrence_Risk_Score', 0.10))
    print("  ✓ Recurrence risk (10% weight)")

# Calculate composite score
if risk_components:
    df['Composite_PoF_Risk_Score'] = 0
    for component, weight in risk_components:
        df['Composite_PoF_Risk_Score'] += df[component].fillna(50) * weight
    
    # Ensure 0-100 range
    df['Composite_PoF_Risk_Score'] = df['Composite_PoF_Risk_Score'].clip(0, 100)
    
    print(f"\n✓ Composite PoF Risk Score calculated")
    
    # Risk distribution
    risk_stats = df['Composite_PoF_Risk_Score'].describe()
    print(f"  Mean: {risk_stats['mean']:.1f}")
    print(f"  Median: {risk_stats['50%']:.1f}")
    
    # Risk categories
    df['Risk_Category'] = pd.cut(
        df['Composite_PoF_Risk_Score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low (0-25)', 'Medium (25-50)', 'High (50-75)', 'Critical (75-100)']
    )
    
    print("\n  Risk Distribution:")
    for category in ['Low (0-25)', 'Medium (25-50)', 'High (50-75)', 'Critical (75-100)']:
        count = (df['Risk_Category'] == category).sum()
        pct = count / len(df) * 100
        icon = "✅" if "Low" in category else ("⚠" if "Medium" in category else "❌")
        print(f"    {icon} {category}: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 9: INTERACTION FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 9: CREATING INTERACTION FEATURES")
print("="*100)

# Age × Failure interaction
if 'Ekipman_Yaşı_Yıl' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
    df['Age_Failure_Interaction'] = df['Ekipman_Yaşı_Yıl'] * df['Arıza_Sayısı_12ay']
    print("✓ Age × Failure interaction")

# Age ratio × Recurrence interaction
if 'Yas_Beklenen_Omur_Orani' in df.columns and 'Tekrarlayan_Arıza_90gün_Flag' in df.columns:
    df['AgeRatio_Recurrence_Interaction'] = df['Yas_Beklenen_Omur_Orani'] * df['Tekrarlayan_Arıza_90gün_Flag']
    print("✓ Age Ratio × Recurrence interaction")

# Customer impact × Failure rate interaction
if 'Avg_Customer_Count' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
    df['Customer_Failure_Interaction'] = df['Avg_Customer_Count'] * df['Arıza_Sayısı_12ay']
    print("✓ Customer × Failure interaction")

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

# Check for customer impact columns
has_customer_cols = all(col in df.columns for col in ['urban_mv_Avg', 'urban_lv_Avg',
                                                        'suburban_mv_Avg', 'suburban_lv_Avg',
                                                        'rural_mv_Avg', 'rural_lv_Avg',
                                                        'total_customer_count_Avg'])

if has_customer_cols:
    # Urban customer ratio
    df['Kentsel_Müşteri_Oranı'] = (
        (df['urban_mv_Avg'].fillna(0) + df['urban_lv_Avg'].fillna(0)) /
        (df['total_customer_count_Avg'] + 1)  # +1 to avoid division by zero
    )

    # Rural customer ratio
    df['Kırsal_Müşteri_Oranı'] = (
        (df['rural_mv_Avg'].fillna(0) + df['rural_lv_Avg'].fillna(0)) /
        (df['total_customer_count_Avg'] + 1)
    )

    # MV customer ratio (across all areas)
    df['OG_Müşteri_Oranı'] = (
        (df['urban_mv_Avg'].fillna(0) + df['suburban_mv_Avg'].fillna(0) + df['rural_mv_Avg'].fillna(0)) /
        (df['total_customer_count_Avg'] + 1)
    )

    print(f"✓ Customer ratios calculated:")
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

# Customer-weighted risk (consequence metric, not PoF)
if 'Composite_PoF_Risk_Score' in df.columns and has_customer_cols:
    df['Müşteri_Başına_Risk'] = df['Composite_PoF_Risk_Score'] / (df['total_customer_count_Avg'] + 1)
    print(f"✓ Customer-weighted risk:")
    print(f"  Mean={df['Müşteri_Başına_Risk'].mean():.3f}, Max={df['Müşteri_Başına_Risk'].max():.3f}")
else:
    df['Müşteri_Başına_Risk'] = 0

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

# List new features
print("\n✅ New Features Created:")
new_features = [
    'Beklenen_Ömür_Yıl',
    'Yas_Beklenen_Omur_Orani',
    'Age_Risk_Category',
    'Geographic_Cluster',
    'Failure_Rate_Per_Year',
    'Recent_Failure_Intensity',
    'Failure_Acceleration',
    'Reliability_Score',
    'Time_Since_Last_Normalized',
    'Failure_Free_3M',
    'Customer_Minutes_Risk_Annual',
    'Customer_Impact_Category',
    'Arıza_Sayısı_12ay_Class_Avg',
    'Failure_vs_Class_Avg',
    'Composite_PoF_Risk_Score',
    'Risk_Category',
    'Age_Failure_Interaction',
]

for i, feature in enumerate(new_features, 1):
    if feature in df.columns:
        coverage = df[feature].notna().sum() / len(df) * 100
        print(f"  {i:2d}. {feature:<40} {coverage:>6.1f}% coverage")

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

output_path = Path('data/features_engineered.csv')

print(f"\n💾 Saving to: {output_path}")
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ Successfully saved!")
print(f"   Records: {len(df):,}")
print(f"   Features: {df.shape[1]}")
print(f"   File size: {output_path.stat().st_size / 1024**2:.2f} MB")

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

feature_catalog.to_csv('data/feature_catalog.csv', index=False)
print(f"✅ Feature catalog saved to: data/feature_catalog.csv")

# Print summary by category
print("\n--- Features by Category ---")
category_counts = feature_catalog['Category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category:<25} {count:>3} features")

# ============================================================================
# STEP 12: HIGH-RISK EQUIPMENT IDENTIFICATION
# ============================================================================
print("\n" + "="*100)
print("STEP 12: HIGH-RISK EQUIPMENT IDENTIFICATION")
print("="*100)

if 'Risk_Category' in df.columns:
    high_risk = df[df['Risk_Category'].isin(['High (50-75)', 'Critical (75-100)'])]
    print(f"\n⚠️  Total high-risk equipment: {len(high_risk):,} ({len(high_risk)/len(df)*100:.1f}%)")
    
    if len(high_risk) > 0:
        print("\n--- High-Risk Equipment by Class ---")
        risk_by_class = high_risk['Equipment_Class_Primary'].value_counts().head(10)
        for eq_class, count in risk_by_class.items():
            pct = count / len(high_risk) * 100
            print(f"  {eq_class:<30} {count:>4,} ({pct:>5.1f}%)")
        
        # Save high-risk equipment list
        output_cols = ['Ekipman_ID', 'Equipment_Class_Primary', 'Ekipman_Yaşı_Yıl',
                      'Yas_Beklenen_Omur_Orani', 'Arıza_Sayısı_12ay',
                      'Composite_PoF_Risk_Score', 'Risk_Category']

        # Add optional columns if they exist
        if 'KOORDINAT_X' in high_risk.columns:
            output_cols.append('KOORDINAT_X')
        if 'KOORDINAT_Y' in high_risk.columns:
            output_cols.append('KOORDINAT_Y')
        if 'İlçe' in high_risk.columns:
            output_cols.append('İlçe')

        high_risk_output = high_risk[output_cols].sort_values('Composite_PoF_Risk_Score', ascending=False)
        
        high_risk_output.to_csv('data/high_risk_equipment.csv', index=False)
        print(f"\n✅ High-risk equipment list saved to: data/high_risk_equipment.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("FEATURE ENGINEERING COMPLETE")
print("="*100)

print("\n🎯 KEY ACCOMPLISHMENTS:")
print(f"   ✅ Expected Life Ratios calculated ({df['Yas_Beklenen_Omur_Orani'].notna().sum():,} equipment)")
if 'Geographic_Cluster' in df.columns:
    print(f"   ✅ Geographic Clusters created ({(df['Geographic_Cluster'] >= 0).sum():,} equipment)")
print(f"   ✅ Composite Risk Scores computed (all equipment)")
print(f"   ✅ {new_feature_count} new features engineered")

print("\n📊 RISK DISTRIBUTION:")
if 'Risk_Category' in df.columns:
    for category in ['Low (0-25)', 'Medium (25-50)', 'High (50-75)', 'Critical (75-100)']:
        count = (df['Risk_Category'] == category).sum()
        pct = count / len(df) * 100
        icon = "✅" if "Low" in category else ("⚠" if "Medium" in category else "❌")
        print(f"   {icon} {category:<20} {count:>4,} equipment ({pct:>5.1f}%)")

print("\n📂 OUTPUT FILES:")
print(f"   • data/features_engineered.csv ({df.shape[1]} features)")
print(f"   • data/feature_catalog.csv (feature documentation)")
if 'Risk_Category' in df.columns:
    high_risk_count = df[df['Risk_Category'].isin(['High (50-75)', 'Critical (75-100)'])].shape[0]
    if high_risk_count > 0:
        print(f"   • data/high_risk_equipment.csv ({high_risk_count:,} high-risk equipment)")

print("\n🚀 READY FOR NEXT PHASE:")
print("   1. Feature selection (VIF analysis + importance filtering)")
print("   2. Model training (XGBoost + CatBoost)")
print("   3. PoF predictions (3/6/12/24 months)")

print("\n" + "="*100)
print(f"{'FEATURE ENGINEERING PIPELINE COMPLETE':^100}")
print("="*100)