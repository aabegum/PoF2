"""COMPREHENSIVE DATA PROFILING - TURKISH EDAŞ POF PROJECT v2.0
Optimized for Your Actual Column Structure

Key Mappings:
- Equipment ID: HEPSI_ID (primary), cbs_id (fallback)
- Equipment Class: Equipment_Type (primary), Ekipman Sınıfı (secondary)
- Age Calculation: TESIS_TARIHI (primary), EDBS_IDATE (fallback)
- Fault Timestamp: started at
- Fault Classification: cause code, Kategori
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 60)

print("="*100)
print(" "*30 + "TURKISH EDAŞ EQUIPMENT DATA")
print(" "*25 + "COMPREHENSIVE PROFILING v2.0")
print("="*100)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "="*100)
print("SECTION 1: DATA LOADING & INITIAL INSPECTION")
print("="*100)

data_path = Path('data/combined_data.xlsx')

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("\nPlease ensure combined_data.xlsx is in the 'data' directory")
    exit(1)

print(f"\n✓ Found: {data_path}")
print(f"  File size: {data_path.stat().st_size / 1024**2:.2f} MB")

try:
    print(f"  Loading data...")
    df = pd.read_excel(data_path)
    print(f"✓ Successfully loaded!")
except Exception as e:
    print(f"❌ Error loading: {e}")
    exit(1)

print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Critical insight about data structure
print(f"\n⚠️  CRITICAL OBSERVATION:")
print(f"   This is FAULT-LEVEL data (each row = one fault event)")
print(f"   For PoF modeling, you'll need to transform to EQUIPMENT-LEVEL")
print(f"   (where each row = one piece of equipment with aggregated fault history)")

# ============================================================================
# 2. COLUMN INVENTORY
# ============================================================================
print("\n" + "="*100)
print("SECTION 2: COMPLETE COLUMN INVENTORY")
print("="*100)

print(f"\nTotal Columns: {len(df.columns)}")
print("\nAll Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:3d}. {col}")

print("\n--- Data Type Distribution ---")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# ============================================================================
# 3. CRITICAL COLUMN VERIFICATION
# ============================================================================
print("\n" + "="*100)
print("SECTION 3: CRITICAL COLUMN VERIFICATION FOR POF")
print("="*100)

# Define expected columns based on your actual data
expected_columns = {
    '🔑 Equipment Identifiers': {
        'columns': ['HEPSI_ID', 'cbs_id', 'Ekipman ID', 'Ekipman Kodu'],
        'importance': 'CRITICAL'
    },
    '🏷️ Equipment Classification': {
        'columns': ['Equipment_Type', 'Ekipman Sınıfı', 'Kesinti Ekipman Sınıfı', 'Ekipman Sınıf'],
        'importance': 'CRITICAL'
    },
    '📅 Installation/Age Data': {
        'columns': ['TESIS_TARIHI', 'EDBS_IDATE'],
        'importance': 'CRITICAL'
    },
    '⏰ Fault Timestamps': {
        'columns': ['started at', 'ended at', 'created', 'Başlangıç Tarihi', 'Tamamlanma Tarihi'],
        'importance': 'CRITICAL'
    },
    '🔍 Fault Classification': {
        'columns': ['Kategori', 'cause code', 'Kategori Tanımı', 'Kategori Kodu', 'Kesinti Nedeni'],
        'importance': 'HIGH'
    },
    '📍 Geographic Data': {
        'columns': ['KOORDINAT_X', 'KOORDINAT_Y', 'İl', 'İlçe', 'Mahalle', 'coordinate'],
        'importance': 'HIGH'
    },
    '👥 Customer Impact': {
        'columns': ['total customer count', 'affected transformer count'],
        'importance': 'HIGH'
    },
    '🔧 Maintenance Records': {
        'columns': ['Bakım Olanlar', 'Tamamlanma Tarihi'],
        'importance': 'MEDIUM'
    },
    '🏭 Equipment Brand/Manufacturer': {
        'columns': ['MARKA', 'FIRMA', 'MARKA_MODEL'],
        'importance': 'MEDIUM'
    },
    '⏱️ Response Times': {
        'columns': ['Ekip Atama Zamanı', 'Ulaşma Zamanı', 'Yola Çıkma Zamanı'],
        'importance': 'LOW'
    }
}

actual_columns = set(df.columns)

for category, info in expected_columns.items():
    print(f"\n{category} [{info['importance']}]")
    found_any = False
    
    for col in info['columns']:
        if col in actual_columns:
            found_any = True
            non_null = df[col].notna().sum()
            non_null_pct = (non_null / len(df) * 100)
            
            if non_null_pct > 90:
                status = "✅"
            elif non_null_pct > 70:
                status = "✓"
            elif non_null_pct > 50:
                status = "⚠"
            else:
                status = "❌"
            
            print(f"  {status} {col}: {non_null_pct:.1f}% ({non_null:,} records)")
    
    if not found_any:
        print(f"  ❌ No columns found in this category!")

# ============================================================================
# 4. EQUIPMENT AGE CALCULATION ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("SECTION 4: EQUIPMENT AGE CALCULATION STRATEGY")
print("="*100)

current_year = datetime.now().year
print(f"\n📅 Current Year: {current_year}")

age_sources = {
    'TESIS_TARIHI': 'Primary installation date',
    'EDBS_IDATE': 'Fallback installation date'
}

print("\n--- Installation Date Column Analysis ---")

for col, description in age_sources.items():
    if col in df.columns:
        # Convert to datetime
        if df[col].dtype != 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        non_null = df[col].notna().sum()
        non_null_pct = (non_null / len(df) * 100)
        
        print(f"\n{col} ({description}):")
        print(f"  Coverage: {non_null_pct:.1f}% ({non_null:,} records)")
        
        if non_null > 0:
            valid_dates = df[col].dropna()
            year_range = f"{valid_dates.dt.year.min():.0f} to {valid_dates.dt.year.max():.0f}"
            print(f"  Year Range: {year_range}")
            
            # Calculate sample ages
            sample_ages = current_year - valid_dates.dt.year
            print(f"  Sample Ages:")
            print(f"    Mean: {sample_ages.mean():.1f} years")
            print(f"    Median: {sample_ages.median():.1f} years")
            print(f"    Range: {sample_ages.min():.0f} to {sample_ages.max():.0f} years")

# Combined coverage
print("\n🎯 AGE CALCULATION STRATEGY:")
if 'TESIS_TARIHI' in df.columns and 'EDBS_IDATE' in df.columns:
    tesis_coverage = (df['TESIS_TARIHI'].notna().sum() / len(df) * 100)
    edbs_coverage = (df['EDBS_IDATE'].notna().sum() / len(df) * 100)
    
    combined = df['TESIS_TARIHI'].notna() | df['EDBS_IDATE'].notna()
    combined_coverage = (combined.sum() / len(df) * 100)
    
    print(f"\n  1. PRIMARY SOURCE: TESIS_TARIHI ({tesis_coverage:.1f}% coverage)")
    print(f"  2. FALLBACK SOURCE: EDBS_IDATE ({edbs_coverage:.1f}% coverage)")
    print(f"  3. CALCULATION: Equipment_Age = {current_year} - Installation_Year")
    print(f"\n  📊 COMBINED COVERAGE: {combined_coverage:.1f}%")
    
    if combined_coverage > 90:
        print(f"     ✅ EXCELLENT: Can calculate age for >{combined_coverage:.0f}% of equipment")
    elif combined_coverage > 70:
        print(f"     ✓ GOOD: Can calculate age for ~{combined_coverage:.0f}% of equipment")
    else:
        print(f"     ⚠ LIMITED: Only {combined_coverage:.0f}% age coverage")

# ============================================================================
# 5. EQUIPMENT CLASS DISTRIBUTION
# ============================================================================
print("\n" + "="*100)
print("SECTION 5: EQUIPMENT CLASS DISTRIBUTION ANALYSIS")
print("="*100)

equipment_class_cols = ['Equipment_Type', 'Ekipman Sınıfı', 'Kesinti Ekipman Sınıfı', 'Ekipman Sınıf']
available_class_cols = [col for col in equipment_class_cols if col in df.columns]

print(f"\n📋 Found {len(available_class_cols)} equipment class columns")

# Determine best column
best_col = None
best_coverage = 0

for col in available_class_cols:
    coverage = (df[col].notna().sum() / len(df) * 100)
    if coverage > best_coverage:
        best_coverage = coverage
        best_col = col

if best_col:
    print(f"\n🎯 RECOMMENDED PRIMARY COLUMN: '{best_col}' ({best_coverage:.1f}% coverage)")
    
    # Show detailed distribution
    print(f"\n--- {best_col} Distribution ---")
    value_counts = df[best_col].value_counts()
    total = len(df)
    
    print(f"\nTotal Unique Equipment Types: {len(value_counts)}")
    print(f"\nTop 15 Equipment Types:")
    print(f"{'Equipment Type':<30} {'Count':>10} {'Percentage':>12}")
    print("-" * 55)
    
    for val, count in value_counts.head(15).items():
        pct = (count / total * 100)
        print(f"{str(val):<30} {count:>10,} {pct:>11.1f}%")
    
    if len(value_counts) > 15:
        others_count = value_counts.iloc[15:].sum()
        others_pct = (others_count / total * 100)
        print(f"{'Others (combined)':<30} {others_count:>10,} {others_pct:>11.1f}%")
    
    # Check for key equipment types
    print(f"\n--- Key Equipment Types for Use Cases ---")
    target_equipment = {
        'Trafo': ['Trafo', 'OG/AG Trafo', 'Trafo Bina'],
        'Kesici': ['Kesici'],
        'Ayırıcı': ['Ayırıcı'],
        'Box': ['Box', 'AG Pano Box'],
        'SDK': ['SDK']
    }
    
    for equip_name, search_terms in target_equipment.items():
        total_count = 0
        for term in search_terms:
            mask = df[best_col].astype(str).str.contains(term, case=False, na=False)
            total_count += mask.sum()
        
        if total_count > 0:
            pct = (total_count / len(df) * 100)
            print(f"  ✓ {equip_name}: {total_count:,} records ({pct:.1f}%)")
        else:
            print(f"  ⚠ {equip_name}: Not found (may use different naming)")

# ============================================================================
# 6. FAULT TIMESTAMP ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("SECTION 6: FAULT TIMESTAMP ANALYSIS")
print("="*100)

fault_timestamp_col = 'started at'

if fault_timestamp_col in df.columns:
    # Convert to datetime if needed
    if df[fault_timestamp_col].dtype != 'datetime64[ns]':
        df[fault_timestamp_col] = pd.to_datetime(df[fault_timestamp_col], errors='coerce')
    
    non_null = df[fault_timestamp_col].notna().sum()
    non_null_pct = (non_null / len(df) * 100)
    
    print(f"\n✓ Fault Timestamp Column: '{fault_timestamp_col}'")
    print(f"  Coverage: {non_null_pct:.1f}% ({non_null:,} records)")
    
    if non_null > 0:
        valid_timestamps = df[fault_timestamp_col].dropna()
        min_date = valid_timestamps.min()
        max_date = valid_timestamps.max()
        span_days = (max_date - min_date).days
        
        print(f"\n  📅 Temporal Coverage:")
        print(f"    First Fault: {min_date}")
        print(f"    Last Fault:  {max_date}")
        print(f"    Total Span:  {span_days:,} days ({span_days/365:.1f} years)")
        
        # Faults by year
        df['_fault_year'] = df[fault_timestamp_col].dt.year
        df['_fault_month'] = df[fault_timestamp_col].dt.month
        
        print(f"\n  📊 Fault Distribution by Year:")
        year_counts = df['_fault_year'].value_counts().sort_index()
        for year, count in year_counts.items():
            if not pd.isna(year):
                pct = (count / len(df) * 100)
                print(f"    {int(year)}: {count:>5,} faults ({pct:>5.1f}%)")
        
        # Seasonal distribution
        print(f"\n  🌡️ Fault Distribution by Month:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_counts = df['_fault_month'].value_counts().sort_index()
        for month, count in month_counts.items():
            if not pd.isna(month):
                pct = (count / len(df) * 100)
                month_idx = int(month) - 1
                print(f"    {month_names[month_idx]}: {count:>5,} faults ({pct:>5.1f}%)")
    
    print(f"\n🎯 FAILURE HISTORY CALCULATION FEASIBILITY:")
    if non_null_pct > 95:
        print(f"  ✅ EXCELLENT: Can reliably calculate:")
        print(f"     • Arıza_Sayısı_3ay (3-month failure count)")
        print(f"     • Arıza_Sayısı_6ay (6-month failure count)")
        print(f"     • Arıza_Sayısı_12ay (12-month failure count)")
        print(f"     • MTBF_Gün (Mean Time Between Failures)")
        print(f"     • Tekrarlayan_Arıza flags (30/90 day recurring patterns)")
        print(f"     • Son_Arıza_Gun_Sayisi (days since last fault)")
    elif non_null_pct > 80:
        print(f"  ✓ GOOD: Can calculate most failure metrics")
    else:
        print(f"  ⚠ LIMITED: {non_null_pct:.1f}% coverage may limit reliability")

# ============================================================================
# 7. EQUIPMENT IDENTIFICATION STRATEGY
# ============================================================================
print("\n" + "="*100)
print("SECTION 7: EQUIPMENT IDENTIFICATION STRATEGY")
print("="*100)

id_columns = ['HEPSI_ID', 'cbs_id', 'Ekipman ID', 'Ekipman Kodu']
available_id_cols = [col for col in id_columns if col in df.columns]

print(f"\n📋 Available Equipment ID Columns: {len(available_id_cols)}")

best_id_col = None
best_id_score = 0

for col in available_id_cols:
    non_null = df[col].notna().sum()
    non_null_pct = (non_null / len(df) * 100)
    unique = df[col].nunique()
    uniqueness_ratio = (unique / non_null * 100) if non_null > 0 else 0
    
    # Score based on coverage and uniqueness
    score = (non_null_pct * 0.6) + (min(uniqueness_ratio, 100) * 0.4)
    
    if score > best_id_score:
        best_id_score = score
        best_id_col = col
    
    print(f"\n{col}:")
    print(f"  Coverage: {non_null_pct:.1f}% ({non_null:,} records)")
    print(f"  Unique Values: {unique:,}")
    print(f"  Uniqueness Ratio: {uniqueness_ratio:.1f}%")
    
    if uniqueness_ratio > 90:
        print(f"  ✅ Excellent uniqueness - good for equipment tracking")
    elif uniqueness_ratio > 70:
        print(f"  ✓ Good uniqueness - usable for equipment ID")
    else:
        print(f"  ⚠ Low uniqueness - may have multiple faults per equipment (expected)")

if best_id_col:
    print(f"\n🎯 RECOMMENDED PRIMARY EQUIPMENT ID: '{best_id_col}'")
    print(f"   Score: {best_id_score:.1f}/100")
    
for col in id_columns:
    if col in df.columns:
        coverage = df[col].notna().sum()
        pct = coverage / len(df) * 100
        unique = df[col].nunique()
        uniqueness = unique / coverage * 100 if coverage > 0 else 0
        
        print(f"\n{col}:")
        print(f"  Coverage: {pct:.1f}% ({coverage:,} records)")
        print(f"  Unique Values: {unique:,}")
        print(f"  Uniqueness Ratio: {uniqueness:.1f}%")
        
        if uniqueness > 80:
            print(f"  ✓ Good uniqueness - usable for equipment ID")
        else:
            print(f"  ⚠ Low uniqueness - may have multiple faults per equipment (expected)")

# PRIMARY RECOMMENDATION: cbs_id → HEPSI_ID
print(f"\n🎯 RECOMMENDED PRIMARY EQUIPMENT ID: 'cbs_id'")
print(f"   Primary: cbs_id (100% coverage)")
print(f"   Fallback: HEPSI_ID (94.6% coverage)")
print(f"   Rationale: Domain expert recommendation - cbs_id most reliable")


# ============================================================================
# 8. FAULT CLASSIFICATION ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("SECTION 8: FAULT CLASSIFICATION READINESS")
print("="*100)

fault_class_cols = {
    'Kategori': 'High-level fault category',
    'cause code': 'Detailed fault cause code',
    'Kategori Tanımı': 'Category description',
    'Kesinti Nedeni': 'Outage reason'
}

print(f"\n--- Fault Classification Columns ---")

for col, description in fault_class_cols.items():
    if col in df.columns:
        non_null = df[col].notna().sum()
        non_null_pct = (non_null / len(df) * 100)
        unique = df[col].nunique()
        
        status = "✅" if non_null_pct > 90 else ("✓" if non_null_pct > 70 else "⚠")
        
        print(f"\n{status} {col} ({description}):")
        print(f"  Coverage: {non_null_pct:.1f}% ({non_null:,} records)")
        print(f"  Unique Categories: {unique:,}")
        
        if non_null > 0 and unique < 100:
            print(f"  Top 10 Categories:")
            top_cats = df[col].value_counts().head(10)
            for cat, count in top_cats.items():
                pct = (count / len(df) * 100)
                cat_str = str(cat)[:50]  # Truncate long names
                print(f"    • {cat_str}: {count:,} ({pct:.1f}%)")

print(f"\n🎯 MODULE 3 READINESS (Arıza Nedeni Kodu → Varlık Sınıfı):")
if 'cause code' in df.columns and best_col:
    cause_coverage = (df['cause code'].notna().sum() / len(df) * 100)
    class_coverage = (df[best_col].notna().sum() / len(df) * 100)
    
    if cause_coverage > 90 and class_coverage > 90:
        print(f"  ✅ READY: Can map fault codes to equipment classes")
        print(f"     • 'cause code' coverage: {cause_coverage:.1f}%")
        print(f"     • '{best_col}' coverage: {class_coverage:.1f}%")
        print(f"     • Strategy: Build hierarchical mapping")
    else:
        print(f"  ⚠ PARTIAL: Limited coverage")

# ============================================================================
# 9. GEOGRAPHIC COVERAGE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("SECTION 9: GEOGRAPHIC DATA COVERAGE")
print("="*100)

geo_cols = {
    'KOORDINAT_X': 'Longitude',
    'KOORDINAT_Y': 'Latitude',
    'İl': 'Province',
    'İlçe': 'District',
    'Mahalle': 'Neighborhood'
}

print(f"\n--- Geographic Data Availability ---")

coord_x_available = False
coord_y_available = False

for col, description in geo_cols.items():
    if col in df.columns:
        non_null = df[col].notna().sum()
        non_null_pct = (non_null / len(df) * 100)
        
        status = "✅" if non_null_pct > 90 else ("✓" if non_null_pct > 70 else "⚠")
        
        print(f"{status} {col} ({description}): {non_null_pct:.1f}% coverage")
        
        if col == 'KOORDINAT_X':
            coord_x_available = non_null_pct > 50
        if col == 'KOORDINAT_Y':
            coord_y_available = non_null_pct > 50

# Coordinate pair analysis
if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    coord_pairs = (df['KOORDINAT_X'].notna() & df['KOORDINAT_Y'].notna()).sum()
    coord_pair_pct = (coord_pairs / len(df) * 100)
    
    print(f"\n🗺️ COORDINATE PAIR COVERAGE: {coord_pair_pct:.1f}%")
    
    if coord_pair_pct > 95:
        print(f"  ✅ EXCELLENT: Can create detailed heat maps and spatial analysis")
    elif coord_pair_pct > 80:
        print(f"  ✓ GOOD: Sufficient for geographic risk mapping")
    elif coord_pair_pct > 60:
        print(f"  ⚠ FAIR: Consider supplementing with İlçe/Mahalle aggregation")
    else:
        print(f"  ❌ LIMITED: Use administrative boundaries (İlçe/Mahalle) instead")

# ============================================================================
# 10. CUSTOMER IMPACT ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("SECTION 10: CUSTOMER IMPACT DATA")
print("="*100)

customer_cols = ['total customer count', 'affected transformer count']

for col in customer_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        non_null_pct = (non_null / len(df) * 100)
        
        print(f"\n✓ {col}:")
        print(f"  Coverage: {non_null_pct:.1f}%")
        
        if non_null > 0:
            stats = df[col].describe()
            print(f"  Statistics:")
            print(f"    Mean:   {stats['mean']:.1f}")
            print(f"    Median: {stats['50%']:.1f}")
            print(f"    Min:    {stats['min']:.0f}")
            print(f"    Max:    {stats['max']:.0f}")
            
            # High-impact events
            high_threshold = stats['75%']
            high_impact = (df[col] > high_threshold).sum()
            high_impact_pct = (high_impact / len(df) * 100)
            print(f"  High-Impact Events (>75th percentile): {high_impact:,} ({high_impact_pct:.1f}%)")

print(f"\n🎯 MODULE: Kesintiden Etkilenen Müşteri")
if 'total customer count' in df.columns:
    customer_coverage = (df['total customer count'].notna().sum() / len(df) * 100)
    if customer_coverage > 90:
        print(f"  ✅ READY: Can prioritize by customer impact")
        print(f"     • Customer count coverage: {customer_coverage:.1f}%")
    else:
        print(f"  ⚠ PARTIAL: Limited customer data")

# ============================================================================
# 11. MISSING DATA SUMMARY
# ============================================================================
print("\n" + "="*100)
print("SECTION 11: MISSING DATA SUMMARY")
print("="*100)

missing_stats = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Pct': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Pct', ascending=False)

total_missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)

print(f"\n📊 Overall Missing Data: {total_missing_pct:.2f}%")
print(f"   Columns with missing data: {len(missing_stats)} out of {len(df.columns)}")

# Categorize by severity
critical_missing = missing_stats[missing_stats['Missing_Pct'] > 50]
high_missing = missing_stats[(missing_stats['Missing_Pct'] > 20) & (missing_stats['Missing_Pct'] <= 50)]
medium_missing = missing_stats[(missing_stats['Missing_Pct'] > 5) & (missing_stats['Missing_Pct'] <= 20)]
low_missing = missing_stats[missing_stats['Missing_Pct'] <= 5]

print(f"\n--- Missing Data Severity ---")
print(f"  ❌ CRITICAL (>50% missing): {len(critical_missing)} columns")
print(f"  ⚠  HIGH (20-50% missing):  {len(high_missing)} columns")
print(f"  ⚠  MEDIUM (5-20% missing): {len(medium_missing)} columns")
print(f"  ✓  LOW (<5% missing):      {len(low_missing)} columns")

if len(critical_missing) > 0:
    print(f"\n--- Top 10 Most Missing Columns ---")
    for _, row in critical_missing.head(10).iterrows():
        print(f"  • {row['Column']}: {row['Missing_Pct']:.1f}% missing")

# ============================================================================
# 12. DATA QUALITY SCORECARD
# ============================================================================
print("\n" + "="*100)
print("SECTION 12: DATA QUALITY SCORECARD")
print("="*100)

quality_checks = []

# 1. Equipment ID
if best_id_col:
    id_coverage = (df[best_id_col].notna().sum() / len(df) * 100)
    quality_checks.append(('Equipment ID', id_coverage > 90, id_coverage))
else:
    quality_checks.append(('Equipment ID', False, 0))

# 2. Equipment Class
if best_col:
    class_coverage = (df[best_col].notna().sum() / len(df) * 100)
    quality_checks.append(('Equipment Class', class_coverage > 80, class_coverage))
else:
    quality_checks.append(('Equipment Class', False, 0))

# 3. Installation Date
if 'TESIS_TARIHI' in df.columns or 'EDBS_IDATE' in df.columns:
    age_coverage = 0
    if 'TESIS_TARIHI' in df.columns:
        age_coverage = max(age_coverage, df['TESIS_TARIHI'].notna().sum() / len(df) * 100)
    if 'EDBS_IDATE' in df.columns:
        age_coverage = max(age_coverage, df['EDBS_IDATE'].notna().sum() / len(df) * 100)
    quality_checks.append(('Installation Date', age_coverage > 80, age_coverage))
else:
    quality_checks.append(('Installation Date', False, 0))

# 4. Fault Timestamp
if 'started at' in df.columns:
    fault_coverage = (df['started at'].notna().sum() / len(df) * 100)
    quality_checks.append(('Fault Timestamp', fault_coverage > 90, fault_coverage))
else:
    quality_checks.append(('Fault Timestamp', False, 0))

# 5. Fault Classification
if 'cause code' in df.columns:
    cause_coverage = (df['cause code'].notna().sum() / len(df) * 100)
    quality_checks.append(('Fault Classification', cause_coverage > 80, cause_coverage))
else:
    quality_checks.append(('Fault Classification', False, 0))

# 6. Geographic Coordinates
if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    geo_coverage = ((df['KOORDINAT_X'].notna() & df['KOORDINAT_Y'].notna()).sum() / len(df) * 100)
    quality_checks.append(('Geographic Coordinates', geo_coverage > 80, geo_coverage))
else:
    quality_checks.append(('Geographic Coordinates', False, 0))

# 7. Customer Impact
if 'total customer count' in df.columns:
    customer_coverage = (df['total customer count'].notna().sum() / len(df) * 100)
    quality_checks.append(('Customer Impact', customer_coverage > 80, customer_coverage))
else:
    quality_checks.append(('Customer Impact', False, 0))

# 8. Temporal Coverage
if 'started at' in df.columns and df['started at'].notna().sum() > 0:
    span_days = (df['started at'].max() - df['started at'].min()).days
    quality_checks.append(('Temporal Coverage', span_days > 365, span_days))
else:
    quality_checks.append(('Temporal Coverage', False, 0))

# 9. Data Completeness
overall_completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
quality_checks.append(('Overall Completeness', overall_completeness > 70, overall_completeness))

# 10. No Duplicates
dup_count = df.duplicated().sum()
dup_pct = (dup_count / len(df) * 100)
quality_checks.append(('No Duplicates', dup_pct < 5, 100 - dup_pct))

# Calculate score
quality_score = sum(1 for _, passed, _ in quality_checks if passed)

print(f"\n{'Quality Criterion':<30} {'Status':<10} {'Score':>10}")
print("-" * 52)

for criterion, passed, score in quality_checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    if criterion in ['Temporal Coverage', 'No Duplicates']:
        score_str = f"{score:.0f}"
    else:
        score_str = f"{score:.1f}%"
    print(f"{criterion:<30} {status:<10} {score_str:>10}")

print("-" * 52)
print(f"{'TOTAL SCORE':<30} {quality_score}/10")
print("-" * 52)

if quality_score >= 9:
    rating = "✅ EXCELLENT"
    message = "Data is ready for PoF modeling!"
elif quality_score >= 7:
    rating = "✓ GOOD"
    message = "Data is usable with minor preprocessing"
elif quality_score >= 5:
    rating = "⚠ FAIR"
    message = "Moderate preprocessing needed"
else:
    rating = "❌ POOR"
    message = "Significant data quality issues"

print(f"\n{rating}: {message}")

# ============================================================================
# 13. USE CASE READINESS ASSESSMENT
# ============================================================================
print("\n" + "="*100)
print("SECTION 13: USE CASE READINESS ASSESSMENT")
print("="*100)

use_cases = [
    {
        'name': 'Module 1: PoF Prediction (3/6/12 months)',
        'required': ['Equipment ID', 'Equipment Class', 'Installation Date', 'Fault Timestamp'],
        'checks': [
            best_id_col is not None,
            best_col is not None,
            'TESIS_TARIHI' in df.columns or 'EDBS_IDATE' in df.columns,
            'started at' in df.columns
        ]
    },
    {
        'name': 'Module 3: Arıza Nedeni Kodu → Varlık Sınıfı',
        'required': ['Fault Classification', 'Equipment Class'],
        'checks': [
            'cause code' in df.columns,
            best_col is not None
        ]
    },
    {
        'name': 'Module 1: Tekrarlayan Arıza (30/90 day patterns)',
        'required': ['Equipment ID', 'Fault Timestamp'],
        'checks': [
            best_id_col is not None,
            'started at' in df.columns
        ]
    },
    {
        'name': 'Module 1&2: Bakım Gecikmesi → Arıza Riski',
        'required': ['Equipment ID', 'Maintenance Records', 'Fault Timestamp'],
        'checks': [
            best_id_col is not None,
            'Tamamlanma Tarihi' in df.columns or 'Bakım Olanlar' in df.columns,
            'started at' in df.columns
        ]
    },
    {
        'name': 'Module 1: Kesintiden Etkilenen Müşteri',
        'required': ['Equipment ID', 'Customer Impact', 'Geographic Data'],
        'checks': [
            best_id_col is not None,
            'total customer count' in df.columns,
            'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns
        ]
    }
]

print("\n" + "=" * 90)
for i, uc in enumerate(use_cases, 1):
    readiness_pct = (sum(uc['checks']) / len(uc['checks']) * 100)
    
    if readiness_pct >= 75:
        status = "✅ READY"
    elif readiness_pct >= 50:
        status = "⚠ PARTIAL"
    else:
        status = "❌ NOT READY"
    
    print(f"\n{i}. {uc['name']}")
    print(f"   Readiness: {readiness_pct:.0f}% {status}")
    print(f"   Required: {', '.join(uc['required'])}")
    
    if readiness_pct < 100:
        missing = [req for req, check in zip(uc['required'], uc['checks']) if not check]
        if missing:
            print(f"   ⚠ Missing: {', '.join(missing)}")

# ============================================================================
# 14. CRITICAL NEXT STEPS
# ============================================================================
print("\n" + "="*100)
print("SECTION 14: CRITICAL NEXT STEPS & RECOMMENDATIONS")
print("="*100)

print("\n🚀 IMMEDIATE PRIORITY ACTIONS:\n")

print("=" * 80)
print("1. DATA TRANSFORMATION (CRITICAL - MUST DO FIRST)")
print("=" * 80)
print("\n   ⚠️  Your data is currently at FAULT-LEVEL")
print("      Current: 1 row = 1 fault event")
print("      Target:  1 row = 1 equipment with aggregated fault history")
print("\n   📋 Transformation Steps:")
print(f"      a. Choose primary equipment ID: {best_id_col if best_id_col else 'TBD'}")
print("      b. Group all faults by equipment ID")
print("      c. Aggregate fault timestamps to calculate:")
print("         • Arıza_Sayısı_3ay (fault count last 3 months)")
print("         • Arıza_Sayısı_6ay (fault count last 6 months)")
print("         • Arıza_Sayısı_12ay (fault count last 12 months)")
print("         • MTBF_Gün (mean time between failures)")
print("         • Son_Arıza_Gun_Sayisi (days since last fault)")
print("         • Tekrarlayan_Arıza flags (30/90 day recurring faults)")

print("\n" + "=" * 80)
print("2. EQUIPMENT AGE CALCULATION")
print("=" * 80)
print("\n   📅 Strategy:")
print("      Primary:  TESIS_TARIHI")
print("      Fallback: EDBS_IDATE")
print(f"      Formula:  Equipment_Age = {current_year} - Installation_Year")
if 'TESIS_TARIHI' in df.columns and 'EDBS_IDATE' in df.columns:
    combined = df['TESIS_TARIHI'].notna() | df['EDBS_IDATE'].notna()
    print(f"      Expected Coverage: ~{combined.sum()/len(df)*100:.0f}%")

print("\n" + "=" * 80)
print("3. EQUIPMENT CLASS STANDARDIZATION")
print("=" * 80)
if best_col:
    print(f"\n   🏷️  Primary Column: '{best_col}'")
    print("      Create unified 'Ekipman_Sınıfı_Standard' column")
    print("      Map to standard categories:")
    print("         • Trafo (OG/AG Trafo, Trafo Bina Tip)")
    print("         • Kesici")
    print("         • Ayırıcı")
    print("         • Box (AG Pano Box)")
    print("         • AG Anahtar")
    print("         • Others")

print("\n" + "=" * 80)
print("4. FAULT TIMESTAMP PROCESSING")
print("=" * 80)
print("\n   ⏰ Use 'started at' as primary fault timestamp")
print("      Extract temporal features:")
print("         • Year, Month, Day of Week")
print("         • Season (Winter/Spring/Summer/Fall)")
print("         • Peak Period Flags:")
print("           - Summer Peak: June-September")
print("           - Winter Peak: December-February")
print("         • Time-to-repair: ended at - started at")

print("\n" + "=" * 80)
print("5. GEOGRAPHIC DATA PROCESSING")
print("=" * 80)
if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    coord_cov = ((df['KOORDINAT_X'].notna() & df['KOORDINAT_Y'].notna()).sum() / len(df) * 100)
    print(f"\n   🗺️  Coordinate Coverage: {coord_cov:.1f}%")
    if coord_cov > 90:
        print("      ✅ Use coordinates for heat maps and spatial clustering")
    else:
        print("      ⚠ Supplement with İlçe/Mahalle for incomplete coordinates")
    print("\n      Create geographic clusters:")
    print("         • K-means clustering on coordinates")
    print("         • Or use administrative boundaries (İlçe/Mahalle)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("PROFILING COMPLETE - EXECUTIVE SUMMARY")
print("="*100)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   • Records: {df.shape[0]:,} fault events")
print(f"   • Features: {df.shape[1]} columns")
print(f"   • Temporal Span: ", end="")
if 'started at' in df.columns and df['started at'].notna().sum() > 0:
    print(f"{df['started at'].min().year} to {df['started at'].max().year}")
print(f"   • Equipment Classes: ", end="")
if best_col:
    print(f"{df[best_col].nunique()} unique types")
print(f"   • Quality Score: {quality_score}/10")

print(f"\n🎯 KEY COLUMNS IDENTIFIED:")
if best_id_col:
    print(f"   • Equipment ID: {best_id_col}")
if best_col:
    print(f"   • Equipment Class: {best_col}")
print(f"   • Installation Date: TESIS_TARIHI (primary), EDBS_IDATE (fallback)")
print(f"   • Fault Timestamp: started at")
print(f"   • Fault Classification: cause code, Kategori")

print(f"\n✅ USE CASE READINESS:")
all_ready = True
for uc in use_cases:
    readiness = sum(uc['checks']) / len(uc['checks']) * 100
    status = "✅" if readiness >= 75 else "⚠"
    print(f"   {status} {uc['name'].split(':')[1].strip()}: {readiness:.0f}%")
    if readiness < 75:
        all_ready = False

print(f"\n🚀 NEXT PHASE:")
print(f"   1. Transform fault-level → equipment-level data")
print(f"   2. Engineer temporal and failure history features")
print(f"   3. Build baseline PoF model (start with {df[best_col].value_counts().index[0] if best_col else 'most common equipment'})")
print(f"   4. Validate with time-based train/test split")
print(f"   5. Extend to all equipment classes")

print("\n" + "="*100)
print(f"{'END OF PROFILING REPORT':^100}")
print("="*100)
print("\n✓ Ready to proceed with data transformation!")
print(f"  Next script: data_transformation.py")