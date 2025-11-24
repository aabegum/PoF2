"""
================================================================================
SCRIPT 01: DATA PROFILING v5.0
================================================================================
Turkish EDAS PoF (Probability of Failure) Prediction Pipeline

PIPELINE STRATEGY: Multi-Horizon PoF Prediction (3M, 6M, 12M)
- Validates data quality for temporal PoF modeling
- Confirms timestamp coverage with flexible date parser
- Reports equipment ID strategy, age calculation sources, customer impact coverage

WHAT THIS SCRIPT DOES:
Comprehensive data quality profiling of fault-level data before transformation:
- Equipment identification strategy (cbs_id only, no fallback)
- Temporal coverage validation (fault timestamps, installation dates)
- Data completeness scoring (10/10 quality score expected)
- Customer impact coverage analysis

ENHANCEMENTS in v5.0:
+ Updated for Sebekeye_Baglanma_Tarihi (Grid Connection Date) as age source
+ ID Strategy: cbs_id only (no fallback, no UNKNOWN generation)
+ Equipment Type: Şebeke Unsuru only (no fallback)
+ Dynamic input file from config (combined_data_son.xlsx)
+ Progress Indicators: [Step X/Y] for pipeline visibility
+ Flexible Date Parser: Supports mixed date formats

CROSS-REFERENCES:
- Script 02: Uses validated data for equipment-level transformation
- Script 03: Creates advanced features from validated equipment data
- config.py: Defines INPUT_FILE and validation thresholds

Input:  config.INPUT_FILE (fault records)
Output: Data quality report + validation for Script 02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix Unicode encoding for Windows console (Turkish cp1254 issue)
if sys.platform == 'win32':
    try:
        # Set console to UTF-8 mode for Unicode symbols
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        # Reconfigure stdout with UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        # If encoding setup fails, continue anyway
        pass

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 60)

print("\n" + "="*80)
print("SCRIPT 01: DATA PROFILING v5.0 (Multi-Horizon PoF)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[Step 1/7] Loading Fault-Level Data...")

# Import INPUT_FILE from config
from config import INPUT_FILE
data_path = INPUT_FILE

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print(f"\nPlease ensure {data_path.name} is in the '{data_path.parent}' directory")
    exit(1)

print(f"\nLoading: {data_path}")
print(f"File size: {data_path.stat().st_size / 1024**2:.2f} MB")

try:
    df = pd.read_excel(data_path)
    print(f"✓ Successfully loaded!\n")
except Exception as e:
    print(f"❌ Error loading: {e}")
    exit(1)

print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n⚠️  Data Structure: FAULT-LEVEL (each row = one fault event)")
print(f"   Target: EQUIPMENT-LEVEL (each row = one equipment)")

# Initialize report
report_lines = []
report_lines.append("="*80)
report_lines.append("DATA QUALITY REPORT - TURKISH EDAŞ POF PROJECT")
report_lines.append("="*80)
report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Total Records: {df.shape[0]:,}")
report_lines.append(f"Total Columns: {df.shape[1]}")

# ============================================================================
# 2. EQUIPMENT IDENTIFICATION STRATEGY
# ============================================================================
print("\n[Step 2/7] Equipment Identification Strategy...")

id_columns_priority = ['cbs_id', 'id', 'Ekipman ID', 'HEPSI_ID', 'Ekipman Kodu']
available_id_cols = [col for col in id_columns_priority if col in df.columns]

print(f"\nEquipment ID Priority Order:")
for i, col in enumerate(id_columns_priority, 1):
    if col in df.columns:
        coverage = df[col].notna().sum()
        pct = coverage / len(df) * 100
        unique = df[col].nunique()
        print(f"  {i}. {col:20s} → {pct:5.1f}% coverage ({coverage:,} records, {unique:,} unique)")
    else:
        print(f"  {i}. {col:20s} → NOT FOUND")

# Determine best ID column
best_id_col = None
for col in id_columns_priority:
    if col in df.columns and df[col].notna().sum() / len(df) > 0.5:
        best_id_col = col
        break

if best_id_col:
    print(f"\n🎯 Selected Equipment ID: '{best_id_col}'")
    report_lines.append(f"\nPrimary Equipment ID: {best_id_col}")
    report_lines.append(f"Coverage: {df[best_id_col].notna().sum() / len(df) * 100:.1f}%")
else:
    print(f"\n❌ WARNING: No reliable equipment ID column found!")

# ============================================================================
# 3. EQUIPMENT CLASSIFICATION STRATEGY
# ============================================================================
print("\n" + "="*100)
print("STEP 3: EQUIPMENT CLASSIFICATION STRATEGY")
print("="*100)

class_columns_priority = ['Equipment_Type', 'Ekipman Sınıfı', 'Kesinti Ekipman Sınıfı', 'Ekipman Sınıf']
available_class_cols = [col for col in class_columns_priority if col in df.columns]

print(f"\nEquipment Class Priority Order:")
for i, col in enumerate(class_columns_priority, 1):
    if col in df.columns:
        coverage = df[col].notna().sum()
        pct = coverage / len(df) * 100
        unique = df[col].nunique()
        print(f"  {i}. {col:25s} → {pct:5.1f}% coverage ({coverage:,} records, {unique:,} types)")
    else:
        print(f"  {i}. {col:25s} → NOT FOUND")

# Determine best classification column
best_class_col = None
for col in class_columns_priority:
    if col in df.columns and df[col].notna().sum() / len(df) > 0.5:
        best_class_col = col
        break

if best_class_col:
    print(f"\n🎯 Selected Equipment Class: '{best_class_col}'")

    # Show top equipment types
    print(f"\nTOP EQUIPMENT TYPES:")
    value_counts = df[best_class_col].value_counts()
    total = len(df)

    print(f"\n{'Equipment Type':<25} {'Count':>10} {'Percentage':>12}")
    print("-" * 50)

    for val, count in value_counts.head(15).items():
        pct = count / total * 100
        print(f"{str(val)[:24]:<25} {count:>10,} {pct:>11.1f}%")

    if len(value_counts) > 15:
        others = value_counts.iloc[15:].sum()
        print(f"{'Others':<25} {others:>10,} {others/total*100:>11.1f}%")

    # ML Readiness: Class Imbalance Assessment
    print(f"\n{'='*100}")
    print(f"ML READINESS: CLASS IMBALANCE ASSESSMENT")
    print(f"{'='*100}")

    top_2_pct = (value_counts.head(2).sum() / total * 100)
    print(f"\n  Top 2 classes: {top_2_pct:.1f}% of data")

    if top_2_pct > 80:
        print(f"  🚨 SEVERE IMBALANCE: Model will be heavily biased toward dominant classes")
        print(f"     → Recommendation: Use stratified sampling + oversampling (SMOTE/ADASYN)")
        print(f"     → Expected Impact: Predictions for minority classes will be unreliable")
    elif top_2_pct > 60:
        print(f"  ⚠️  MODERATE IMBALANCE: Model may underperform on minority classes")
        print(f"     → Recommendation: Use stratified sampling + class weights")
    else:
        print(f"  ✓ BALANCED: Good class distribution for modeling")

    # Check for rare classes (<10 samples)
    rare_classes = value_counts[value_counts < 10]
    if len(rare_classes) > 0:
        print(f"\n  Rare Equipment Classes (<10 samples): {len(rare_classes)}")
        for cls, count in rare_classes.items():
            print(f"    • {str(cls)[:40]}: {count} samples")
        print(f"\n  🚨 CRITICAL: Predictions for rare classes = LOW CONFIDENCE")
        print(f"     → Recommendation: Flag these in final predictions with confidence warnings")
        print(f"     → Consider: Collapse into 'Other' category or exclude from training")

    # Identify dominant classes (>20% each)
    dominant = value_counts[value_counts / total > 0.20]
    if len(dominant) > 0:
        print(f"\n  Dominant Classes (>20% each): {len(dominant)}")
        for cls, count in dominant.items():
            pct = count / total * 100
            print(f"    • {str(cls)[:40]}: {count:,} ({pct:.1f}%)")
        print(f"  → Model will be optimized for these classes")

    report_lines.append(f"\nPrimary Equipment Class: {best_class_col}")
    report_lines.append(f"Unique Equipment Types: {len(value_counts)}")
    report_lines.append(f"Class Imbalance: Top 2 = {top_2_pct:.1f}%, Rare classes (<10) = {len(rare_classes)}")
    report_lines.append(f"\nEQUIPMENT TYPE DISTRIBUTION (Consolidated):")
    report_lines.append(str(best_class_col))
    for val, count in value_counts.head(15).items():
        report_lines.append(f"{str(val):<20} {count:>6,}")

# ============================================================================
# 3b. EQUIPMENT SPECIFICATIONS (OPTIONAL - Future Enhancements)
# ============================================================================
print("\n" + "="*100)
print("STEP 3b: EQUIPMENT SPECIFICATIONS (Optional)")
print("="*100)

spec_columns = {
    'voltage_level': 'Voltage Level (LV/MV/HV)',
    'kVa_rating': 'Transformer Capacity Rating (kVA)',
    'component voltage': 'Component Voltage',
    'MARKA': 'Equipment Brand',
    'MARKA_MODEL': 'Equipment Model'
}

print(f"\nEquipment Specification Columns:")
found_specs = False

for col, description in spec_columns.items():
    if col in df.columns:
        found_specs = True
        coverage = df[col].notna().sum()
        pct = coverage / len(df) * 100
        unique = df[col].nunique()

        status = "✅" if pct > 90 else ("✓" if pct > 70 else "⚠")
        print(f"  {status} {col:20s} → {pct:5.1f}% coverage ({unique:,} unique values)")

        # Show value distribution for key columns
        if col in ['voltage_level', 'component voltage'] and coverage > 0 and unique < 20:
            print(f"     Distribution:")
            val_counts = df[col].value_counts().head(5)
            for val, count in val_counts.items():
                val_pct = count / len(df) * 100
                print(f"       • {str(val)[:30]:30s}: {count:>6,} ({val_pct:>5.1f}%)")

        if col == 'kVa_rating' and coverage > 0:
            valid_ratings = df[col].dropna()
            if pd.api.types.is_numeric_dtype(valid_ratings):
                print(f"     Statistics:")
                print(f"       Mean:   {valid_ratings.mean():>8.1f} kVA")
                print(f"       Median: {valid_ratings.median():>8.1f} kVA")
                print(f"       Range:  {valid_ratings.min():>8.1f} - {valid_ratings.max():>8.1f} kVA")
    else:
        print(f"  ❌ {col:20s} → NOT FOUND (may be added in future)")

if not found_specs:
    print("\n  ⚠️  No equipment specification columns found")
    print("     These columns are optional but helpful for advanced analysis:")
    print("       • voltage_level: For voltage-based segmentation")
    print("       • kVa_rating: For transformer capacity analysis")
    print("       • component voltage: Alternative voltage information")
else:
    # Add to report
    report_lines.append(f"\nEQUIPMENT SPECIFICATIONS:")
    for col, description in spec_columns.items():
        if col in df.columns:
            coverage = df[col].notna().sum()
            pct = coverage / len(df) * 100
            report_lines.append(f"  {col}: {pct:.1f}% coverage")

print("\n💡 Note: voltage_level and kVa_rating can be added to enhance:")
print("   • Equipment segmentation by voltage class")
print("   • Transformer-specific failure analysis")
print("   • Capacity-based risk modeling")

# ============================================================================
# 4. EQUIPMENT AGE ANALYSIS (WITH FLEXIBLE DATE PARSER)
# ============================================================================
print("\n" + "="*100)
print("STEP 4: EQUIPMENT AGE ANALYSIS")
print("="*100)

current_year = datetime.now().year
print(f"\nCurrent Year: {current_year}")

# Flexible date parser (handles mixed formats)
def parse_date_flexible(value):
    """
    Parse date with multiple format support - handles mixed format data
    Supports: ISO, Turkish (DD-MM-YYYY), European (DD/MM/YYYY), Excel serial dates
    """
    # Already a timestamp/datetime
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)

    # Handle NaN/None
    if pd.isna(value):
        return pd.NaT

    # Excel serial date (numeric)
    if isinstance(value, (int, float)):
        if 1 <= value <= 100000:
            try:
                return pd.Timestamp('1899-12-30') + pd.Timedelta(days=value)
            except:
                return pd.NaT
        else:
            return pd.NaT

    # String parsing with multiple format attempts
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return pd.NaT

        # Try multiple formats in order of likelihood
        formats = [
            '%Y-%m-%d %H:%M:%S',     # 2021-01-15 12:30:45 (ISO)
            '%d-%m-%Y %H:%M:%S',     # 15-01-2021 12:30:45 (Turkish/European with dash)
            '%d/%m/%Y %H:%M:%S',     # 15/01/2021 12:30:45 (Turkish/European with slash)
            '%Y-%m-%d',              # 2021-01-15
            '%d-%m-%Y',              # 15-01-2021
            '%d/%m/%Y',              # 15/01/2021
            '%d.%m.%Y %H:%M:%S',     # 15.01.2021 12:30:45 (Turkish dot format)
            '%d.%m.%Y',              # 15.01.2021
            '%m/%d/%Y %H:%M:%S',     # 01/15/2021 12:30:45 (US format - try last)
            '%m/%d/%Y',              # 01/15/2021
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except:
                continue

        # Last resort: let pandas infer
        try:
            return pd.to_datetime(value, infer_datetime_format=True, dayfirst=True)
        except:
            return pd.NaT

    return pd.NaT

# Convert date columns with flexible parser
for col in ['Sebekeye_Baglanma_Tarihi', 'started at', 'ended at']:
    if col in df.columns and df[col].dtype != 'datetime64[ns]':
        df[col] = df[col].apply(parse_date_flexible)

# Calculate age from Sebekeye_Baglanma_Tarihi (Grid Connection Date)
age_source_col = None

if 'Sebekeye_Baglanma_Tarihi' in df.columns:
    df['_age_from_grid_connection'] = current_year - df['Sebekeye_Baglanma_Tarihi'].dt.year
    age_source_col = '_age_from_grid_connection'

    # Create combined age column with source tracking
    df['_equipment_age'] = np.nan
    df['_age_source'] = 'MISSING'

    mask = df['Sebekeye_Baglanma_Tarihi'].notna()
    df.loc[mask, '_equipment_age'] = df.loc[mask, '_age_from_grid_connection']
    df.loc[mask, '_age_source'] = 'GRID_CONNECTION'

    # Age source distribution
    print(f"\nAGE SOURCE DISTRIBUTION:")
    report_lines.append(f"\nAGE SOURCE DISTRIBUTION:")

    age_source_counts = df['_age_source'].value_counts()
    for source in ['GRID_CONNECTION', 'MISSING']:
        count = age_source_counts.get(source, 0)
        pct = count / len(df) * 100
        avg_age = df[df['_age_source'] == source]['_equipment_age'].mean()
        avg_age_str = f"{avg_age:.1f}" if not np.isnan(avg_age) else "nan"

        print(f"  {source:20s}: {count:>6,} ({pct:>5.1f}%) - Avg age: {avg_age_str:>5s} yrs")
        report_lines.append(f"  {source:20s}: {count:>6,} ({pct:>5.1f}%) - Avg age: {avg_age_str:>5s} yrs")
else:
    print(f"\n⚠️  WARNING: 'Sebekeye_Baglanma_Tarihi' column not found!")
    print(f"   Equipment age calculation will not be possible")
    df['_equipment_age'] = np.nan
    df['_age_source'] = 'MISSING'

# Equipment age statistics
valid_ages = df['_equipment_age'].dropna()

if len(valid_ages) > 0:
        print(f"\nEQUIPMENT AGE STATISTICS:")
        print(f"  Mean:   {valid_ages.mean():>6.1f} years")
        print(f"  Median: {valid_ages.median():>6.1f} years")
        print(f"  Min:    {valid_ages.min():>6.1f} years")
        print(f"  Max:    {valid_ages.max():>6.1f} years")

        # Age over 50 warning
        old_equipment = (valid_ages > 50).sum()
        if old_equipment > 0:
            print(f"  Age > 50 years: {old_equipment}")

        report_lines.append(f"\nEQUIPMENT AGE STATISTICS:")
        report_lines.append(f"  Mean: {valid_ages.mean():.1f} years")
        report_lines.append(f"  Median: {valid_ages.median():.1f} years")
        report_lines.append(f"  Max: {valid_ages.max():.1f} years")
        if old_equipment > 0:
            report_lines.append(f"  Age > 50 years: {old_equipment}")

        # Age distribution
        print(f"\nAGE DISTRIBUTION:")
        report_lines.append(f"\nAGE DISTRIBUTION:")

        age_bins = [0, 5, 10, 20, 30, 50, 1000]
        age_labels = ['0-5 years (New)', '5-10 years', '10-20 years', '20-30 years', '30-50 years', '50+ years ⚠️']

        df['_age_category'] = pd.cut(df['_equipment_age'], bins=age_bins, labels=age_labels, include_lowest=True)
        age_dist = df['_age_category'].value_counts().sort_index()

        for category in age_labels:
            count = age_dist.get(category, 0)
            pct = count / len(df) * 100
            print(f"  {category:20s} {count:>6,} ({pct:>5.1f}%)")
            report_lines.append(f"  {category:20s} {count:>6,} ({pct:>5.1f}%)")

# ============================================================================
# 5. FAULT TIMESTAMP ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 5: FAULT TIMESTAMP ANALYSIS")
print("="*100)

fault_timestamp_cols = ['started at', 'ended at']
available_fault_cols = [col for col in fault_timestamp_cols if col in df.columns]

print(f"\nFault Timestamp Columns:")
for col in fault_timestamp_cols:
    if col in df.columns:
        coverage = df[col].notna().sum()
        pct = coverage / len(df) * 100
        print(f"  {col:15s} → {pct:5.1f}% coverage ({coverage:,} records)")
    else:
        print(f"  {col:15s} → NOT FOUND")

if 'started at' in df.columns:
    valid_timestamps = df['started at'].dropna()

    if len(valid_timestamps) > 0:
        min_date = valid_timestamps.min()
        max_date = valid_timestamps.max()
        span_days = (max_date - min_date).days

        print(f"\nTemporal Coverage:")
        print(f"  First Fault: {min_date.strftime('%Y-%m-%d')}")
        print(f"  Last Fault:  {max_date.strftime('%Y-%m-%d')}")
        print(f"  Span:        {span_days:,} days ({span_days/365:.1f} years)")

        report_lines.append(f"\nFAULT TEMPORAL COVERAGE:")
        report_lines.append(f"  First: {min_date.strftime('%Y-%m-%d')}")
        report_lines.append(f"  Last:  {max_date.strftime('%Y-%m-%d')}")
        report_lines.append(f"  Span:  {span_days/365:.1f} years")

        # Year distribution
        df['_fault_year'] = df['started at'].dt.year
        year_counts = df['_fault_year'].value_counts().sort_index()

        print(f"\nFault Distribution by Year:")
        for year, count in year_counts.items():
            if not pd.isna(year):
                pct = count / len(df) * 100
                print(f"  {int(year)}: {count:>6,} ({pct:>5.1f}%)")

# ============================================================================
# 6. CUSTOMER IMPACT ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 6: CUSTOMER IMPACT ANALYSIS")
print("="*100)

customer_impact_cols = [
    'urban mv+suburban mv',
    'urban lv+suburban lv',
    'urban mv',
    'urban lv',
    'suburban mv',
    'suburban lv',
    'rural mv',
    'rural lv',
    'total customer count'
]

available_customer_cols = [col for col in customer_impact_cols if col in df.columns]

print(f"\nCustomer Impact Columns:")
for col in customer_impact_cols:
    if col in df.columns:
        coverage = df[col].notna().sum()
        pct = coverage / len(df) * 100

        if coverage > 0:
            mean_val = df[col].mean()
            max_val = df[col].max()
            print(f"  {col:25s} → {pct:5.1f}% coverage (Mean: {mean_val:>7.1f}, Max: {max_val:>7.0f})")
        else:
            print(f"  {col:25s} → {pct:5.1f}% coverage")
    else:
        print(f"  {col:25s} → NOT FOUND")

if 'total customer count' in df.columns:
    valid_customers = df['total customer count'].dropna()

    if len(valid_customers) > 0:
        print(f"\nTotal Customer Impact Statistics:")
        print(f"  Mean:   {valid_customers.mean():>10.1f}")
        print(f"  Median: {valid_customers.median():>10.1f}")
        print(f"  Max:    {valid_customers.max():>10.0f}")

        # High impact events
        high_impact = (valid_customers > valid_customers.quantile(0.75)).sum()
        print(f"  High Impact (>75th percentile): {high_impact:,}")

        report_lines.append(f"\nCUSTOMER IMPACT:")
        report_lines.append(f"  Mean customers affected: {valid_customers.mean():.1f}")
        report_lines.append(f"  High-impact events: {high_impact:,}")

# ============================================================================
# 7. GEOGRAPHIC COVERAGE
# ============================================================================
print("\n" + "="*100)
print("STEP 7: GEOGRAPHIC COVERAGE")
print("="*100)

geo_cols = ['KOORDINAT_X', 'KOORDINAT_Y', 'İl', 'İlçe', 'Mahalle']

print(f"\nGeographic Data Availability:")
for col in geo_cols:
    if col in df.columns:
        coverage = df[col].notna().sum()
        pct = coverage / len(df) * 100
        status = "✅" if pct > 90 else ("✓" if pct > 70 else "⚠")
        print(f"  {status} {col:15s} → {pct:5.1f}% coverage")
    else:
        print(f"  ❌ {col:15s} → NOT FOUND")

if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    coord_pairs = (df['KOORDINAT_X'].notna() & df['KOORDINAT_Y'].notna()).sum()
    coord_pct = coord_pairs / len(df) * 100

    print(f"\nCoordinate Pairs: {coord_pct:.1f}% ({coord_pairs:,} records)")

    if coord_pct > 95:
        print(f"  ✅ EXCELLENT: Can create detailed heat maps")
    elif coord_pct > 80:
        print(f"  ✓ GOOD: Sufficient for geographic analysis")
    else:
        print(f"  ⚠ LIMITED: Consider using İlçe/Mahalle instead")

# ============================================================================
# 8. DATA QUALITY SCORECARD
# ============================================================================
print("\n" + "="*100)
print("STEP 8: DATA QUALITY SCORECARD")
print("="*100)

quality_checks = []

# 1. Equipment ID
if best_id_col:
    id_coverage = df[best_id_col].notna().sum() / len(df) * 100
    quality_checks.append(('Equipment ID', id_coverage > 90, id_coverage))
else:
    quality_checks.append(('Equipment ID', False, 0))

# 2. Equipment Class
if best_class_col:
    class_coverage = df[best_class_col].notna().sum() / len(df) * 100
    quality_checks.append(('Equipment Class', class_coverage > 80, class_coverage))
else:
    quality_checks.append(('Equipment Class', False, 0))

# 3. Installation Date (Grid Connection Date)
if 'Sebekeye_Baglanma_Tarihi' in df.columns:
    age_coverage = df['Sebekeye_Baglanma_Tarihi'].notna().sum() / len(df) * 100
    quality_checks.append(('Installation Date', age_coverage > 80, age_coverage))
else:
    quality_checks.append(('Installation Date', False, 0))

# 4. Fault Timestamp
if 'started at' in df.columns:
    fault_coverage = df['started at'].notna().sum() / len(df) * 100
    quality_checks.append(('Fault Timestamp', fault_coverage > 90, fault_coverage))
else:
    quality_checks.append(('Fault Timestamp', False, 0))

# 5. Customer Impact
if 'total customer count' in df.columns:
    customer_coverage = df['total customer count'].notna().sum() / len(df) * 100
    quality_checks.append(('Customer Impact', customer_coverage > 80, customer_coverage))
else:
    quality_checks.append(('Customer Impact', False, 0))

# 6. Geographic Data
if 'KOORDINAT_X' in df.columns and 'KOORDINAT_Y' in df.columns:
    geo_coverage = (df['KOORDINAT_X'].notna() & df['KOORDINAT_Y'].notna()).sum() / len(df) * 100
    quality_checks.append(('Geographic Data', geo_coverage > 80, geo_coverage))
else:
    quality_checks.append(('Geographic Data', False, 0))

# 7. Temporal Coverage
if 'started at' in df.columns and df['started at'].notna().sum() > 0:
    span_days = (df['started at'].max() - df['started at'].min()).days
    quality_checks.append(('Temporal Span', span_days > 365, span_days))
else:
    quality_checks.append(('Temporal Span', False, 0))

# 8. Data Completeness
overall_completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
quality_checks.append(('Overall Completeness', overall_completeness > 70, overall_completeness))

# 9. No Duplicates
dup_count = df.duplicated().sum()
dup_pct = dup_count / len(df) * 100
quality_checks.append(('No Duplicates', dup_pct < 5, 100 - dup_pct))

# 10. Equipment Variety
if best_class_col:
    equipment_variety = df[best_class_col].nunique()
    quality_checks.append(('Equipment Variety', equipment_variety > 5, equipment_variety))
else:
    quality_checks.append(('Equipment Variety', False, 0))

# Calculate score
quality_score = sum(1 for _, passed, _ in quality_checks if passed)

print(f"\n{'Quality Criterion':<25} {'Status':<10} {'Score':>15}")
print("-" * 52)

for criterion, passed, score in quality_checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    if criterion in ['Temporal Span', 'Equipment Variety']:
        score_str = f"{score:.0f}"
    else:
        score_str = f"{score:.1f}%"
    print(f"{criterion:<25} {status:<10} {score_str:>15}")

print("-" * 52)
print(f"{'TOTAL QUALITY SCORE':<25} {quality_score}/10")
print("-" * 52)

# Optional/Bonus Checks (Future Enhancements)
optional_checks = []

# Voltage Level (optional)
if 'voltage_level' in df.columns:
    voltage_coverage = df['voltage_level'].notna().sum() / len(df) * 100
    optional_checks.append(('Voltage Level', voltage_coverage > 80, voltage_coverage))
elif 'component voltage' in df.columns:
    voltage_coverage = df['component voltage'].notna().sum() / len(df) * 100
    optional_checks.append(('Component Voltage', voltage_coverage > 80, voltage_coverage))

# kVa Rating (optional)
if 'kVa_rating' in df.columns:
    kva_coverage = df['kVa_rating'].notna().sum() / len(df) * 100
    optional_checks.append(('kVa Rating', kva_coverage > 80, kva_coverage))

# Equipment Brand/Model (optional)
if 'MARKA' in df.columns or 'MARKA_MODEL' in df.columns:
    brand_coverage = 0
    if 'MARKA' in df.columns:
        brand_coverage = max(brand_coverage, df['MARKA'].notna().sum() / len(df) * 100)
    if 'MARKA_MODEL' in df.columns:
        brand_coverage = max(brand_coverage, df['MARKA_MODEL'].notna().sum() / len(df) * 100)
    optional_checks.append(('Equipment Brand/Model', brand_coverage > 70, brand_coverage))

if len(optional_checks) > 0:
    optional_score = sum(1 for _, passed, _ in optional_checks if passed)

    print(f"\n{'OPTIONAL CHECKS (Bonus)':<25} {'Status':<10} {'Score':>15}")
    print("-" * 52)

    for criterion, passed, score in optional_checks:
        status = "✅ PASS" if passed else "⚠️  N/A"
        score_str = f"{score:.1f}%"
        print(f"{criterion:<25} {status:<10} {score_str:>15}")

    print("-" * 52)
    print(f"{'OPTIONAL SCORE':<25} {optional_score}/{len(optional_checks)}")
    print("-" * 52)
    print(f"\n💡 Optional checks provide enhanced analysis capabilities")
    print(f"   These are not required for core PoF modeling")
else:
    print(f"\n💡 OPTIONAL CHECKS: None found (voltage_level, kVa_rating can be added later)")
    print(f"   These enhance analysis but are not required for core PoF modeling")

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

report_lines.append(f"\nDATA QUALITY SCORE: {quality_score}/10")
report_lines.append(f"Rating: {rating}")

# Add optional checks to report
if len(optional_checks) > 0:
    report_lines.append(f"\nOPTIONAL SPECIFICATIONS:")
    for criterion, passed, score in optional_checks:
        status = "✅" if passed else "⚠️"
        report_lines.append(f"  {status} {criterion}: {score:.1f}% coverage")

# ============================================================================
# 9. MISSING DATA ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STEP 9: MISSING DATA SUMMARY")
print("="*100)

missing_stats = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Pct': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Pct', ascending=False)

total_missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)

print(f"\nOverall Missing Data: {total_missing_pct:.2f}%")
print(f"Columns with missing data: {len(missing_stats)}/{len(df.columns)}")

# Categorize by severity
critical_missing = missing_stats[missing_stats['Missing_Pct'] > 50]
high_missing = missing_stats[(missing_stats['Missing_Pct'] > 20) & (missing_stats['Missing_Pct'] <= 50)]
medium_missing = missing_stats[(missing_stats['Missing_Pct'] > 5) & (missing_stats['Missing_Pct'] <= 20)]

print(f"\nMissing Data Severity:")
print(f"  ❌ CRITICAL (>50%): {len(critical_missing)} columns")
print(f"  ⚠  HIGH (20-50%):  {len(high_missing)} columns")
print(f"  ⚠  MEDIUM (5-20%): {len(medium_missing)} columns")

if len(critical_missing) > 0:
    print(f"\nTop 10 Most Missing Columns:")
    for _, row in critical_missing.head(10).iterrows():
        print(f"  • {row['Column'][:50]:50s} {row['Missing_Pct']:>6.1f}%")

# ============================================================================
# 10. NEXT STEPS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*100)
print("STEP 10: CRITICAL NEXT STEPS")
print("="*100)

print("\n🚀 TRANSFORMATION REQUIRED:")
print("\n  1. DATA TRANSFORMATION (Fault-level → Equipment-level)")
print(f"     • Primary ID: {best_id_col if best_id_col else 'TBD'}")
print(f"     • Primary Class: {best_class_col if best_class_col else 'TBD'}")
print("     • Group by equipment ID and aggregate:")
print("       - Arıza_Sayısı_3ay/6ay/12ay (failure counts)")
print("       - MTBF_Gün (mean time between failures)")
print("       - Son_Arıza_Gun_Sayisi (days since last fault)")
print("       - Tekrarlayan_Arıza flags (recurring patterns)")

print("\n  2. EQUIPMENT AGE CALCULATION")
print("     • Source: Sebekeye_Baglanma_Tarihi (Grid Connection Date)")
print(f"     • Formula: Equipment_Age = {current_year} - Grid_Connection_Year")

print("\n  3. TEMPORAL FEATURE ENGINEERING")
print("     • Extract from 'started at':")
print("       - Season flags (Summer/Winter peaks)")
print("       - Year, Month, Day of Week")
print("       - Time-to-repair (ended at - started at)")

print("\n  4. GEOGRAPHIC CLUSTERING")
print("     • Use KOORDINAT_X, KOORDINAT_Y for clustering")
print("     • Fallback to İlçe/Mahalle if coordinates missing")

# ============================================================================
# 11. SAVE QUALITY REPORT
# ============================================================================
print("\n" + "="*100)
print("STEP 11: SAVING QUALITY REPORT")
print("="*100)

# Create reports directory
Path('reports').mkdir(exist_ok=True)

# Save text report
report_path = 'reports/data_quality_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n✓ Quality report saved: {report_path}")

# Save detailed missing data CSV
if len(missing_stats) > 0:
    missing_path = 'reports/missing_data_analysis.csv'
    missing_stats.to_csv(missing_path, index=False, encoding='utf-8-sig')
    print(f"✓ Missing data analysis saved: {missing_path}")

# Save column inventory
col_inventory = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes.astype(str),
    'Non_Null_Count': df.count(),
    'Null_Count': df.isnull().sum(),
    'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
col_inventory_path = 'reports/column_inventory.csv'
col_inventory.to_csv(col_inventory_path, index=False, encoding='utf-8-sig')
print(f"✓ Column inventory saved: {col_inventory_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("PROFILING COMPLETE - EXECUTIVE SUMMARY")
print("="*100)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   • Records: {df.shape[0]:,} fault events")
print(f"   • Features: {df.shape[1]} columns")
print(f"   • Quality Score: {quality_score}/10 {rating}")

print(f"\n🎯 KEY COLUMNS IDENTIFIED:")
if best_id_col:
    print(f"   • Equipment ID: {best_id_col} (cbs_id only, no fallback)")
if best_class_col:
    print(f"   • Equipment Class: {best_class_col}")
    print(f"   • Equipment Types: {df[best_class_col].nunique()} unique")
print(f"   • Installation Date: Sebekeye_Baglanma_Tarihi (Grid Connection)")
print(f"   • Fault Timestamp: started at, ended at")

if 'started at' in df.columns and df['started at'].notna().sum() > 0:
    print(f"   • Temporal Span: {df['started at'].min().year} to {df['started at'].max().year}")

# Optional columns status
optional_found = []
if 'voltage_level' in df.columns:
    optional_found.append('voltage_level')
elif 'component voltage' in df.columns:
    optional_found.append('component voltage')
if 'kVa_rating' in df.columns:
    optional_found.append('kVa_rating')
if 'MARKA' in df.columns or 'MARKA_MODEL' in df.columns:
    optional_found.append('Equipment Brand/Model')

if len(optional_found) > 0:
    print(f"\n🌟 OPTIONAL SPECIFICATIONS FOUND:")
    for col in optional_found:
        print(f"   • {col}")
else:
    print(f"\n💡 OPTIONAL SPECIFICATIONS (Can be added later):")
    print(f"   • voltage_level - For voltage-based segmentation")
    print(f"   • kVa_rating - For transformer capacity analysis")

print(f"\n📂 REPORTS GENERATED:")
print(f"   • {report_path}")
print(f"   • {missing_path if len(missing_stats) > 0 else 'reports/missing_data_analysis.csv'}")
print(f"   • {col_inventory_path}")

print(f"\n🚀 NEXT PHASE:")
print(f"   → Run: 02_data_transformation.py")
print(f"   → Transform to equipment-level data")
print(f"   → Engineer failure history features")

print("\n" + "="*100)
print(f"{'END OF PROFILING REPORT':^100}")
print("="*100)
print("\n✓ Ready to proceed with data transformation!")
