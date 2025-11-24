"""
COLUMN MAPPING & NAMING STANDARDIZATION
Turkish EDAŞ PoF Prediction Project

Purpose:
- Centralized column naming in Turkish
- EN→TR and TR→EN mapping dictionaries
- Validation and transformation functions
- Consistent naming across all pipeline scripts

Author: Data Analytics Team
Date: 2025
"""

# ============================================================================
# MASTER COLUMN MAPPING (English → Turkish)
# ============================================================================

# This is the single source of truth for column naming
# All pipeline scripts should import from here

COLUMN_MAP_EN_TO_TR = {
    # ========================================================================
    # TIER 1: Equipment Identification & Characteristics
    # ========================================================================
    'Ekipman_ID': 'Ekipman_ID',  # Keep as-is (already Turkish)
    'Equipment_Class_Primary': 'Ekipman_Sınıfı',
    'Equipment_Type': 'Ekipman_Tipi',
    'component_voltage': 'Gerilim_Seviyesi',
    'Voltage_Class': 'Gerilim_Sınıfı',
    'voltage_level': 'Gerilim_Seviyesi',  # Duplicate - maps to same
    'Is_MV': 'OG_Bayrağı',
    'Is_LV': 'AG_Bayrağı',
    'Is_HV': 'YG_Bayrağı',

    # ========================================================================
    # TIER 2: Age & Lifecycle
    # ========================================================================
    'Ekipman_Yaşı_Yıl': 'Ekipman_Yaşı_Yıl',  # Keep as-is
    'Yas_Beklenen_Omur_Orani': 'Yaş_Ömür_Oranı',
    'Beklenen_Ömür_Yıl': 'Beklenen_Ömür_Yıl',  # Keep as-is
    'Age_Risk_Category': 'Yaş_Risk_Kategorisi',

    # ========================================================================
    # TIER 3: Failure History - Counts
    # ========================================================================
    'Toplam_Arıza_Sayisi_Lifetime': 'Toplam_Arıza_Sayısı',
    'Arıza_Sayısı_3ay': 'Arıza_Sayısı_3Ay',
    'Arıza_Sayısı_6ay': 'Arıza_Sayısı_6Ay',
    'Arıza_Sayısı_12ay': 'Arıza_Sayısı_12Ay',

    # ========================================================================
    # TIER 3: Failure History - Temporal
    # ========================================================================
    'Son_Arıza_Gun_Sayisi': 'Son_Arıza_Gün_Sayısı',
    'Son_Arıza_Tarihi': 'Son_Arıza_Tarihi',  # Keep as-is
    'Ilk_Ariza_Tarihi': 'İlk_Arıza_Tarihi',
    'Ilk_Arizaya_Kadar_Yil': 'İlk_Arızaya_Kadar_Yıl',

    # ========================================================================
    # TIER 3: Failure History - Repair Times
    # ========================================================================
    'Time_To_Repair_Hours_mean': 'Onarım_Süresi_Ort_Saat',
    'Time_To_Repair_Hours_max': 'Onarım_Süresi_Maks_Saat',
    'Time_To_Repair_Hours_min': 'Onarım_Süresi_Min_Saat',
    'Time_To_Repair_Hours_std': 'Onarım_Süresi_Std_Saat',

    # ========================================================================
    # TIER 4: MTBF & Reliability (OAZS = Ortalama Arızalar Arası Süre)
    # ========================================================================
    'MTBF_Gün': 'OAZS_Gün',
    'MTBF_InterFault_Gün': 'OAZS_ArızaArası_Gün',
    'MTBF_Lifetime_Gün': 'OAZS_Ömür_Gün',
    'MTBF_Observable_Gün': 'OAZS_Gözlem_Gün',
    'MTBF_Degradation_Ratio': 'OAZS_Bozulma_Oranı',
    'MTBF_InterFault_Trend': 'Arıza_Sıklık_Trendi',  # Renamed: was misleading
    'MTBF_InterFault_StdDev': 'OAZS_Değişkenlik_Oranı',  # Renamed: not actual StdDev
    'Baseline_Hazard_Rate': 'Temel_Tehlike_Oranı',
    'Reliability_Score': 'Güvenilirlik_Skoru',

    # ========================================================================
    # TIER 5: Failure Cause Patterns
    # ========================================================================
    'Arıza_Nedeni_Çeşitlilik': 'Arıza_Nedeni_Çeşitlilik',  # Keep as-is
    'Arıza_Nedeni_Tutarlılık': 'Arıza_Nedeni_Tutarlılık',  # Keep as-is
    'Arıza_Nedeni_İlk': 'Arıza_Nedeni_İlk',  # Keep as-is
    'Arıza_Nedeni_Son': 'Arıza_Nedeni_Son',  # Keep as-is
    'Arıza_Nedeni_Sık': 'Arıza_Nedeni_Sık',  # Keep as-is
    'Neden_Değişim_Flag': 'Neden_Değişim_Bayrağı',
    'Tek_Neden_Flag': 'Tek_Neden_Bayrağı',
    'Çok_Nedenli_Flag': 'Çok_Nedenli_Bayrağı',
    'Ekipman_Neden_Kombinasyonu': 'Ekipman_Neden_Kombinasyonu',  # Keep as-is
    'Ekipman_Neden_Risk_Skoru': 'Ekipman_Neden_Risk_Skoru',  # Keep as-is

    # ========================================================================
    # TIER 6: Customer Impact
    # ========================================================================
    'Urban_Customer_Ratio_mean': 'Kentsel_Müşteri_Oranı',
    'Rural_Customer_Ratio_mean': 'Kırsal_Müşteri_Oranı',
    'MV_Customer_Ratio_mean': 'OG_Müşteri_Oranı',
    'urban_lv_Avg': 'Kentsel_AG_Müşteri_Ort',
    'urban_mv_Avg': 'Kentsel_OG_Müşteri_Ort',
    'suburban_lv_Avg': 'Yarıkentsel_AG_Müşteri_Ort',
    'suburban_mv_Avg': 'Yarıkentsel_OG_Müşteri_Ort',
    'rural_lv_Avg': 'Kırsal_AG_Müşteri_Ort',
    'rural_mv_Avg': 'Kırsal_OG_Müşteri_Ort',
    'total_customer_count_Avg': 'Toplam_Müşteri_Sayısı_Ort',
    'Avg_Customer_Count': 'Ortalama_Müşteri_Sayısı',
    'Customer_Minutes_Risk_Annual': 'Yıllık_Müşteri_Dakika_Riski',
    'Customer_Impact_Category': 'Müşteri_Etki_Kategorisi',

    # ========================================================================
    # TIER 7: Geographic & Environmental
    # ========================================================================
    'İlçe': 'İlçe',  # Keep as-is
    'Bölge_Tipi': 'Bölge_Tipi',  # Keep as-is
    'X_KOORDINAT': 'X_Koordinat',
    'Y_KOORDINAT': 'Y_Koordinat',
    'Summer_Peak_Flag_sum': 'Yaz_Pik_Toplam',
    'Winter_Peak_Flag_sum': 'Kış_Pik_Toplam',
    'Son_Arıza_Mevsim': 'Son_Arıza_Mevsimi',

    # ========================================================================
    # TIER 8: Derived & Interaction Features
    # ========================================================================
    'Overdue_Factor': 'Gecikme_Faktörü',
    'AgeRatio_Recurrence_Interaction': 'Yaş_Tekrar_Etkileşimi',
    'Time_Since_Last_Normalized': 'Son_Arıza_Normalize',
    'Failure_Free_3M': 'Arızasız_3Ay_Bayrağı',
    'Ekipman_Yoğunluk_Skoru': 'Ekipman_Yoğunluk_Skoru',  # Keep as-is

    # ========================================================================
    # TIER 9: Targets & Labels
    # ========================================================================
    'Tekrarlayan_Arıza_90gün_Flag': 'Kronik_Arıza_Bayrağı',
    'Target_3M': 'Hedef_3Ay',
    'Target_6M': 'Hedef_6Ay',
    'Target_12M': 'Hedef_12Ay',

    # ========================================================================
    # TIER 10: Model Outputs
    # ========================================================================
    'PoF_Probability': 'Arıza_Olasılığı',
    'Risk_Class': 'Risk_Sınıfı',
    'Risk_Score': 'Risk_Skoru',
    'Chronic_Probability': 'Kronik_Olasılık',
    'Chronic_Class': 'Kronik_Sınıf',
}

# Create reverse mapping (Turkish → English)
COLUMN_MAP_TR_TO_EN = {v: k for k, v in COLUMN_MAP_EN_TO_TR.items()}

# ============================================================================
# PROTECTED FEATURES (Turkish Names)
# ============================================================================

PROTECTED_FEATURES_TR = [
    # Essential ID
    'Ekipman_ID',

    # TIER 1: Equipment Characteristics
    'Ekipman_Sınıfı',
    'Gerilim_Seviyesi',
    'Gerilim_Sınıfı',

    # TIER 2: Age & Lifecycle
    'Ekipman_Yaşı_Yıl',
    'Yaş_Ömür_Oranı',
    'Beklenen_Ömür_Yıl',

    # TIER 3: Failure History - Temporal
    'Son_Arıza_Gün_Sayısı',
    'Onarım_Süresi_Ort_Saat',
    'Onarım_Süresi_Maks_Saat',

    # TIER 4: MTBF & Reliability
    'OAZS_Gün',
    'OAZS_Bozulma_Oranı',
    'Arıza_Sıklık_Trendi',

    # TIER 5: Failure Cause Patterns
    'Arıza_Nedeni_Çeşitlilik',
    'Arıza_Nedeni_Tutarlılık',
    'Neden_Değişim_Bayrağı',

    # TIER 6: Customer Impact
    'Kentsel_Müşteri_Oranı',
    'Kentsel_AG_Müşteri_Ort',
    'Kentsel_OG_Müşteri_Ort',
    'OG_Müşteri_Oranı',
    'Toplam_Müşteri_Sayısı_Ort',

    # TIER 7: Geographic
    'İlçe',
    'Bölge_Tipi',
    'Yaz_Pik_Toplam',

    # TIER 8: Interactions
    'Gecikme_Faktörü',
    'Yaş_Tekrar_Etkileşimi',

    # Target
    'Kronik_Arıza_Bayrağı',
]

# ============================================================================
# LEAKAGE PATTERNS (Auto-detection rules)
# ============================================================================

LEAKAGE_PATTERNS = {
    # Pattern: Column name contains these → potentially leaky
    'temporal_windows': [
        '_3Ay', '_6Ay', '_12Ay',  # Turkish
        '_3ay', '_6ay', '_12ay',  # Mixed case
        '_3M', '_6M', '_12M',     # English abbreviation
    ],

    # These column name patterns indicate target leakage
    'target_derived': [
        'Hedef_', 'Target_',
        'Arıza_Olasılığı', 'PoF_Probability',
        'Risk_Sınıfı', 'Risk_Class',
    ],

    # Aggregations that may include future data
    'aggregation_leakage': [
        '_Cluster_Avg', '_Class_Avg',
        '_Küme_Ort', '_Sınıf_Ort',
    ],

    # Safe patterns (whitelist)
    'safe_patterns': [
        'Toplam_Arıza_Sayısı',  # Lifetime count (before cutoff)
        'OAZS_',  # MTBF metrics (calculated from history)
        'Onarım_Süresi_',  # Repair times (historical)
        'Ekipman_Yaşı',  # Age (static)
    ],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def rename_columns_to_turkish(df, inplace=False):
    """
    Rename DataFrame columns from English to Turkish

    Args:
        df: pandas DataFrame
        inplace: If True, modify df in place

    Returns:
        DataFrame with Turkish column names
    """
    if not inplace:
        df = df.copy()

    # Build rename dict for columns that exist
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAP_EN_TO_TR:
            new_name = COLUMN_MAP_EN_TO_TR[col]
            if new_name != col:  # Only rename if different
                rename_dict[col] = new_name

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    return df


def rename_columns_to_english(df, inplace=False):
    """
    Rename DataFrame columns from Turkish to English

    Args:
        df: pandas DataFrame
        inplace: If True, modify df in place

    Returns:
        DataFrame with English column names
    """
    if not inplace:
        df = df.copy()

    # Build rename dict for columns that exist
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAP_TR_TO_EN:
            new_name = COLUMN_MAP_TR_TO_EN[col]
            if new_name != col:
                rename_dict[col] = new_name

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    return df


def get_turkish_name(english_name):
    """Get Turkish name for an English column name"""
    return COLUMN_MAP_EN_TO_TR.get(english_name, english_name)


def get_english_name(turkish_name):
    """Get English name for a Turkish column name"""
    return COLUMN_MAP_TR_TO_EN.get(turkish_name, turkish_name)


def validate_column_names(df, expected_language='tr'):
    """
    Validate that column names match expected language

    Args:
        df: pandas DataFrame
        expected_language: 'tr' for Turkish, 'en' for English

    Returns:
        dict with validation results
    """
    results = {
        'valid': [],
        'invalid': [],
        'unknown': [],
    }

    expected_names = set(COLUMN_MAP_EN_TO_TR.values()) if expected_language == 'tr' else set(COLUMN_MAP_EN_TO_TR.keys())

    for col in df.columns:
        if col in expected_names:
            results['valid'].append(col)
        elif col in COLUMN_MAP_EN_TO_TR or col in COLUMN_MAP_TR_TO_EN:
            results['invalid'].append(col)
        else:
            results['unknown'].append(col)

    return results


def is_protected_feature(column_name):
    """Check if a column is in the protected features list"""
    # Check both Turkish and English names
    if column_name in PROTECTED_FEATURES_TR:
        return True

    # Check if English equivalent is protected
    english_name = get_english_name(column_name)
    turkish_name = get_turkish_name(column_name)

    return turkish_name in PROTECTED_FEATURES_TR


def detect_leakage_pattern(column_name):
    """
    Detect if a column name matches leakage patterns

    Returns:
        tuple: (is_leaky, pattern_type, is_safe)
    """
    col_lower = column_name.lower()

    # Check safe patterns first (whitelist)
    for pattern in LEAKAGE_PATTERNS['safe_patterns']:
        if pattern.lower() in col_lower:
            return (False, None, True)

    # Check temporal window patterns
    for pattern in LEAKAGE_PATTERNS['temporal_windows']:
        if pattern.lower() in col_lower:
            return (True, 'temporal_window', False)

    # Check target-derived patterns
    for pattern in LEAKAGE_PATTERNS['target_derived']:
        if pattern.lower() in col_lower:
            return (True, 'target_derived', False)

    # Check aggregation patterns
    for pattern in LEAKAGE_PATTERNS['aggregation_leakage']:
        if pattern.lower() in col_lower:
            return (True, 'aggregation', False)

    return (False, None, False)


# ============================================================================
# FEATURE CATEGORIES (for reporting)
# ============================================================================

FEATURE_CATEGORIES = {
    'Ekipman_Özellikleri': [
        'Ekipman_ID', 'Ekipman_Sınıfı', 'Ekipman_Tipi',
        'Gerilim_Seviyesi', 'Gerilim_Sınıfı',
    ],
    'Yaş_Ömür': [
        'Ekipman_Yaşı_Yıl', 'Yaş_Ömür_Oranı', 'Beklenen_Ömür_Yıl',
        'Yaş_Risk_Kategorisi',
    ],
    'Arıza_Geçmişi': [
        'Toplam_Arıza_Sayısı', 'Son_Arıza_Gün_Sayısı',
        'Onarım_Süresi_Ort_Saat', 'Onarım_Süresi_Maks_Saat',
    ],
    'Güvenilirlik_OAZS': [
        'OAZS_Gün', 'OAZS_Bozulma_Oranı', 'Arıza_Sıklık_Trendi',
        'OAZS_Değişkenlik_Oranı',
    ],
    'Arıza_Nedenleri': [
        'Arıza_Nedeni_Çeşitlilik', 'Arıza_Nedeni_Tutarlılık',
        'Neden_Değişim_Bayrağı',
    ],
    'Müşteri_Etkisi': [
        'Kentsel_Müşteri_Oranı', 'Kırsal_Müşteri_Oranı',
        'Toplam_Müşteri_Sayısı_Ort',
    ],
    'Coğrafi': [
        'İlçe', 'Bölge_Tipi', 'Yaz_Pik_Toplam',
    ],
    'Türetilmiş': [
        'Gecikme_Faktörü', 'Yaş_Tekrar_Etkileşimi',
    ],
}


def categorize_feature(column_name):
    """Get the category for a feature"""
    for category, features in FEATURE_CATEGORIES.items():
        if column_name in features:
            return category
    return 'Diğer'


# ============================================================================
# PRINT SUMMARY
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("COLUMN MAPPING MODULE")
    print("="*80)
    print(f"\nTotal mappings: {len(COLUMN_MAP_EN_TO_TR)}")
    print(f"Protected features: {len(PROTECTED_FEATURES_TR)}")
    print(f"Feature categories: {len(FEATURE_CATEGORIES)}")

    print("\n--- Protected Features (Turkish) ---")
    for feat in PROTECTED_FEATURES_TR:
        print(f"  • {feat}")

    print("\n--- Leakage Detection Patterns ---")
    for pattern_type, patterns in LEAKAGE_PATTERNS.items():
        print(f"  {pattern_type}: {len(patterns)} patterns")
