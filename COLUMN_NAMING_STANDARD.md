# Column Naming Standard
## Turkish EDAŞ PoF Prediction Pipeline

### Purpose
This document defines the column naming convention for the PoF prediction pipeline.

---

## Naming Convention

### 1. **Domain/Business Features (Turkish)**
Features that represent business concepts familiar to Turkish EDAŞ operators:

**Pattern**: `Turkish_Word_With_Underscores`

**Examples:**
- `Arıza_Sayısı_12ay` (12-month fault count)
- `Ekipman_Yaşı_Yıl` (equipment age in years)
- `Kentsel_Müşteri_Oranı` (urban customer ratio)
- `Bölge_Tipi` (region type: urban/rural)
- `Yas_Beklenen_Omur_Orani` (age to expected life ratio)

**Rationale:**
- End-user reports and dashboards use these names
- Domain experts recognize business terminology
- Maintains consistency with Turkish EDAŞ documentation

---

### 2. **Technical/ML Features (English)**
Features created for modeling that don't have established Turkish business names:

**Pattern**: `English_Word_With_Underscores`

**Examples:**
- `Composite_PoF_Risk_Score` (calculated risk metric)
- `Failure_Rate_Per_Year` (normalized metric)
- `Geographic_Cluster` (unsupervised learning feature)
- `Age_Failure_Interaction` (engineered interaction term)
- `MTBF_Risk_Score` (derived risk component)

**Rationale:**
- Standard ML terminology (easier for data scientists)
- Documentation references international standards (IEEE, IEC)
- Model interpretation uses established ML vocabulary

---

### 3. **Raw Data Columns (Original Format)**
Preserve original column names from source systems:

**Pattern**: Keep as-is from source

**Examples:**
- `cbs_id` (EdaBİS system ID - lowercase)
- `Ekipman ID` (Equipment ID - space preserved)
- `started at` (fault start timestamp - lowercase with space)
- `cause code` (fault cause - lowercase with space)
- `TESIS_TARIHI` (installation date - uppercase Turkish)
- `KOORDINAT_X` (X coordinate - uppercase Turkish)

**Rationale:**
- Traceability to source systems
- Easier debugging (names match source databases)
- Avoids transformation errors from renaming

---

### 4. **Hybrid Columns (English Structure, Turkish Content)**
When English structure is needed but content is Turkish:

**Pattern**: `English_Structure_Primary` → contains Turkish values

**Examples:**
- `Equipment_Class_Primary` → values: "AG Hat", "OG Hat", "Ayırıcı"
- `Voltage_Class` → values: "AG", "OG", "YG"
- `Son_Arıza_Mevsim` → values: "Yaz", "Kış", "İlkbahar", "Sonbahar"

**Rationale:**
- Column name is standardized for code
- Values preserve Turkish business terminology
- Best of both worlds for international teams

---

## Naming Rules

### ✅ DO:
1. Use **snake_case** (underscores) for all columns
2. Use **descriptive names** (avoid abbreviations unless standard)
3. Include **units** in name when applicable:
   - `Ekipman_Yaşı_Yıl` (years)
   - `Ekipman_Yaşı_Gün` (days)
   - `Time_To_Repair_Hours` (hours)
   - `MTBF_Gün` (days)

4. Use **consistent suffixes**:
   - `_Flag` for binary indicators (0/1)
   - `_Score` for calculated metrics (0-100)
   - `_Ratio` or `_Oranı` for proportions (0-1)
   - `_Count` or `_Sayısı` for counts (integers)
   - `_Avg` or `_mean` for averages
   - `_max` / `_min` for aggregations

### ❌ DON'T:
1. Mix languages in single column name (except `_Primary` suffix)
2. Use spaces in new features (preserve only in raw data)
3. Use camelCase or PascalCase
4. Use overly abbreviated names (`Arıza_Sy` ❌ → `Arıza_Sayısı` ✓)

---

## Feature Categories

### Source Data (Raw)
Preserve original names from EdaBİS, TESIS, fault logs

### Domain Features (Turkish)
Business KPIs and operational metrics

### Engineered Features (English)
ML-specific transformations and interactions

### Risk Metrics (English)
Model outputs and predictions

---

## Migration Guide

If you need to rename columns:

```python
# Group 1: Business features (Turkish)
turkish_features = {
    'fault_count_12m': 'Arıza_Sayısı_12ay',
    'equipment_age_years': 'Ekipman_Yaşı_Yıl',
    'customer_ratio_urban': 'Kentsel_Müşteri_Oranı',
}

# Group 2: Technical features (English)
english_features = {
    'pof_score': 'Composite_PoF_Risk_Score',
    'cluster_id': 'Geographic_Cluster',
}

# Apply renaming
df.rename(columns={**turkish_features, **english_features}, inplace=True)
```

---

## Validation

Check naming consistency:
```python
import re

def validate_column_name(col):
    """Check if column follows naming convention"""
    # Allow: Turkish/English words with underscores
    # Allow: Numbers in names
    # Allow: Special chars in raw data columns (preserved)

    if re.match(r'^[A-Za-zğüşıöçĞÜŞİÖÇ0-9_\s]+$', col):
        return True
    return False

# Check all columns
invalid_cols = [col for col in df.columns if not validate_column_name(col)]
if invalid_cols:
    print(f"Invalid column names: {invalid_cols}")
```

---

## Example Feature List

### Turkish (Business Domain)
- `Arıza_Sayısı_3ay`, `Arıza_Sayısı_6ay`, `Arıza_Sayısı_12ay`
- `Ekipman_Yaşı_Yıl`, `Ekipman_Yaşı_Gün`
- `Beklenen_Ömür_Yıl`, `Yas_Beklenen_Omur_Orani`
- `Kentsel_Müşteri_Oranı`, `Kırsal_Müşteri_Oranı`
- `Bölge_Tipi` (Kentsel/Kırsal)
- `Son_Arıza_Mevsim` (Yaz/Kış/İlkbahar/Sonbahar)

### English (ML/Technical)
- `Composite_PoF_Risk_Score`, `Risk_Category`
- `Age_Risk_Score`, `Recent_Failure_Risk_Score`
- `Failure_Rate_Per_Year`, `Recent_Failure_Intensity`
- `Geographic_Cluster`, `Reliability_Score`
- `Age_Failure_Interaction`, `Customer_Failure_Interaction`

### Raw Data (Preserved)
- `cbs_id`, `Ekipman ID`
- `started at`, `resolved at`
- `cause code`, `district`
- `TESIS_TARIHI`, `KOORDINAT_X`, `KOORDINAT_Y`

---

## Future Considerations

### Internationalization (i18n)
If expanding to other regions:

1. Create translation dictionaries:
```python
FEATURE_LABELS = {
    'Arıza_Sayısı_12ay': {
        'tr': 'Arıza Sayısı (12 ay)',
        'en': '12-Month Fault Count',
        'display': 'Fault Count (12M)'
    }
}
```

2. Generate reports in multiple languages
3. Keep column names in code (English or Turkish)
4. Translate only for display/UI

---

**Document Version:** 1.0
**Last Updated:** 2025-01-14
**Author:** Data Analytics Team
