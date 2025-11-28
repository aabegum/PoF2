# Turkish Outputs Localization Plan
**Date**: 2025-11-27
**Project**: Turkish EDAŞ PoF Prediction
**Status**: ⚠️ ACTION REQUIRED

---

## Executive Summary

**CURRENT STATE**: All outputs are in English
**REQUIREMENT**: Outputs for Turkish distribution company should be in Turkish
**IMPACT**: Reports, predictions, visualizations need translation

**Action Required**: Comprehensive localization plan BEFORE finalizing outputs

---

## Output Files Requiring Turkish Localization

### Critical Outputs (Must Be Turkish)

#### 1. **Risk Assessment / CAPEX Priority List**
**File**: `predictions/capex_priority_list.csv` or similar

**Current** (English):
```
Equipment_ID | Risk_Score | Failure_Probability | Age_Years | Recommended_Action
3001 | 0.89 | 0.78 | 12.3 | Replace
3002 | 0.76 | 0.65 | 8.1 | Monitor
```

**Should Be** (Turkish):
```
Ekipman_ID | Risk_Puanı | Arıza_Olasılığı | Yaş_Yıl | Önerilen_İşlem
3001 | 0.89 | 0.78 | 12.3 | Değiştir
3002 | 0.76 | 0.65 | 8.1 | İzle
```

#### 2. **Chronic Repeater Predictions**
**File**: `predictions/chronic_repeaters.csv`

**Current** (English):
```
Equipment_ID | Chronic_Repeater_Probability | Classification | Failure_Count_90D
3001 | 0.92 | High Risk | 5
3002 | 0.34 | Low Risk | 1
```

**Should Be** (Turkish):
```
Ekipman_ID | Tekrarlayan_Arıza_Olasılığı | Sınıflandırma | 90Gün_Arıza_Sayısı
3001 | 0.92 | Yüksek Risk | 5
3002 | 0.34 | Düşük Risk | 1
```

#### 3. **Temporal PoF Predictions (3M/6M/12M)**
**File**: `predictions/temporal_pof_predictions.csv`

**Current** (English):
```
Equipment_ID | PoF_3M | PoF_6M | PoF_12M | Equipment_Class | Region
```

**Should Be** (Turkish):
```
Ekipman_ID | ArızaOlasılığı_3Ay | ArızaOlasılığı_6Ay | ArızaOlasılığı_12Ay | Ekipman_Sınıfı | Bölge
```

#### 4. **Survival Analysis Results**
**File**: `predictions/survival_probabilities.csv`

**Current** (English):
```
Equipment_ID | Equipment_Class | Survival_3M | Survival_6M | Survival_12M | Hazard_Rate
```

**Should Be** (Turkish):
```
Ekipman_ID | Ekipman_Sınıfı | Hayatta_Kalma_3Ay | Hayatta_Kalma_6Ay | Hayatta_Kalma_12Ay | Hazard_Oranı
```

### Important Outputs (Should Be Turkish)

#### 5. **Feature Importance Report**
**File**: `reports/feature_importance.csv`

**Current** (English):
```
Feature | Importance_Score | Impact | Category
Time_To_Repair_Hours_max | 0.23 | High | Equipment History
```

**Should Be** (Turkish):
```
Özellik | Önem_Puanı | Etki | Kategori
Onarım_Süresi_Maks_Saat | 0.23 | Yüksek | Ekipman Geçmişi
```

#### 6. **Data Quality Report**
**File**: `reports/data_quality_report.txt`

**Current** (English):
```
DATA QUALITY ASSESSMENT
========================
Total Equipment: 5,567
...
```

**Should Be** (Turkish):
```
VERİ KALİTESİ DEĞERLENDİRMESİ
================================
Toplam Ekipman: 5,567
...
```

#### 7. **Model Performance Report**
**File**: `reports/model_performance.txt`

**Current** (English):
```
MODEL PERFORMANCE SUMMARY
==========================
PoF Model AUC: 0.87
Chronic Classifier AUC: 0.82
...
```

**Should Be** (Turkish):
```
MODEL PERFORMANSI ÖZETİ
=========================
PoF Model AUC: 0.87
Kronik Sınıflandırıcı AUC: 0.82
...
```

### Secondary Outputs (Console Messages)

#### 8. **Pipeline Execution Logs**
**Files**: `logs/run_*/master_log.txt` and step logs

**Current**: English with Turkish feature names mixed
**Should Be**: Consistent Turkish console output

---

## Localization Strategy

### Approach 1: Dual Language (RECOMMENDED)

**Benefit**: English for technical, Turkish for business

```
reports/
├── capex_priority_list.csv              # Turkish (business users)
├── capex_priority_list_EN.csv           # English (technical team)
├── chronic_repeaters_TR.csv             # Turkish (distribution company)
├── chronic_repeaters_EN.csv             # English (tech)
└── model_performance_TR.txt             # Turkish (executive summary)
```

### Approach 2: Turkish Only

**Benefit**: Single set of outputs, simpler deployment

**Risk**: Technical team needs Turkish

### Approach 3: Configurable (BEST FOR PROJECT)

**Benefit**: Run once, generate both automatically

```python
# config.py
OUTPUT_LANGUAGE = 'TR'  # or 'EN'
GENERATE_BILINGUAL = True  # Generate both TR and EN
```

---

## Implementation Plan

### Phase 1: Translation Dictionary

**File to Create**: `localization/turkish_glossary.py`

```python
GLOSSARY_TR_EN = {
    # Equipment
    'Ekipman_ID': 'Equipment_ID',
    'Ekipman_Sınıfı': 'Equipment_Class',
    'Ekipman_Yaşı_Yıl': 'Equipment_Age_Years',

    # Failure/Risk
    'Arıza_Olasılığı': 'Failure_Probability',
    'Arıza_Riski': 'Failure_Risk',
    'Tekrarlayan_Arıza': 'Recurring_Failure',
    'Kronik_Arıza': 'Chronic_Failure',

    # Actions
    'Değiştir': 'Replace',
    'Onar': 'Repair',
    'İzle': 'Monitor',
    'Acil': 'Urgent',

    # Status
    'Yüksek Risk': 'High Risk',
    'Orta Risk': 'Medium Risk',
    'Düşük Risk': 'Low Risk',
}

# Column name mappings
COLUMN_NAMES_TR = {
    'equipment_id': 'Ekipman_ID',
    'pof_3m': 'ArızaOlasılığı_3Ay',
    'pof_6m': 'ArızaOlasılığı_6Ay',
    'pof_12m': 'ArızaOlasılığı_12Ay',
    'risk_score': 'Risk_Puanı',
    'recommended_action': 'Önerilen_İşlem',
}

# Report headers
REPORT_HEADERS_TR = {
    'EXECUTIVE_SUMMARY': 'YÖNETİCİ ÖZETİ',
    'RISK_ASSESSMENT': 'RİSK DEĞERLENDİRMESİ',
    'EQUIPMENT_RANKING': 'EKIPMAN SINIFLAMASI',
}
```

### Phase 2: Output Generation with Localization

**Files to Update**:
- `06_temporal_pof_model.py` - PoF predictions
- `07_chronic_classifier.py` - Chronic repeater predictions
- `09_calibration.py` - Probability calibration
- `10_survival_model.py` - Survival analysis
- `11_consequence_of_failure.py` - Risk assessment

**Pattern**:
```python
# Before: English only
predictions_df.to_csv('predictions/temporal_pof_predictions.csv')

# After: Turkish + English
predictions_df_tr = predictions_df.copy()
predictions_df_tr.columns = [COLUMN_NAMES_TR.get(col, col) for col in predictions_df.columns]
predictions_df_tr.to_csv('predictions/temporal_pof_predictions_TR.csv')  # Turkish
predictions_df.to_csv('predictions/temporal_pof_predictions_EN.csv')     # English
```

### Phase 3: Report Generation with Turkish Headers

**Files to Update**:
- `08_explainability.py` - Feature importance
- `pipeline_validation.py` - Validation reports
- New: `reports_generator.py` - Centralized report generation

**Pattern**:
```python
report_content = f"""
{REPORT_HEADERS_TR['RISK_ASSESSMENT']}
{'='*50}

Ekipman Sayısı: {total_equipment}
Yüksek Risk: {high_risk_count}
Orta Risk: {medium_risk_count}
Düşük Risk: {low_risk_count}
...
"""
```

### Phase 4: Console Messages

**Update All Scripts**:
- `00_input_data_source_analysis.py`
- `01_data_profiling.py`
- All pipeline scripts

**Pattern**:
```python
# Before
print(f"[STEP {step}/{total}] Data Transformation")

# After (with bilingual option)
if OUTPUT_LANGUAGE == 'TR':
    print(f"[ADIM {step}/{total}] Veri Dönüştürme")
else:
    print(f"[STEP {step}/{total}] Data Transformation")
```

---

## Turkish Feature Names (Complete Mapping)

### Equipment Characteristics
```
Equipment_ID → Ekipman_ID
Equipment_Class → Ekipman_Sınıfı
Equipment_Type → Ekipman_Tipi
Voltage_Class → Gerilim_Sınıfı
component_voltage → Bileşen_Gerilimi
```

### Age & Lifecycle
```
Equipment_Age_Years → Ekipman_Yaşı_Yıl
Expected_Life_Years → Beklenen_Yaşam_Yıl
Age_Life_Ratio → Yaş_Ömür_Oranı
Age_Risk_Category → Yaş_Risk_Kategorisi
```

### Failure History
```
Total_Faults → Toplam_Arıza_Sayısı
Days_Since_Last_Fault → Son_Arızadan_Gün
Time_To_Repair_Hours_mean → Onarım_Süresi_Ort_Saat
Failure_Count_90D → 90Gün_Arıza_Sayısı
```

### Risk & Probability
```
PoF_Probability → Arıza_Olasılığı
Risk_Score → Risk_Puanı
Hazard_Rate → Hazard_Oranı
Failure_Risk → Arıza_Riski
Chronic_Repeater_Probability → Tekrarlayan_Arıza_Olasılığı
```

### Actions
```
Recommended_Action → Önerilen_İşlem
Priority_Level → Öncelik_Seviyesi
Action_Date → İşlem_Tarihi
```

### Status Values
```
High Risk → Yüksek Risk
Medium Risk → Orta Risk
Low Risk → Düşük Risk
Urgent → Acil
Monitor → İzle
Repair → Onar
Replace → Değiştir
```

---

## Timeline & Priority

### CRITICAL (Must Complete Before Final Output)
- [ ] Create Turkish glossary (`localization/turkish_glossary.py`)
- [ ] Update main prediction exports (11 scripts)
- [ ] Update main reports (8 scripts)
- [ ] Test bilingual output

### IMPORTANT (Should Complete)
- [ ] Update console messages (5 scripts)
- [ ] Turkish visualizations (if any)
- [ ] Turkish documentation (3 guides)

### NICE-TO-HAVE (Phase 2)
- [ ] Email/communication templates
- [ ] API response formats
- [ ] Web dashboard translations

---

## Files to Create/Modify

### New Files:
1. `localization/turkish_glossary.py` - All translations
2. `localization/__init__.py` - Import module
3. `reports_generator_tr.py` - Turkish report generation
4. `TURKISH_OUTPUTS_GUIDE.md` - User guide in Turkish

### Modify:
1. `config.py` - Add OUTPUT_LANGUAGE setting
2. `06_temporal_pof_model.py` - Turkish outputs
3. `07_chronic_classifier.py` - Turkish outputs
4. `09_calibration.py` - Turkish outputs
5. `10_survival_model.py` - Turkish outputs
6. `11_consequence_of_failure.py` - Turkish outputs
7. `08_explainability.py` - Turkish reports
8. All pipeline scripts - Turkish console output

---

## Quick Implementation Checklist

- [ ] **Step 1**: Create glossary with all translations
- [ ] **Step 2**: Add OUTPUT_LANGUAGE to config
- [ ] **Step 3**: Create localization helper functions
- [ ] **Step 4**: Update prediction export functions
- [ ] **Step 5**: Update report generation
- [ ] **Step 6**: Update console output
- [ ] **Step 7**: Test both languages
- [ ] **Step 8**: Generate sample outputs in Turkish
- [ ] **Step 9**: Validate with Turkish stakeholders
- [ ] **Step 10**: Document for maintenance

---

## Example: Complete Turkish Output Flow

```
Input: Raw fault data (combined_data_son.xlsx)
  ↓
Step 0-5: Data processing (Turkish console messages)
  ↓
Step 6: PoF Model
  Output: temporal_pof_predictions_TR.csv (Turkish column names)
  ↓
Step 7: Chronic Classifier
  Output: chronic_repeaters_TR.csv (Turkish column names)
  ↓
Step 11: Risk Assessment
  Output: capex_priority_list_TR.csv (Turkish, executive ready)
  ↓
Step 12: Reporting
  Output: risk_assessment_report_TR.pdf (Turkish, stakeholder-ready)
  ↓
Final: Turkish company receives outputs in Turkish
  ✓ Column names in Turkish
  ✓ Report headers in Turkish
  ✓ Recommendations in Turkish
  ✓ Ready for stakeholder presentation
```

---

## Success Criteria

- ✅ All CSV outputs have Turkish column names
- ✅ All reports have Turkish headers
- ✅ All console messages in Turkish (if OUTPUT_LANGUAGE='TR')
- ✅ Bilingual option works (both TR and EN)
- ✅ Turkish stakeholders can read and act on outputs
- ✅ No lost information in translation
- ✅ Consistent terminology across all outputs
- ✅ Translation glossary documented for maintenance

---

**STATUS**: Implementation plan ready
**ACTION**: Approve and prioritize Turkish localization
**TIMELINE**: Can be completed alongside smart selection fix
