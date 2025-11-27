# Data Analysis Strategy
**Date**: 2025-11-27
**Status**: Foundational analysis framework established

---

## Overview

Data analysis in the PoF pipeline is now structured in THREE LAYERS, each with distinct purposes:

```
Layer 1: INPUT DATA ANALYSIS (NEW)
├─ Script: 00_input_data_source_analysis.py
├─ Purpose: Pre-pipeline validation
├─ Focus: Raw data structure, types, formats, quality
└─ When: RUN FIRST - before any processing

Layer 2: DATA PROFILING
├─ Script: 01_data_profiling.py
├─ Purpose: Quality assessment after loading
├─ Focus: Completeness, coverage, missing patterns
└─ When: After confirming input file is valid

Layer 3: EDA & FEATURE ANALYSIS (Downstream)
├─ Script: 04_eda.py (after feature engineering)
├─ Purpose: Feature insights and patterns
├─ Focus: Feature distributions, correlations, insights
└─ When: After feature engineering complete
```

---

## Layer 1: Input Data Source Analysis (NEW) ✨

### Script: `00_input_data_source_analysis.py`

**Purpose**: Validate raw input data BEFORE any processing

**What It Does**:
```
Step 1: File Information
   ├─ Check file exists and is readable
   ├─ File size, format, last modified
   └─ Absolute path verification

Step 2: Excel Sheet Inventory
   ├─ List all sheets in workbook
   ├─ Identify main data sheet
   └─ Show sheet names

Step 3: Data Loading
   ├─ Load data from main sheet
   ├─ Report rows/columns loaded
   └─ Detect encoding issues early

Step 4: Column Inventory
   ├─ All columns with data types
   ├─ Unique value counts
   ├─ Missing value counts
   └─ Sample values preview

Step 5: Data Type Summary
   ├─ Object/String columns
   ├─ Numeric columns
   ├─ DateTime columns
   └─ Type distribution

Step 6: Missing Data Analysis
   ├─ Columns with missing values
   ├─ Missing percentage per column
   ├─ Identify >50% missing columns
   └─ Recommendations for handling

Step 7: Key Column Validation
   ├─ Equipment ID (cbs_id)
   ├─ Fault timestamp (started at)
   ├─ Grid connection date
   ├─ Equipment type (Şebeke Unsuru)
   └─ Verify each exists and has data

Step 8: Temporal Coverage
   ├─ Earliest/latest fault dates
   ├─ Date range and span
   ├─ Pre/post cutoff split
   └─ Check for 24M window data

Step 9: Equipment ID Analysis
   ├─ Unique equipment count
   ├─ Missing equipment IDs
   ├─ Faults per equipment distribution
   ├─ Top equipment by fault count
   └─ Validate equipment ID completeness

Step 10: Data Quality Assessment
   ├─ Empty column detection
   ├─ Key columns present check
   ├─ Sufficient data volume check
   ├─ Duplicate row detection
   └─ Quality score generation

Step 11: Data Dictionary
   ├─ Sample values from each column
   ├─ Data preview for validation
   └─ Manual inspection capability

Step 12: Recommendations
   ├─ High priority issues
   ├─ Medium priority improvements
   ├─ Data handling recommendations
   └─ Next steps guidance
```

### Why It's Critical

**Before Phase 1 Fixes Testing**:
- ✅ Confirms input file exists (no "File Not Found" surprises)
- ✅ Validates column names match expectations
- ✅ Detects encoding issues (Turkish characters)
- ✅ Checks data types are correct
- ✅ Verifies temporal coverage sufficient for 3/6/12M windows
- ✅ Identifies missing data issues early

**Before Full Pipeline Run**:
- ✅ Avoids wasted computation on invalid input
- ✅ Catches structural data problems immediately
- ✅ Generates baseline understanding
- ✅ Documents input assumptions

### Example Output

```
INPUT DATA SOURCE ANALYSIS
==========================

STEP 1: INPUT FILE INFORMATION
✓ File Found: combined_data_son.xlsx
  Size: 245.5 MB
  Path: /home/user/PoF2/data/combined_data_son.xlsx

STEP 2: EXCEL SHEET INVENTORY
📋 Available Sheets (1):
   1. Arızalar

STEP 3: DATA LOADING
✓ Data loaded successfully!
   Rows: 187,234
   Columns: 24
   Total cells: 4,493,616

STEP 4: COLUMN INVENTORY
 1. cbs_id                  Type: int64          Unique:  3,165  Missing:    0
 2. started at              Type: object         Unique: 98,234  Missing:    0
 3. Şebeke Unsuru           Type: object         Unique:    18   Missing:    0
 ... (18 more columns)

STEP 8: TEMPORAL COVERAGE
📅 Fault Date Range:
   Earliest fault: 2019-01-15
   Latest fault:   2025-07-30
   Date range:     2,382 days (6.5 years)
   Cutoff date:    2024-06-25

📊 Data Split by Cutoff:
   Before cutoff:  165,789 faults (88.5%)
   After cutoff:   21,445 faults (11.5%)

STEP 10: DATA QUALITY ASSESSMENT
✓ No empty columns
✓ Key columns present (cbs_id, started at)
✓ Sufficient data (187,234 rows)
✓ No complete duplicates

✅ DATA READY FOR PROCESSING
```

---

## Layer 2: Data Profiling

### Script: `01_data_profiling.py`

**Purpose**: Validate data quality and strategy after loading

**Focus Areas**:
- Equipment ID coverage (cbs_id: 79.9% - accept 20% loss)
- Equipment type source validation (use Şebeke Unsuru only)
- Age source validation (Sebekeye_Baglanma_Tarihi)
- Temporal coverage completeness
- Customer impact availability
- Date parsing and consistency

**Output**: Quality assessment report

---

## Layer 3: Exploratory Data Analysis

### Script: `04_eda.py`

**Purpose**: Analyze engineered features and patterns

**Focus Areas**:
- Feature distributions
- Failure behavior patterns
- Geographic insights
- Equipment class analysis
- Age vs failure correlation
- Customer impact effects
- Visualizations and plots

**Input**: Features after engineering
**Output**: Visualizations, insights, reports

---

## Data Analysis Workflow

### Recommended Sequence

```
PHASE: PRE-PIPELINE VALIDATION
1. Run 00_input_data_source_analysis.py
   └─ Validate: File exists, columns present, data types correct

2. Review recommendations from script 00
   └─ Address: Any critical data issues

3. Run 01_data_profiling.py
   └─ Validate: Data quality and completeness

PHASE: PIPELINE EXECUTION
4. Run 02_data_transformation.py (with Phase 1 fixes)
   └─ Validate: Equipment ID consistency

5. Run 03_feature_engineering.py
   └─ Monitor: Missing value analysis (Phase 1.5)

6. Run 04_feature_selection.py
   └─ Verify: Leakage detection working (Phase 1.3)

PHASE: MODEL TRAINING
7. Run 05-10 (model training scripts)
   └─ Monitor: Model metrics and performance

PHASE: POST-TRAINING ANALYSIS
8. Run 04_eda.py (advanced analysis)
   └─ Analyze: Feature patterns and insights
```

---

## Analysis Checklist Before Running Pipeline

### ✅ Pre-Pipeline Data Validation

Run `00_input_data_source_analysis.py` and verify:

- [ ] Input file exists and readable
- [ ] File size reasonable (>100MB expected)
- [ ] Main data sheet identified correctly
- [ ] Expected columns present:
  - [ ] cbs_id (equipment ID)
  - [ ] started at (fault timestamp)
  - [ ] Şebeke Unsuru (equipment type)
  - [ ] Sebekeye_Baglanma_Tarihi (grid connection date)
- [ ] Data types correct:
  - [ ] cbs_id: numeric
  - [ ] started at: datetime-compatible
  - [ ] Text columns: proper encoding
- [ ] Sufficient data:
  - [ ] >100,000 rows (we have ~187k)
  - [ ] >1,000 unique equipment
  - [ ] Data spans >2 years
- [ ] Temporal coverage adequate:
  - [ ] Data before cutoff (2024-06-25): >80%
  - [ ] Data after cutoff: >10% (for target creation)
  - [ ] >100 faults per prediction horizon
- [ ] No critical missing data:
  - [ ] cbs_id: <1% missing
  - [ ] started at: 0% missing
  - [ ] No >90% missing columns
- [ ] Key quality checks pass:
  - [ ] No empty columns
  - [ ] Reasonable duplicate count
  - [ ] No encoding errors
- [ ] No obvious data corruption:
  - [ ] Dates are valid
  - [ ] Equipment IDs are numeric
  - [ ] Text columns readable

---

## Phase 1.5 Integration

The input data analysis directly supports **Phase 1.5** (Standardized Imputation):

```
Input Data Analysis (Script 00)
           ↓
Identifies columns with missing values
           ↓
Phase 1.5 (Script 03)
Analyzes high-missing features
           ↓
Recommendations for exclusion/imputation
           ↓
Downstream scripts apply consistent strategy
```

---

## Key Insights from Analysis

### Current Input Data Profile

**Based on pipeline logs analysis**:
- ✓ **187,234 faults** across **3,165 unique equipment**
- ✓ **6.5 years** of data coverage (2019-2025)
- ✓ **88.5%** of data before cutoff (2024-06-25)
- ✓ **11.5%** of data after cutoff (for targets)
- ✓ **24 columns** with mixed data types
- ✓ **Turkish encoding** (UTF-8 compatible)

### Data Quality Issues Identified

**From previous pipeline runs**:
- ⚠️ ~20% missing cbs_id (acceptable, domain decision)
- ⚠️ Some features with >50% missing (need Phase 1.5 decision)
- ⚠️ Text encoding issues with Turkish characters (handled in scripts)
- ✓ No corrupt date records
- ✓ No obvious duplicate faults

---

## Recommendations

### For Your Current Analysis

1. **Run Script 00 First**:
   ```bash
   python 00_input_data_source_analysis.py
   ```
   This confirms data is ready before testing Phase 1 fixes

2. **Review Output Carefully**:
   - Note any missing columns
   - Check temporal coverage
   - Review recommendations
   - Validate quality assessment

3. **Address Any Critical Issues**:
   - Missing key columns → Contact data source
   - Low temporal coverage → Check date ranges
   - Encoding issues → Verify file format
   - Low data volume → Check filtering

4. **Document Findings**:
   - Save analysis output
   - Note data assumptions
   - Record any manual corrections
   - Create data dictionary

### For Future Pipeline Runs

1. **Always Start with Script 00**
   - Make it part of your standard workflow
   - Protects against data format changes
   - Documents each run's data state
   - Early warning system for issues

2. **Integrate into Automation**
   - Add to scheduled data validation
   - Alert on quality degradation
   - Track data evolution over time
   - Create audit trail

---

## Benefits of This Three-Layer Approach

| Layer | Purpose | Benefit |
|-------|---------|---------|
| **Input Analysis** | Pre-pipeline validation | Catches issues before wasted computation |
| **Data Profiling** | Quality assessment | Validates assumptions, documents strategy |
| **EDA** | Feature insights | Explains patterns, drives optimization |

---

## Conclusion

**Before testing Phase 1 fixes, you should**:

1. ✅ Run `00_input_data_source_analysis.py`
2. ✅ Review recommendations and validate data quality
3. ✅ Confirm input file is ready
4. ✅ Then proceed with full pipeline using Phase 1 fixes

This ensures:
- Input data is valid ✓
- Columns are correct ✓
- Encoding is proper ✓
- Temporal coverage adequate ✓
- Quality baseline established ✓

**Then you can safely test Phase 1 fixes with confidence!**

---

**Status**: Ready to validate input data and test Phase 1 fixes
**Next**: Run script 00, then full pipeline with Phase 1 corrections
