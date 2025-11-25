# Script Naming Analysis
**Date**: 2025-11-25
**Issue**: Inconsistent and confusing script numbering

---

## Problem Summary

Script numbers **do NOT reflect execution order**, causing confusion:

- **Missing numbers**: 04, 09
- **Duplicate prefixes**: Three scripts start with `06_`, two with `10_`
- **Non-sequential execution**: Script 10 runs before 06, then 06 runs 3x, then 10 again

---

## Current State (CONFUSING)

### Actual Execution Order vs Filename Numbers:

| Execution Step | Current Filename | Number Issue |
|----------------|------------------|--------------|
| **Step 1** | `01_data_profiling.py` | ✅ Correct |
| **Step 2** | `02_data_transformation.py` | ✅ Correct |
| **Step 3** | `03_feature_engineering.py` | ✅ Correct |
| **Step 4** | `05_feature_selection.py` | ⚠️ Should be 04 |
| **Step 5** | `10_equipment_id_audit.py` | ❌ Should be 05 |
| **Step 6** | `06_temporal_pof_model.py` | ⚠️ Happens to match |
| **Step 7** | `06_chronic_classifier.py` | ❌ Should be 07 |
| **Step 8** | `07_explainability.py` | ⚠️ Should be 08 |
| **Step 9** | `08_calibration.py` | ⚠️ Should be 09 |
| **Step 10** | `06_survival_model.py` | ❌ Should be 10 |
| **Step 11** | `10_consequence_of_failure.py` | ⚠️ Should be 11 |

### Issues Identified:

1. **Gap at 04**: No script named `04_*` (was `04_eda.py`, removed from main pipeline)
2. **Gap at 09**: No script named `09_*` (was `09_train_with_smote.py`, now archived)
3. **Three scripts start with 06**:
   - `06_temporal_pof_model.py` (executes 6th) ✓
   - `06_chronic_classifier.py` (executes 7th) ✗
   - `06_survival_model.py` (executes 10th) ✗
4. **Two scripts start with 10**:
   - `10_equipment_id_audit.py` (executes 5th) ✗
   - `10_consequence_of_failure.py` (executes 11th) ✓

---

## User Impact

**Confusing Scenarios**:

1. **Directory Listing**:
   ```bash
   $ ls *.py
   01_data_profiling.py
   02_data_transformation.py
   03_feature_engineering.py
   05_feature_selection.py         # Where's 04?
   06_chronic_classifier.py         # Which runs first?
   06_survival_model.py             # These three?
   06_temporal_pof_model.py         # Or alphabetically?
   10_equipment_id_audit.py         # Is this step 10?
   10_consequence_of_failure.py     # Or this?
   ```

2. **Manual Execution**:
   - User thinks: "I'll run 06, 07, 08 in order"
   - Actually runs: temporal_pof, explainability, calibration
   - **Skips**: chronic_classifier, survival_model ❌

3. **Documentation Confusion**:
   - Docs say "Step 7: Chronic Classifier"
   - File is named `06_chronic_classifier.py`
   - Mismatch between step number and filename

4. **Onboarding New Developers**:
   - Can't understand pipeline flow from filenames alone
   - Must read documentation to know execution order

---

## Root Cause Analysis

**Historical Changes**:

1. **04_eda.py was removed** from main pipeline → Left gap at 04
2. **09_train_with_smote.py was archived** → Left gap at 09
3. **Multiple model training scripts** grouped under 06 prefix (design decision)
4. **Diagnostic scripts** grouped under 10 prefix (design decision)
5. **Scripts never renumbered** after removal/archiving

**Design Intent** (likely):
- `06_*` = All model training scripts
- `10_*` = All risk assessment/diagnostic scripts
- **Problem**: Breaks sequential numbering convention

---

## Proposed Solutions

### **Option 1: Renumber to Match Execution Order** ⭐ RECOMMENDED

**Change all scripts to sequential 01-11**:

| Step | Current Name | Proposed Name |
|------|--------------|---------------|
| 1 | `01_data_profiling.py` | `01_data_profiling.py` ✓ |
| 2 | `02_data_transformation.py` | `02_data_transformation.py` ✓ |
| 3 | `03_feature_engineering.py` | `03_feature_engineering.py` ✓ |
| 4 | `05_feature_selection.py` | `04_feature_selection.py` |
| 5 | `10_equipment_id_audit.py` | `05_equipment_id_audit.py` |
| 6 | `06_temporal_pof_model.py` | `06_temporal_pof_model.py` ✓ |
| 7 | `06_chronic_classifier.py` | `07_chronic_classifier.py` |
| 8 | `07_explainability.py` | `08_explainability.py` |
| 9 | `08_calibration.py` | `09_calibration.py` |
| 10 | `06_survival_model.py` | `10_survival_model.py` |
| 11 | `10_consequence_of_failure.py` | `11_consequence_of_failure.py` |

**Benefits**:
- ✅ Execution order immediately clear from filenames
- ✅ No confusion for new developers
- ✅ Easier to explain pipeline flow
- ✅ Consistent sequential numbering

**Costs**:
- ❌ Breaking change - affects 8 files
- ❌ All documentation needs updating
- ❌ Git history becomes harder to track
- ❌ May break user scripts importing these modules

**Effort**: 4-6 hours (rename + update all references + test)

---

### **Option 2: Add Step Prefix to Filenames**

**Keep current names but add execution step**:

| Step | Current Name | Proposed Name |
|------|--------------|---------------|
| 1 | `01_data_profiling.py` | `step01_data_profiling.py` |
| 2 | `02_data_transformation.py` | `step02_data_transformation.py` |
| 3 | `03_feature_engineering.py` | `step03_feature_engineering.py` |
| 4 | `05_feature_selection.py` | `step04_feature_selection.py` |
| 5 | `10_equipment_id_audit.py` | `step05_equipment_id_audit.py` |
| ... | ... | ... |

**Benefits**:
- ✅ Clear execution order
- ✅ Explicit "step" terminology

**Costs**:
- ❌ Even more breaking changes (all 11 files)
- ❌ Longer filenames
- ❌ More work than Option 1

**Effort**: 6-8 hours
**Not Recommended**: More work for similar benefit

---

### **Option 3: Keep Current Names + Better Documentation**

**No renaming, improve docs instead**:
- Add prominent execution order table to README
- Add comments in each script: `# EXECUTION ORDER: Step 7/11`
- Update all documentation to clarify discrepancy

**Benefits**:
- ✅ No breaking changes
- ✅ Zero code changes
- ✅ Backward compatible

**Costs**:
- ❌ Confusion remains
- ❌ New developers still confused
- ❌ Technical debt persists

**Effort**: 1-2 hours (documentation only)
**Not Recommended**: Doesn't fix root problem

---

### **Option 4: Remove Numbers Entirely**

**Use descriptive names only**:

| Current | Proposed |
|---------|----------|
| `01_data_profiling.py` | `data_profiling.py` |
| `02_data_transformation.py` | `data_transformation.py` |
| `06_temporal_pof_model.py` | `temporal_pof_model.py` |
| ... | ... |

**Benefits**:
- ✅ No false expectations about order
- ✅ Cleaner names
- ✅ Focus on what script does, not when it runs

**Costs**:
- ❌ Loses visual ordering in directory listings
- ❌ Harder to see pipeline flow at a glance
- ❌ Breaking change to all 11 files

**Effort**: 4-6 hours
**Not Recommended**: Loses useful ordering information

---

## Recommendation

### **Choose Option 1: Sequential Renumbering**

**Why**:
1. **Principle of Least Surprise**: Numbers should reflect execution order
2. **Onboarding**: New developers can understand flow from filenames
3. **Documentation Alignment**: Step numbers match filenames
4. **Standard Practice**: Most pipelines use sequential numbering

**Implementation Plan**:

### Phase 1: Prepare (30 min)
1. Create mapping document (current → new names)
2. Audit all import statements
3. Audit all documentation references

### Phase 2: Rename Scripts (1-2h)
```bash
git mv 05_feature_selection.py 04_feature_selection.py
git mv 10_equipment_id_audit.py 05_equipment_id_audit.py
git mv 06_chronic_classifier.py 07_chronic_classifier.py
git mv 07_explainability.py 08_explainability.py
git mv 08_calibration.py 09_calibration.py
git mv 06_survival_model.py 10_survival_model.py
git mv 10_consequence_of_failure.py 11_consequence_of_failure.py
```

### Phase 3: Update References (2-3h)
**Files to Update**:
- `run_pipeline.py` - Update all script names (11 references)
- `config.py` - Any script name references
- `pipeline_validation.py` - Validation logic
- All documentation files:
  - `PIPELINE_EXECUTION_ORDER.md`
  - `docs/PIPELINE_USAGE.md`
  - `REMAINING_IMPROVEMENTS.md`
  - `README.md` (if exists)

**Search for references**:
```bash
grep -r "05_feature_selection\|10_equipment_id_audit\|06_chronic\|07_explainability\|08_calibration\|06_survival\|10_consequence" --include="*.py" --include="*.md"
```

### Phase 4: Test (1h)
1. Run full pipeline: `python run_pipeline.py`
2. Verify all steps execute correctly
3. Check all log outputs reference correct names
4. Verify documentation accuracy

### Phase 5: Update Git History Helpers (30 min)
Add `.gitattributes` entries to preserve git history:
```
04_feature_selection.py merge=union
07_chronic_classifier.py merge=union
# ... etc
```

---

## Alternative: Hybrid Approach

**Keep 06_* and 10_* groupings but document explicitly**:

Instead of renumbering everything, keep the logical grouping but add clear documentation:

```
01-03: Data Preparation
04-05: Feature Pipeline
06-XX: Model Training (all 06_* scripts)
  - 06_temporal_pof_model.py
  - 06_chronic_classifier.py
  - 06_survival_model.py
07-09: Model Analysis
10-XX: Risk Assessment (all 10_* scripts)
  - 10_equipment_id_audit.py
  - 10_consequence_of_failure.py
```

**Benefits**:
- ✅ Preserves logical grouping
- ✅ Less breaking changes (only need to fix 05 → 04)
- ✅ Maintains some semantic meaning in numbers

**Costs**:
- ⚠️ Still somewhat confusing
- ⚠️ Execution order not perfectly reflected

---

## Decision Matrix

| Option | Clarity | Effort | Breaking Changes | Recommended? |
|--------|---------|--------|------------------|--------------|
| **1. Sequential Renumber** | ⭐⭐⭐⭐⭐ | 4-6h | 8 files | ✅ **YES** |
| **2. Step Prefix** | ⭐⭐⭐⭐ | 6-8h | 11 files | ❌ |
| **3. Better Docs** | ⭐⭐ | 1-2h | 0 files | ❌ |
| **4. Remove Numbers** | ⭐⭐⭐ | 4-6h | 11 files | ❌ |
| **5. Hybrid Grouping** | ⭐⭐⭐ | 2-3h | 2 files | ⚠️ Maybe |

---

## Impact Assessment

**Who is affected?**:
- ✅ **Future developers**: Much easier onboarding
- ⚠️ **Current users**: Need to update any scripts importing these modules
- ⚠️ **Documentation**: All references need updating
- ⚠️ **Training materials**: Any screenshots/videos become outdated

**When to do this?**:
- **Best Time**: NOW (before wider deployment)
- **Worst Time**: After production deployment with many users
- **Acceptable Time**: During major version bump (e.g., v2.0 → v3.0)

---

## Recommendation

### **Implement Option 1 (Sequential Renumbering) NOW**

**Rationale**:
1. Codebase is still in active development (not production)
2. Minimal external users to impact
3. Technical debt is cheap to fix now, expensive later
4. Significantly improves maintainability

**Estimated Total Effort**: 4-6 hours
**Breaking Change Risk**: LOW (early development phase)
**Long-term Benefit**: HIGH (reduces confusion forever)

---

## Immediate Next Step

**DECISION REQUIRED**: Do you want to renumber scripts to match execution order?

- **YES** → I'll implement Option 1 (sequential renumbering, 4-6h)
- **NO, but improve docs** → I'll implement Option 3 (documentation only, 1-2h)
- **LATER** → Add to backlog (document the issue for future fix)

Let me know your preference and I'll proceed accordingly.

---

**Last Updated**: 2025-11-25
