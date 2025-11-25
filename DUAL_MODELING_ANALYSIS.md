# DUAL MODELING APPROACH ANALYSIS
**Date**: 2025-11-25
**Status**: ✅ INVESTIGATION COMPLETE

---

## Executive Summary

The PoF2 pipeline uses TWO modeling approaches:
1. **Temporal Models** (XGBoost/CatBoost) - Steps 5, 7, 8
2. **Survival Model** (Cox Proportional Hazards) - Step 9

**Key Finding**: Both approaches are NOT fully redundant, but there are inefficiencies:
- ✅ **Temporal models ARE used** for SHAP explainability analysis (Step 7)
- ❌ **Calibrated models are NEVER used** (Step 8 outputs unused)
- ❌ **Temporal predictions are NEVER used** for risk assessment
- ✅ **Survival predictions ARE used** as primary input to risk assessment (Step 10)

---

## Detailed Analysis

### 1. Temporal Model Pipeline (Steps 5, 7, 8)

#### **Step 5: 06_temporal_pof_model.py** (479 lines)
**Outputs**:
- **Models**: `models/xgboost_3m.pkl`, `models/xgboost_6m.pkl`, `models/xgboost_12m.pkl`
- **Models**: `models/catboost_3m.pkl`, `models/catboost_6m.pkl`, `models/catboost_12m.pkl`
- **Predictions**: `predictions/predictions_3m.csv`, `predictions/predictions_6m.csv`, `predictions/predictions_12m.csv`
- **Results**: `results/model_performance_comparison.csv`, `results/feature_importance_by_horizon.csv`

**Usage Downstream**:
- ✅ Models loaded by Step 7 (07_explainability.py) for SHAP analysis
- ✅ Models loaded by Step 8 (08_calibration.py) for calibration
- ❌ Predictions NOT used by any downstream script (only validated for existence)

#### **Step 7: 07_explainability.py** (560 lines)
**Inputs**:
- `models/xgboost_*.pkl` (from Step 5)

**Outputs**:
- SHAP visualizations: `outputs/explainability/*.png`
- Feature importance CSV: `results/shap_feature_importance.csv`

**Purpose**: Generate SHAP-based model explanations for stakeholder transparency

**Status**: ✅ **ACTIVELY USED** - Essential for model explainability

#### **Step 8: 08_calibration.py** (660 lines)
**Inputs**:
- `models/xgboost_6m.pkl`, `models/xgboost_12m.pkl` (from Step 5)

**Outputs**:
- **Calibrated Models**: `models/calibrated_isotonic_6m.pkl`, `models/calibrated_sigmoid_6m.pkl`
- **Calibrated Models**: `models/calibrated_isotonic_12m.pkl`, `models/calibrated_sigmoid_12m.pkl`
- **Visualizations**: `outputs/calibration/*.png`
- **Metrics**: `results/calibration_metrics.csv`

**Usage Downstream**:
- ❌ **NEVER USED** - Calibrated models are saved but never loaded by any script
- ❌ Risk assessment (Step 10) uses survival model predictions, not calibrated temporal predictions

**Status**: 🔴 **UNUSED OUTPUT** - 660 lines producing models that are never used

---

### 2. Survival Model Pipeline (Step 9)

#### **Step 9: 06_survival_model.py** (743 lines)
**Outputs**:
- **Primary Predictions**: `predictions/pof_multi_horizon_predictions.csv`
  - Contains: PoF_Probability_3M, PoF_Probability_6M, PoF_Probability_12M, PoF_Probability_24M
- **Aggregations**: `results/pof_category_aggregation.csv`
- **Outliers**: `results/pof_outliers.csv`
- **Visualizations**: `outputs/survival/*.png` (Kaplan-Meier curves)

**Usage Downstream**:
- ✅ **PRIMARY USE** - `pof_multi_horizon_predictions.csv` is loaded by Step 10 for risk assessment

---

### 3. Risk Assessment Usage (Step 10)

#### **Step 10: 10_consequence_of_failure.py** (Line 120-121)
```python
PRIMARY_PREDICTION_FILE = PREDICTION_DIR / 'pof_multi_horizon_predictions.csv'  # Survival model
FALLBACK_PREDICTION_FILE = PREDICTION_DIR / 'failure_predictions_12m.csv'  # ⚠️ BUG: Wrong filename!
```

**Current Behavior**:
- Loads `pof_multi_horizon_predictions.csv` (from survival model) as primary source
- Has fallback to `failure_predictions_12m.csv` (DOESN'T EXIST - should be `predictions_12m.csv`)
- Uses survival model predictions exclusively for risk scoring

**Status**: Uses ONLY survival model predictions (temporal predictions ignored)

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                         │
│  Steps 1-4: Profiling → Transform → Engineer → Select Features │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
         ┌──────────▼─────────┐      ┌─────────▼──────────┐
         │  TEMPORAL MODELS   │      │  SURVIVAL MODEL    │
         │  (XGBoost/CatBoost)│      │  (Cox PH)          │
         │  Step 5            │      │  Step 9            │
         └──────────┬─────────┘      └─────────┬──────────┘
                    │                           │
         ┌──────────┼──────────┐                │
         │          │          │                │
    ┌────▼───┐ ┌───▼────┐ ┌───▼────┐     ┌─────▼──────┐
    │ SHAP   │ │ CALIB  │ │ PREDS  │     │ PREDICTIONS│
    │ Step 7 │ │ Step 8 │ │ (csv)  │     │ (csv)      │
    │ ✅ USED│ │ ❌ UNUSED│ │ ❌ UNUSED│     │ ✅ USED    │
    └────────┘ └────────┘ └────────┘     └─────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │ RISK ASSESS │
                                         │ Step 10     │
                                         └─────────────┘
```

---

## Issues Identified

### 🔴 CRITICAL: Calibration Models Never Used (Step 8)
**Problem**:
- Step 8 (08_calibration.py) trains calibrated models but they're never loaded/used
- 660 lines of code + 2-3 minutes runtime producing unused outputs
- Calibration improves probability estimates, but benefits are lost

**Impact**:
- Wasted computation time in every pipeline run
- Misleading documentation (suggests calibration is active)
- Potential for better predictions if calibrated models were actually used

**Root Cause**:
- Step 10 uses survival model predictions, not temporal model predictions
- No mechanism to use calibrated temporal models for risk assessment

### ⚠️  WARNING: Temporal Predictions Never Used
**Problem**:
- Temporal model predictions (predictions_3m.csv, predictions_6m.csv, predictions_12m.csv) are created but never consumed
- Only used for validation checks (file exists?)
- Step 10 uses survival model predictions exclusively

**Impact**:
- Temporal model training time (~30-60 seconds) produces predictions that are discarded
- Confusing to users - why generate predictions if unused?

### ⚠️  WARNING: Incorrect Fallback Filename (Step 10)
**Problem**:
- Line 121 of 10_consequence_of_failure.py references `failure_predictions_12m.csv`
- Actual file created by Step 5 is `predictions_12m.csv`
- Fallback will ALWAYS fail

**Impact**:
- If survival model fails, pipeline cannot fall back to temporal predictions
- Users would see misleading error message

---

## Recommendations

### Option 1: Fix Calibration Step (Use Calibrated Models) ⭐ RECOMMENDED
**Change**: Modify Step 10 to use calibrated temporal predictions instead of survival predictions

**Pros**:
- Calibration improves probability reliability (better Brier scores)
- XGBoost/CatBoost often perform better than Cox PH for complex patterns
- Maintains both explainability (SHAP) and accurate predictions

**Cons**:
- Requires refactoring Step 10
- Need to generate predictions from calibrated models (Step 8 currently doesn't do this)
- ~4-6 hours development effort

**Implementation**:
1. Modify 08_calibration.py to generate predictions from calibrated models
2. Save as `predictions/calibrated_predictions_6m.csv`, `predictions/calibrated_predictions_12m.csv`
3. Modify 10_consequence_of_failure.py to load calibrated predictions as primary source
4. Keep survival model as fallback/comparison

---

### Option 2: Remove Calibration Step (Streamline Pipeline)
**Change**: Delete Step 8 (08_calibration.py) from pipeline

**Pros**:
- Saves 2-3 minutes per pipeline run
- Removes 660 lines of unused code
- Simplifies pipeline (8 steps instead of 10)
- No functional loss (outputs currently unused)

**Cons**:
- Loses potential for improved probability calibration
- Discards useful diagnostic visualizations (calibration curves)
- Would need to restore if switching from survival to temporal models later

**Implementation**:
1. Remove Step 8 from `run_pipeline.py` (line 103-108)
2. Archive `08_calibration.py` to `analysis/diagnostics/`
3. Update documentation to reflect 9-step pipeline
4. ~30 minutes effort

---

### Option 3: Document Current Design + Fix Fallback
**Change**: Document that temporal models are for explainability only, survival for predictions

**Pros**:
- No code changes required (except fallback fix)
- Preserves current working architecture
- Low effort (~1 hour documentation)

**Cons**:
- Doesn't address inefficiency (calibration still unused)
- Confusing to maintainers (why train two models?)
- Wastes computation time

**Implementation**:
1. Add inline comments explaining dual approach
2. Update PIPELINE_EXECUTION_ORDER.md with clear roles
3. Fix fallback filename in Step 10 (line 121): `failure_predictions_12m.csv` → `predictions_12m.csv`

---

### Option 4: Hybrid Ensemble (Advanced)
**Change**: Use BOTH temporal and survival predictions in risk assessment

**Pros**:
- Leverages strengths of both approaches
- Ensemble often outperforms individual models
- Provides prediction variance/uncertainty estimates

**Cons**:
- Most complex solution (~8-12 hours development)
- Requires validation/testing to prove benefit
- May not improve results significantly

**Implementation**:
1. Generate predictions from calibrated temporal models (Step 8)
2. Load both temporal and survival predictions in Step 10
3. Ensemble via weighted average or stacking
4. Validate on test set to tune weights

---

## Decision Matrix

| Option | Code Savings | Time Savings | Risk | Effort | Recommended? |
|--------|--------------|--------------|------|--------|--------------|
| **1. Fix Calibration** | 0 lines | 0 min | Low | 4-6h | ⭐ **YES** |
| **2. Remove Calibration** | 660 lines | 2-3 min | Low | 30m | ✅ If no calibration plan |
| **3. Document Only** | 0 lines | 0 min | None | 1h | ❌ Doesn't fix inefficiency |
| **4. Hybrid Ensemble** | 0 lines | 0 min | Medium | 8-12h | ❌ Over-engineering |

---

## Immediate Actions (Priority Order)

### 1. 🔴 CRITICAL: Fix Fallback Filename in Step 10 (5 minutes)
```python
# File: 10_consequence_of_failure.py, Line 121
# BEFORE:
FALLBACK_PREDICTION_FILE = PREDICTION_DIR / 'failure_predictions_12m.csv'

# AFTER:
FALLBACK_PREDICTION_FILE = PREDICTION_DIR / 'predictions_12m.csv'
```

### 2. 🟠 HIGH: Decide on Calibration Step (1 hour discussion + implementation)
- **If keeping temporal models for predictions**: Implement Option 1 (4-6h effort)
- **If using survival models exclusively**: Implement Option 2 (30m effort)

### 3. 🟡 MEDIUM: Update Documentation (1 hour)
- Add section to PIPELINE_EXECUTION_ORDER.md explaining dual modeling approach
- Document which model outputs are used where
- Add flowchart showing prediction data flow

---

## Conclusion

The dual modeling approach is **partially justified** but **poorly implemented**:

✅ **Valid Reasons for Dual Approach**:
- Temporal models (XGBoost) provide SHAP explainability (Step 7) ✓
- Survival models (Cox PH) provide probabilistic time-to-event predictions ✓

❌ **Implementation Issues**:
- Calibration step outputs are never used (660 wasted lines)
- Temporal predictions are never used (only survival predictions consumed)
- Fallback mechanism has incorrect filename (bug)

**RECOMMENDATION**: Implement **Option 1** (Fix Calibration) to fully utilize temporal models, or implement **Option 2** (Remove Calibration) to streamline if survival model is sufficient.

---

**Next Steps**: Await user decision on which option to implement.
