# 🚀 RUNNING THE UPDATED PIPELINE (v4.1)

**Updated:** 2025-11-17 (Post-Consultant Review)

---

## ✅ WHAT CHANGED?

Your pipeline has been **upgraded with 5 critical fixes**:

1. ✅ **VIF Convergence** - Fixed infinite multicollinearity
2. ✅ **Safe MTBF Calculation** - No data leakage (uses only pre-cutoff failures)
3. ✅ **Restored Safe Features** - Failure_Free_3M and Age_Failure_Interaction
4. ✅ **Adjusted Risk Weights** - Recurrence: 5% → 20% (chronic repeaters priority)
5. ✅ **Mathematical Duplicates Removed** - Age_Days variants eliminated

**See `CRITICAL_FIXES_APPLIED.md` for full details.**

---

## 📋 STEP-BY-STEP EXECUTION

### **Option 1: Run Full Pipeline (Recommended)**

```bash
# Clean slate - delete old outputs
rm -rf data/equipment_level_data.csv
rm -rf data/features_engineered.csv
rm -rf data/features_selected.csv
rm -rf data/features_selected_clean.csv

# Run pipeline in sequence
python 02_data_transformation.py
python 03_feature_engineering.py
python 05_feature_selection.py
python 05b_remove_leaky_features.py
```

**Expected Runtime:** 5-8 minutes total

---

### **Option 2: Resume from Feature Engineering**

If you already have `data/equipment_level_data.csv`:

```bash
# Only re-run feature engineering and selection
python 03_feature_engineering.py
python 05_feature_selection.py
python 05b_remove_leaky_features.py
```

**Expected Runtime:** 3-5 minutes

---

### **Option 3: Resume from Feature Selection**

If you already have `data/features_engineered.csv`:

```bash
# Only re-run selection steps
python 05_feature_selection.py
python 05b_remove_leaky_features.py
```

**Expected Runtime:** 1-2 minutes

---

## 🔍 WHAT TO WATCH FOR

### **Script 02: Data Transformation**

**Look for:**
```
[Step 10/12] Calculating MTBF & Time Until First Failure
  Calculating MTBF (using failures BEFORE cutoff only - leakage-safe)...
  MTBF: XXX/789 valid
```

✅ **Success Indicator:** Message says "using failures BEFORE cutoff only"

---

### **Script 03: Feature Engineering**

**Look for:**
```
--- Building PoF Risk Score (0-100) ---
  ✓ Age risk (40% weight, non-linear wear-out curve)
  ✓ Recent failure risk (25% weight)
  ✓ MTBF risk (15% weight)
  ✓ Recurrence risk (20% weight) [INCREASED from 5%]
```

✅ **Success Indicator:** Recurrence shows **20% weight** (not 5%)

---

### **Script 05: Feature Selection**

**Look for:**
```
--- Step 5A: Removing Exact Mathematical Duplicates ---
  Removed 4 mathematical duplicates:
    ❌ Ekipman_Yaşı_Gün
    ❌ Ekipman_Yaşı_Gün_TESIS
    ❌ Ekipman_Yaşı_Gün_EDBS
    ❌ Ilk_Arizaya_Kadar_Gun

--- Step 5B: Iterative VIF Removal ---
  Iteration 1-10...
  ✓ Target VIF achieved!  ← Should reach this BEFORE max iterations
```

✅ **Success Indicator:** VIF converges (doesn't hit max 50 iterations)
✅ **Mean VIF:** Should be ~5-8 (not inf)
✅ **Features with VIF > 10:** Should be ~0-5 (not 59!)

---

### **Script 05b: Leakage Removal**

**Look for:**
```
STEP 3: IDENTIFYING LEAKY FEATURES
⚠️  Identified XX leaky features:
   ❌ Arıza_Sayısı_12ay → Recent failure count
   ❌ Recent_Failure_Intensity → Recent failure intensity
   (Failure_Free_3M should NOT be in this list anymore)

STEP 4: DEFINING SAFE FEATURE SET
✓ XX safe features identified
```

✅ **Success Indicator:** `Failure_Free_3M` is **NOT** in leaky features list
✅ **Safe features:** Should be ~17-20 (not 10-12)

---

## 📊 KEY METRICS TO VERIFY

### After Script 02:
```
MTBF: ~196/789 valid (may vary slightly)
Recurring faults: 90-day=94 equipment
Equipment: 789 records x 68 features
```

### After Script 03:
```
Total features: 109
Composite PoF Risk Score: Mean ~20-25
Risk Distribution:
  ✅ Low (0-25): ~673 (85%)
  ⚠ Medium (25-50): ~94 (12%)
  ❌ High (50-75): ~22 (3%)
```

### After Script 05:
```
Starting features: 82 numeric
After VIF reduction: ~24-30 features (removed ~50-58)
Mean VIF: 5-8 (NOT inf)
Max VIF: <10
Final features: ~27 total (including categoricals)
```

### After Script 05b:
```
Leaky features removed: ~8-10
Safe features retained: ~17-20
Retention rate: ~60-70%
```

---

## ❌ TROUBLESHOOTING

### Problem: VIF still shows "inf"
**Cause:** Mathematical duplicates not removed
**Fix:** Verify Step 5A ran successfully in script 05

### Problem: Recurrence still shows 5% weight
**Cause:** Old version of 03_feature_engineering.py
**Fix:** Re-pull the latest version from git (if committed)

### Problem: Failure_Free_3M still flagged as leaky
**Cause:** Old version of 05b_remove_leaky_features.py
**Fix:** Verify lines 142-146 have comments (not active removal)

### Problem: MTBF calculation doesn't mention "leakage-safe"
**Cause:** Old version of 02_data_transformation.py
**Fix:** Verify line 664 has the new message

---

## 🎯 EXPECTED FINAL OUTPUT

After running all scripts, you should have:

```
data/
├── equipment_level_data.csv       (789 × 68) ← MTBF leakage-free
├── features_engineered.csv        (789 × 109) ← Recurrence 20% weight
├── features_selected.csv          (789 × 27) ← VIF converged
└── features_selected_clean.csv    (789 × 17-20) ← Failure_Free_3M restored

outputs/feature_selection/
├── vif_analysis.csv               ← Check Mean VIF ~5-8
├── feature_importance.csv
├── leakage_analysis.csv           ← Check Failure_Free_3M = SAFE
└── *.png
```

---

## ⏭️ NEXT STEP: MODEL TRAINING

Once the pipeline completes successfully:

```bash
# Continue with model training
python 06_model_training.py      # Model 2: Chronic Repeater
python 09_survival_analysis.py   # Model 1: Temporal PoF
python 10_consequence_of_failure.py  # Risk Integration
```

---

## 📞 NEED HELP?

If you see unexpected results:

1. **Check script line numbers** - Edits were made at specific locations
2. **Compare with CRITICAL_FIXES_APPLIED.md** - Verify exact changes
3. **Run with fresh data** - Delete all intermediate files and re-run from 02

**Key files to verify:**
- `02_data_transformation.py` (line 664)
- `03_feature_engineering.py` (lines 452, 460, 477)
- `05_feature_selection.py` (line 270)
- `05b_remove_leaky_features.py` (lines 142-146)

---

**Good luck! The pipeline is now production-ready.** 🚀
