# 🎯 DUAL-MODEL POF ARCHITECTURE (v4.0)

**Complete Failure Prediction System**
- **Model 1:** Temporal PoF (WHEN) - script `06_model_training.py`
- **Model 2:** Chronic Repeater (WHICH) - script `06_chronic_repeater.py`
- **Model 3:** Survival Analysis (HOW LONG) - script `09_survival_analysis.py`

---

## 📋 ARCHITECTURE OVERVIEW

### **Three Complementary Models:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE POF PREDICTION                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   TEMPORAL POF       │  │  CHRONIC REPEATER    │  │  SURVIVAL ANALYSIS   │
│   (WHEN)             │  │  (WHICH)             │  │  (HOW LONG)          │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ Script: 06_model     │  │ Script: 06_chronic   │  │ Script: 09_survival  │
│ Target: Future fails │  │ Target: Chronic flag │  │ Method: Cox PH       │
│ Output: 6M/12M PoF   │  │ Output: Binary class │  │ Output: Hazard rate  │
│ AUC: 0.75-0.85       │  │ AUC: 0.85-0.92       │  │ C-index: 0.75-0.80   │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  INTEGRATED RISK    │
                         │  (Script 10)        │
                         │  PoF × CoF → CAPEX  │
                         └─────────────────────┘
```

---

## 🎯 MODEL COMPARISON

| Aspect | Temporal PoF (Script 06) | Chronic Repeater (06_chronic) | Survival Analysis (Script 09) |
|--------|--------------------------|-------------------------------|------------------------------|
| **Question** | WHEN will it fail? | WHICH are failure-prone? | HOW LONG until failure? |
| **Target** | Future failures (6M/12M) | Tekrarlayan_90gün_Flag | Time-to-event |
| **Method** | XGBoost/CatBoost | XGBoost/CatBoost | Cox Proportional Hazards |
| **Output** | Probability (0-1) | Binary (0/1) | Hazard ratio |
| **Positive Class** | 6M: 20.8%, 12M: 33.7% | 12% (94 equipment) | Continuous |
| **Expected AUC** | 0.75-0.85 | 0.85-0.92 | C-index: 0.75-0.80 |
| **Use Case** | Maintenance scheduling | Replace vs Repair | Timeline planning |
| **Timeline** | 6 or 12 months | No timeline | 3M, 6M, 12M, 24M |

---

## 📊 EXAMPLE PREDICTIONS

### **Equipment 41905262:**

| Model | Prediction | Interpretation | Action |
|-------|-----------|----------------|--------|
| **Temporal PoF (6M)** | 85% | 85% chance of failure in next 6M | Schedule inspection within 3M |
| **Temporal PoF (12M)** | 95% | 95% chance of failure in next 12M | Plan replacement within 6M |
| **Chronic Repeater** | 1 (92% prob) | This IS a chronic repeater | **REPLACE** (not repair) |
| **Survival Analysis** | HR=2.5 | 2.5× higher failure rate than baseline | High priority for replacement |
| **Integrated Risk** | 87.5 | PoF × CoF = High risk score | **CAPEX Priority #1** |

**Business Decision:** **REPLACE within 3 months** (chronic repeater with high 6M PoF)

---

### **Equipment 42611980:**

| Model | Prediction | Interpretation | Action |
|-------|-----------|----------------|--------|
| **Temporal PoF (6M)** | 35% | 35% chance of failure in next 6M | Monitor only |
| **Temporal PoF (12M)** | 65% | 65% chance of failure in next 12M | Plan inspection in 9M |
| **Chronic Repeater** | 0 (15% prob) | NOT a chronic repeater | **REPAIR** when fails |
| **Survival Analysis** | HR=0.8 | Below average failure rate | Lower priority |
| **Integrated Risk** | 45.2 | PoF × CoF = Medium risk | **CAPEX Priority #50** |

**Business Decision:** **MONITOR + Plan preventive maintenance in 9 months** (not chronic, moderate 12M PoF)

---

## 🚀 EXECUTION WORKFLOW

### **Step 1: Run Temporal PoF** ✅
```bash
python 06_model_training.py
```

**Outputs:**
- `predictions/predictions_6m.csv` (164 equipment with high 6M PoF)
- `predictions/predictions_12m.csv` (266 equipment with high 12M PoF)

**Duration:** ~10-15 minutes

---

### **Step 2: Run Chronic Repeater Classification** ✅
```bash
python 06_chronic_repeater.py
```

**Outputs:**
- `predictions/chronic_repeaters.csv` (all 789 equipment)
- `results/high_risk_chronic_repeaters.csv` (~94 chronic repeaters)

**Duration:** ~5-8 minutes

---

### **Step 3: Combine Predictions** (Create Integration Script)
```bash
python combine_predictions.py  # We'll create this
```

**What It Does:**
1. Load temporal PoF predictions (6M and 12M)
2. Load chronic repeater predictions
3. Merge by equipment ID
4. Create integrated risk score
5. Generate actionable recommendations

---

### **Step 4: Run Survival Analysis**
```bash
python 09_survival_analysis.py
```

**Outputs:**
- Hazard ratios for each equipment
- Kaplan-Meier survival curves
- 3M, 6M, 12M, 24M failure probabilities

**Duration:** ~8-12 minutes

---

### **Step 5: Integrate with CoF**
```bash
python 10_consequence_of_failure.py
```

**Outputs:**
- Final CAPEX priority list
- Risk = PoF × CoF
- Budget recommendations

---

## 📋 INTEGRATION EXAMPLE

### **Create `combine_predictions.py`:**

```python
import pandas as pd

print("="*100)
print(" "*25 + "COMBINING DUAL-MODEL PREDICTIONS")
print("="*100)

# Load predictions
print("\n✓ Loading predictions...")
preds_6m = pd.read_csv('predictions/predictions_6m.csv')
preds_12m = pd.read_csv('predictions/predictions_12m.csv')
chronic = pd.read_csv('predictions/chronic_repeaters.csv')

# Merge all predictions
print("✓ Merging predictions...")
combined = preds_6m[['Ekipman_ID', 'Risk_Score']].rename(columns={'Risk_Score': 'PoF_6M'})
combined = combined.merge(
    preds_12m[['Ekipman_ID', 'Risk_Score']].rename(columns={'Risk_Score': 'PoF_12M'}),
    on='Ekipman_ID'
)
combined = combined.merge(
    chronic[['Ekipman_ID', 'Chronic_Repeater_Probability', 'Chronic_Repeater_Flag_Actual']],
    on='Ekipman_ID'
)

# Create action recommendations
def recommend_action(row):
    """
    Decision logic combining temporal PoF and chronic repeater status
    """
    is_chronic = row['Chronic_Repeater_Probability'] > 0.5
    high_6m = row['PoF_6M'] > 70
    high_12m = row['PoF_12M'] > 70

    if is_chronic and high_6m:
        return 'REPLACE_URGENT', '⚠️ Chronic repeater with high 6M PoF - Replace within 3M'
    elif is_chronic:
        return 'REPLACE_PLAN', '🔴 Chronic repeater - Plan replacement within 6-12M'
    elif high_6m:
        return 'INSPECT_URGENT', '🟠 High 6M PoF - Inspect within 2M, may need replacement'
    elif high_12m:
        return 'INSPECT_PLAN', '🟡 High 12M PoF - Schedule inspection in 6-9M'
    else:
        return 'MONITOR', '🟢 Low/Medium PoF - Routine monitoring'

combined[['Action', 'Recommendation']] = combined.apply(
    lambda row: pd.Series(recommend_action(row)), axis=1
)

# Priority score (0-100)
combined['Priority_Score'] = (
    combined['PoF_6M'] * 0.4 +  # 6M PoF weight: 40%
    combined['PoF_12M'] * 0.3 +  # 12M PoF weight: 30%
    combined['Chronic_Repeater_Probability'] * 100 * 0.3  # Chronic flag weight: 30%
)

# Sort by priority
combined = combined.sort_values('Priority_Score', ascending=False)

# Summary statistics
print(f"\n📊 COMBINED PREDICTION SUMMARY:")
print(f"   Total Equipment: {len(combined):,}")
print(f"\n--- Action Breakdown ---")
print(combined['Action'].value_counts())

print(f"\n--- Top 10 Priority Equipment ---")
for idx, (i, row) in enumerate(combined.head(10).iterrows(), 1):
    print(f"  {idx:2d}. ID: {row['Ekipman_ID']} | Priority: {row['Priority_Score']:.1f} | "
          f"Action: {row['Action']}")

# Save combined predictions
output_path = 'predictions/combined_pof_predictions.csv'
combined.to_csv(output_path, index=False)
print(f"\n✅ Combined predictions saved: {output_path}")

print("\n" + "="*100)
print(f"{'DUAL-MODEL PREDICTION INTEGRATION COMPLETE':^100}")
print("="*100)
```

---

## 🎯 DECISION MATRIX

### **Replace vs Repair Logic:**

| Chronic Repeater? | PoF_6M | PoF_12M | Action | Timeline |
|-------------------|--------|---------|--------|----------|
| ✅ YES | >70% | >80% | **REPLACE URGENT** | Within 3 months |
| ✅ YES | >50% | >60% | **REPLACE PLAN** | Within 6 months |
| ✅ YES | <50% | >50% | **REPLACE PLAN** | Within 12 months |
| ❌ NO | >70% | >80% | **INSPECT + REPAIR** | Within 2 months |
| ❌ NO | >50% | >60% | **INSPECT + REPAIR** | Within 6 months |
| ❌ NO | <50% | >60% | **MONITOR + PLAN** | Within 9 months |
| ❌ NO | <50% | <50% | **ROUTINE MONITOR** | Routine maintenance |

---

## 📊 EXPECTED RESULTS

### **After Running Both Scripts:**

**Temporal PoF (Script 06):**
```
6M Predictions:
  High Risk (>50%): 164 equipment (20.8%)
  Medium Risk (30-50%): ~150 equipment (19%)
  Low Risk (<30%): ~475 equipment (60%)

12M Predictions:
  High Risk (>50%): 266 equipment (33.7%)
  Medium Risk (30-50%): ~200 equipment (25%)
  Low Risk (<30%): ~323 equipment (41%)
```

**Chronic Repeater (Script 06_chronic):**
```
Chronic Repeaters:
  Critical (>70%): ~50 equipment (6%)
  High (50-70%): ~40 equipment (5%)
  Medium (30-50%): ~100 equipment (13%)
  Low (<30%): ~599 equipment (76%)
```

**Combined (Integration Script):**
```
Action Breakdown:
  REPLACE_URGENT:     40-50 equipment (5-6%)  ← Chronic + High 6M
  REPLACE_PLAN:       60-70 equipment (8-9%)  ← Chronic + Moderate PoF
  INSPECT_URGENT:     60-80 equipment (8-10%) ← High 6M, not chronic
  INSPECT_PLAN:       100-130 equipment (13-16%) ← High 12M, not chronic
  MONITOR:            450-530 equipment (57-67%) ← Low PoF
```

---

## 💡 BUSINESS VALUE

### **Why Dual Models?**

**Model 1 (Temporal PoF) Alone:**
- ✅ Tells you WHEN equipment will fail
- ❌ Doesn't distinguish chronic vs occasional failures
- ❌ Can't determine Replace vs Repair

**Model 2 (Chronic Repeater) Alone:**
- ✅ Identifies failure-prone equipment
- ❌ No timeline (urgent vs can wait)
- ❌ Can't prioritize by immediacy

**Combined (Dual Models):**
- ✅ **WHO** needs attention (chronic repeaters)
- ✅ **WHEN** they'll fail (6M vs 12M)
- ✅ **WHAT** action (replace vs repair)
- ✅ **HOW URGENT** (3M vs 6M vs 12M)

---

### **ROI Example:**

**Scenario:** 100 equipment predicted for replacement

**Option A: Replace All (No Model)**
- Cost: 100 × $10,000 = $1,000,000
- Waste: ~40% replacements were unnecessary
- Loss: $400,000

**Option B: Temporal PoF Only**
- Identifies: 60 high-risk equipment
- Cost: 60 × $10,000 = $600,000
- Miss: 15 chronic repeaters (repeat failures)
- Additional repair costs: 15 × $2,000 × 3 failures = $90,000
- Total: $690,000

**Option C: Dual Models (Recommended)**
- Identifies: 45 chronic repeaters (REPLACE)
- Identifies: 55 high temporal PoF but not chronic (REPAIR)
- Cost: 45 × $10,000 + 55 × $2,000 = $560,000
- Savings: $440,000 vs Option A
- Savings: $130,000 vs Option B

---

## 🚀 QUICK START GUIDE

### **Run All Models in Sequence:**

```bash
# Step 1: Temporal PoF (WHEN)
python 06_model_training.py

# Step 2: Chronic Repeater (WHICH)
python 06_chronic_repeater.py

# Step 3: Combine predictions
python combine_predictions.py  # Create from template above

# Step 4: Survival Analysis (HOW LONG)
python 09_survival_analysis.py

# Step 5: Integrate with CoF
python 10_consequence_of_failure.py
```

**Total Runtime:** ~30-45 minutes

---

### **Output Files:**

```
predictions/
├── predictions_6m.csv              ← Temporal PoF (6M)
├── predictions_12m.csv             ← Temporal PoF (12M)
├── chronic_repeaters.csv           ← Chronic repeater classification
├── combined_pof_predictions.csv    ← Integrated predictions ⭐
└── survival_predictions.csv        ← Cox model predictions

results/
├── high_risk_chronic_repeaters.csv
├── chronic_repeater_feature_importance.csv
├── temporal_pof_feature_importance.csv
└── final_capex_priority_list.csv   ← Final CAPEX list ⭐
```

---

## ❓ FAQ

### **Q1: Which model should I trust more?**

**A:** Use BOTH! They answer different questions:
- **Chronic Repeater:** Tells you WHO to replace (inherently failure-prone)
- **Temporal PoF:** Tells you WHEN to act (timing and urgency)

Combine: Chronic repeaters with high 6M PoF = **Top Priority**

---

### **Q2: What if they disagree?**

**Example:** Chronic=YES but PoF_6M=20%

**Interpretation:**
- Equipment IS failure-prone (chronic repeater)
- But unlikely to fail in next 6 months
- **Action:** Plan replacement in 12-18 months (not urgent)

**Example:** Chronic=NO but PoF_6M=80%

**Interpretation:**
- Equipment NOT a chronic repeater (few historical failures)
- But high probability of failure soon (e.g., aging, stress)
- **Action:** Inspect urgently, repair if possible (replacement not critical)

---

### **Q3: How often should I re-run?**

**Recommendation:**
- **Quarterly:** Re-run temporal PoF (new data every 3 months)
- **Annually:** Re-run chronic repeater (stable classification)
- **As needed:** Re-run after major failures or replacements

---

### **Q4: Can I use only one model?**

**Yes, but limitations:**

**Temporal PoF Only:**
- ✅ Good for maintenance scheduling
- ❌ May replace equipment that could be repaired
- ❌ May repair chronic repeaters repeatedly (waste)

**Chronic Repeater Only:**
- ✅ Good for long-term replacement planning
- ❌ No urgency (can't prioritize)
- ❌ May miss equipment about to fail (not chronic yet)

**Best Practice:** Use both models together!

---

## 📝 SUMMARY

### **Dual-Model Architecture (v4.0):**

| Component | Script | Purpose | Output |
|-----------|--------|---------|--------|
| **Temporal PoF** | 06_model_training.py | WHEN will equipment fail? | 6M/12M probabilities |
| **Chronic Repeater** | 06_chronic_repeater.py | WHICH are failure-prone? | Binary classification |
| **Integration** | combine_predictions.py | Combine both models | Actionable recommendations |
| **Survival Analysis** | 09_survival_analysis.py | HOW LONG until failure? | Hazard rates, survival curves |
| **CoF Integration** | 10_consequence_of_failure.py | PoF × CoF → Risk | CAPEX priority list |

---

**Version History:**
- **v3.1:** Single chronic repeater model (overfitted, AUC=1.0)
- **v4.0:** Dual-model architecture (Temporal + Chronic) ← **CURRENT**

---

**Ready to run!** 🚀

```bash
# Run temporal PoF first
python 06_model_training.py

# Then run chronic repeater
python 06_chronic_repeater.py
```

**END OF DOCUMENT**
