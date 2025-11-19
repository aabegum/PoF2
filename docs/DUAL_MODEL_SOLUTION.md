# DUAL-MODEL SOLUTION FOR POF
**Turkish EDAŞ - Complete Asset Management System**

## 🎯 YOUR COMPLETE REQUIREMENTS

You want **BOTH** outputs:
1. ✅ **Temporal Prediction**: "Predict which equipment will fail in next 3/6/12/24 months"
2. ✅ **Chronic Repeater Identification**: "Among failed equipment, identify chronic repeaters for proactive replacement"

**Solution**: Implement **TWO COMPLEMENTARY MODELS**

---

## ❌ PROBLEM: Why 6M/12M Performance is Identical

### **Current Model Issue**:
```python
# From 06_model_training.py (lines 90-93)
TARGET_THRESHOLDS = {
    '6M': 2,   # >= 2 lifetime failures
    '12M': 2   # >= 2 lifetime failures (SAME!)
}

# Line 214
targets[horizon_name] = (df_full['Toplam_Arıza_Sayisi_Lifetime'] >= 2).astype(int)
```

**Result**:
```
Equipment A: Toplam_Arıza_Sayisi_Lifetime = 3
├─ 6M Target:  3 >= 2 → 1 (high-risk)
├─ 12M Target: 3 >= 2 → 1 (high-risk)  ← IDENTICAL!
└─ Prediction: Same for both horizons

Equipment B: Toplam_Arıza_Sayisi_Lifetime = 1
├─ 6M Target:  1 >= 2 → 0 (low-risk)
├─ 12M Target: 1 >= 2 → 0 (low-risk)  ← IDENTICAL!
└─ Prediction: Same for both horizons
```

**Why This is Wrong**:
- ❌ Target is **static** (based on total lifetime failures)
- ❌ No temporal component (doesn't look at WHEN failures occur)
- ❌ 6M and 12M targets are **mathematically identical**
- ❌ Cannot answer: "Will equipment fail in next N months?"

**What You're Actually Predicting**:
> "Is this equipment a chronic repeater?" (static classification)

**NOT**:
> "Will this equipment fail in next 6/12 months?" (temporal probability)

---

## ✅ SOLUTION: Dual-Model Architecture

### **MODEL 1: Survival Analysis (Temporal PoF)** 🎯 **NEW**

**Purpose**: Predict **WHEN** equipment will fail next

**Methodology**: Survival Analysis (Cox + Random Survival Forest)

**Target Structure**:
```python
Equipment_ID | Time_To_Next_Failure | Event_Occurred | Covariates
2037800      | 45 days              | 1 (failed)     | Age=12y, Class=Ayırıcı, Cause=mekanik, Customers=250
2038116      | 180 days             | 0 (censored)   | Age=8y, Class=Rekortman, Cause=elektriksel, Customers=150
```

**Output** (Single Model, Multiple Horizons):
```
Equipment 2037800 (Ayırıcı, 12 years, mekanik arıza, 250 customers):

Survival Probabilities:
├─ 3M (90d):   P(survive) = 0.52  →  48% failure risk  → HIGH
├─ 6M (180d):  P(survive) = 0.35  →  65% failure risk  → CRITICAL
├─ 12M (365d): P(survive) = 0.18  →  82% failure risk  → CRITICAL
└─ 24M (730d): P(survive) = 0.08  →  92% failure risk  → REPLACE

Median Time-To-Failure: 4.2 months
Recommended Action: Schedule replacement within 4 months
```

**Different Horizons = Different Predictions!** ✅

---

### **MODEL 2: Chronic Repeater Classifier (Current Model)** ✅ **EXISTING**

**Purpose**: Identify equipment with repeated failure patterns

**Methodology**: XGBoost/CatBoost classification

**Target**:
```python
Target = 1 if Toplam_Arıza_Sayisi_Lifetime >= 2, else 0
```

**Output**:
```
Equipment 2037800:
├─ PoF Score: 0.986 (98.6% chronic repeater probability)
├─ Class: Ayırıcı
├─ Recurring Failures: 90-day repeater (flag=1)
├─ Risk Category: CRITICAL
└─ Recommended Action: Replace (not repair)
```

**Use Case**: Identify equipment for replacement vs repair decisions

---

## 📊 HOW TWO MODELS WORK TOGETHER

### **Decision Matrix**:

| Equipment | Model 1: Survival | Model 2: Chronic | Decision |
|-----------|------------------|------------------|----------|
| **A** | 6M failure: 15% | Chronic: NO (0.2) | **Monitor** - Low temporal risk, not chronic |
| **B** | 6M failure: 75% | Chronic: NO (0.3) | **Urgent Maintenance** - High temporal risk, first-time issue |
| **C** | 6M failure: 20% | Chronic: YES (0.9) | **Replace** - Chronic repeater, even if low short-term risk |
| **D** | 6M failure: 85% | Chronic: YES (0.95) | **IMMEDIATE REPLACE** - Both models agree |

**Combined Risk Score**:
```python
# Option 1: Weighted average
Combined_Risk = 0.6 × Survival_Risk_6M + 0.4 × Chronic_Score

# Option 2: Multiplicative (conservative)
Combined_Risk = Survival_Risk_6M × Chronic_Score

# Option 3: Max (most conservative)
Combined_Risk = max(Survival_Risk_6M, Chronic_Score)
```

---

## 🔧 IMPLEMENTATION PLAN

### **PHASE 1: Keep Current Model (Chronic Repeater)** ✅
**Status**: Working (AUC 0.88)
**File**: `06_model_training.py`
**Use**: Chronic repeater identification

**Action**: No changes needed (already implemented)

---

### **PHASE 2: Add Survival Analysis (Temporal PoF)** 🎯 **NEW**

**Create New File**: `06b_survival_analysis.py`

#### **Step 1: Prepare Survival Data Structure**
```python
# For each equipment, calculate time-to-next-failure
survival_data = []

for equipment_id in equipment_list:
    # Get all faults for this equipment, sorted by date
    faults = df[df['Ekipman_ID'] == equipment_id].sort_values('started at')

    # Calculate time between consecutive failures
    for i in range(len(faults)):
        if i < len(faults) - 1:
            # Time to NEXT failure (not censored)
            time_to_event = (faults.iloc[i+1]['started at'] - faults.iloc[i]['started at']).days
            event_occurred = 1
        else:
            # Time to reference date (censored - no failure observed after this)
            time_to_event = (REFERENCE_DATE - faults.iloc[i]['started at']).days
            event_occurred = 0

        # Get covariates at time of observation
        covariates = {
            'Equipment_Age': calculate_age_at_time(equipment_id, faults.iloc[i]['started at']),
            'Equipment_Class': get_class(equipment_id),
            'Cause_Code': faults.iloc[i]['cause code'],
            'Customer_Impact': get_customer_impact(equipment_id),
            'Prior_Failures': i,  # Number of failures before this
            'MTBF': calculate_mtbf_up_to(equipment_id, i),
            # ... more features
        }

        survival_data.append({
            'Equipment_ID': equipment_id,
            'Time_To_Event': time_to_event,
            'Event_Occurred': event_occurred,
            **covariates
        })
```

#### **Step 2: Train Survival Models**
```python
from sksurv.ensemble import RandomSurvivalForest
from lifelines import CoxPHFitter

# Random Survival Forest (non-parametric, high accuracy)
rsf = RandomSurvivalForest(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42
)

# Convert to structured array for scikit-survival
y = np.array(
    [(row['Event_Occurred'], row['Time_To_Event'])
     for _, row in survival_df.iterrows()],
    dtype=[('Status', '?'), ('Time', '<f8')]
)

X = survival_df[feature_columns]

# Train
rsf.fit(X, y)

# Cox Proportional Hazards (parametric, interpretable)
cph = CoxPHFitter()
cph.fit(survival_df, duration_col='Time_To_Event', event_col='Event_Occurred')
```

#### **Step 3: Generate Predictions**
```python
# Predict survival probabilities for all equipment
horizons = [90, 180, 365, 730]  # 3M, 6M, 12M, 24M

predictions = {}
for equipment_id in equipment_list:
    equipment_features = get_current_features(equipment_id)

    # Random Survival Forest predictions
    survival_function = rsf.predict_survival_function(equipment_features)

    predictions[equipment_id] = {
        'Survival_3M': survival_function(90)[0],
        'Survival_6M': survival_function(180)[0],
        'Survival_12M': survival_function(365)[0],
        'Survival_24M': survival_function(730)[0],
        'Median_TTF': get_median_survival_time(survival_function),
        'Hazard_Ratio': get_hazard_ratio(cph, equipment_features)
    }
```

#### **Step 4: Create Survival Curves (Module 1 Requirement)**
```python
# Kaplan-Meier curves by equipment class
from lifelines import KaplanMeierFitter

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for i, equipment_class in enumerate(['Ayırıcı', 'Rekortman', 'AG Hat', 'OG/AG Trafo']):
    ax = axes.flatten()[i]

    mask = survival_df['Equipment_Class'] == equipment_class

    kmf = KaplanMeierFitter()
    kmf.fit(
        survival_df.loc[mask, 'Time_To_Event'],
        survival_df.loc[mask, 'Event_Occurred'],
        label=equipment_class
    )

    kmf.plot_survival_function(ax=ax)
    ax.set_title(f'Survival Curve: {equipment_class}')
    ax.set_xlabel('Days')
    ax.set_ylabel('Survival Probability')
    ax.axhline(0.5, ls='--', color='red', label='Median survival')

plt.tight_layout()
plt.savefig('outputs/survival_curves_by_class.png')
```

---

### **PHASE 3: Integrate Both Models** 🏆

**Create New File**: `07_integrated_risk_assessment.py`

```python
# Load predictions from both models
chronic_predictions = pd.read_csv('predictions/predictions_6m.csv')  # From Model 2
survival_predictions = pd.read_csv('predictions/survival_predictions.csv')  # From Model 1

# Merge
integrated = chronic_predictions.merge(
    survival_predictions,
    on='Equipment_ID',
    how='inner'
)

# Combined risk score
integrated['Temporal_Risk_6M'] = 1 - integrated['Survival_6M']  # Convert survival to risk
integrated['Chronic_Risk'] = integrated['PoF_Score']  # From Model 2

# Combined scoring (weighted average)
integrated['Combined_Risk_Score'] = (
    0.6 * integrated['Temporal_Risk_6M'] +  # Weight temporal higher
    0.4 * integrated['Chronic_Risk']
)

# Risk categories
def categorize_integrated_risk(row):
    if row['Temporal_Risk_6M'] > 0.7 and row['Chronic_Risk'] > 0.7:
        return 'IMMEDIATE_REPLACE'  # Both models agree - critical
    elif row['Temporal_Risk_6M'] > 0.7:
        return 'URGENT_MAINTENANCE'  # High temporal risk
    elif row['Chronic_Risk'] > 0.7:
        return 'SCHEDULE_REPLACEMENT'  # Chronic repeater
    elif row['Combined_Risk_Score'] > 0.5:
        return 'MONITOR_CLOSELY'
    else:
        return 'ROUTINE_MAINTENANCE'

integrated['Action_Category'] = integrated.apply(categorize_integrated_risk, axis=1)

# Save
integrated.to_csv('predictions/integrated_risk_assessment.csv', index=False)

# Summary report
print(f"\n🎯 Integrated Risk Assessment:")
print(integrated['Action_Category'].value_counts())
```

---

## 📅 IMPLEMENTATION TIMELINE

### **IMMEDIATE (Today/Tomorrow)**
1. ✅ Re-run 02 with fixed cause code detection
2. ✅ Run 03 → 04 → 05 → 05b (see cause code features impact)
3. ✅ Run 06 (Model 2: Chronic repeater - baseline)

### **SHORT TERM (2-3 days)**
4. 🎯 Create `06b_survival_analysis.py` (Model 1: Temporal PoF)
5. 🎯 Train survival models (Cox + RSF)
6. 🎯 Generate survival curves (Module 1 requirement)
7. 🎯 Validate: Different predictions for 3M/6M/12M/24M ✅

### **INTEGRATION (1 week)**
8. 🏆 Create `07_integrated_risk_assessment.py`
9. 🏆 Combine predictions from both models
10. 🏆 Generate final risk-ranked equipment list

---

## 🎯 EXPECTED FINAL OUTPUT

### **Equipment Risk Report** (Sample):

```
Equipment: 2037800
Class: Ayırıcı (12 years old)
Cause: mekanik arıza (75% consistency)
Location: Urban (250 customers affected)

MODEL 1: TEMPORAL POF (Survival Analysis)
├─ 3M failure risk:  48%  → Monitor
├─ 6M failure risk:  65%  → High risk
├─ 12M failure risk: 82%  → Critical
├─ 24M failure risk: 92%  → Very critical
└─ Median time-to-failure: 4.2 months

MODEL 2: CHRONIC REPEATER (Current Model)
├─ Chronic repeater score: 0.98 (98%)
├─ Lifetime failures: 5
├─ 90-day recurring: YES
└─ Classification: CHRONIC REPEATER

INTEGRATED ASSESSMENT:
├─ Combined risk score: 0.86 (CRITICAL)
├─ Action category: IMMEDIATE_REPLACE
├─ Recommendation: Schedule replacement within 3 months
└─ Priority rank: #3 out of 1,148 equipment
```

---

## ✅ BENEFITS OF DUAL-MODEL APPROACH

| Benefit | Model 1 (Survival) | Model 2 (Chronic) | Combined |
|---------|-------------------|-------------------|----------|
| **Temporal prediction** | ✅ Different for 3/6/12/24M | ❌ Static classification | ✅ Best of both |
| **Chronic identification** | ⚠️ Indirect (hazard ratio) | ✅ Direct classification | ✅ Best of both |
| **Maintenance scheduling** | ✅ Time-based priority | ❌ No timing info | ✅ Best of both |
| **Replacement decisions** | ⚠️ Based on time only | ✅ Based on pattern | ✅ Best of both |
| **Module 1 requirements** | ✅ Survival curves | ✅ Risk scores | ✅ Complete |

---

## 🚀 RECOMMENDATION: Proceed in Phases

**TODAY**:
- Re-run 02 (fixed version) → See cause code features
- Run 03 → 04 → 05 → 05b → 06
- Validate Model 2 (chronic repeater) with cause codes

**TOMORROW**:
- I create `06b_survival_analysis.py`
- You run survival analysis
- Compare: 3M ≠ 6M ≠ 12M ≠ 24M predictions ✅ (solves identical performance issue!)

**NEXT WEEK**:
- Integrate both models
- Final risk assessment reports

---

## ❓ DECISION NEEDED

Should I create `06b_survival_analysis.py` now, or wait until you finish running 02 → 06 with cause codes?

**My recommendation**:
1. First, re-run 02 (to see cause code features calculate correctly)
2. Then share output so I can verify cause codes working
3. Then I'll create survival analysis script while you continue pipeline

This way you can see cause code impact AND we implement temporal prediction properly! 🎯
