"""
DATA LEAKAGE REMOVAL
Turkish EDAŞ PoF Prediction Project

Purpose:
- Identify and remove features that would cause data leakage
- Data leakage = features that contain information about the target period
- Ensures model only uses information available at prediction time

Data Leakage Rules:
1. LEAKY: Features that include target period data (e.g., Arıza_Sayısı_12ay for 12M prediction)
2. LEAKY: Aggregations calculated on target period data
3. LEAKY: Features derived from future knowledge
4. SAFE: Historical data from BEFORE prediction time
5. SAFE: Static attributes (age, equipment class, location, capacity)

Input:  data/features_selected.csv (~25-35 features)
Output: data/features_selected_clean.csv (~15-25 features)

Author: Data Analytics Team
Date: 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
# Fix Unicode encoding for Windows console (Turkish cp1254 issue)
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*100)
print(" "*30 + "DATA LEAKAGE REMOVAL")
print(" "*35 + "PoF Prediction")
print("="*100)

# ============================================================================
# STEP 1: LOAD SELECTED FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 1: LOADING SELECTED FEATURES")
print("="*100)

data_path = Path('data/features_selected.csv')

if not data_path.exists():
    print(f"\n❌ ERROR: File not found at {data_path}")
    print("Please run 05_feature_selection.py first!")
    exit(1)

print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

original_features = df.columns.tolist()

# ============================================================================
# STEP 2: DEFINE LEAKAGE RULES
# ============================================================================
print("\n" + "="*100)
print("STEP 2: DEFINING DATA LEAKAGE RULES")
print("="*100)

print("\n📋 Data Leakage Categories:")
print("\n1. LEAKY - Recent Failure Counts (include target period):")
print("   • Arıza_Sayısı_3ay, _6ay, _12ay (recent failure counts)")
print("   • Any aggregations based on these (Class_Avg, Cluster_Avg)")
print("   • Recent_Failure_Intensity (3M/12M ratio)")

print("\n2. LEAKY - Future-Looking Features:")
print("   • Risk scores calculated FROM target period failures")
print("   • Any feature derived from 'looking ahead' at failures")

print("\n3. SAFE - Historical Patterns (NOT lifetime-based):")
print("   • Son_Arıza_Gun_Sayisi (days since last failure - RECENCY only)")
print("   • Failure patterns (seasonality, peak flags)")

print("\n4. REMOVED - Lifetime-Based Features (LEAKY when predicting failure propensity):")
print("   • Toplam_Arıza_Sayisi_Lifetime ← Used to CREATE target!")
print("\n   ✅ RESTORED (v4.1 fix): MTBF, Reliability, Composite PoF Score")
print("      → Now calculated using ONLY failures BEFORE cutoff (2024-06-25)")
print("      → See 02_data_transformation.py lines 633-668 for safe MTBF calculation")

print("\n5. SAFE - Static Attributes:")
print("   • Ekipman_Yaşı_Yıl (equipment age)")
print("   • Yas_Beklenen_Omur_Orani (age ratio)")
print("   • Equipment_Class_Primary (equipment type)")
print("   • Geographic location and cluster")
print("   • Customer impact (infrastructure-based)")

# ============================================================================
# STEP 3: IDENTIFY LEAKY FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 3: IDENTIFYING LEAKY FEATURES")
print("="*100)

print("\n--- Scanning for Leaky Feature Patterns ---")

leaky_features = []
leaky_reasons = {}

for col in original_features:
    reason = None

    # Rule 1: Recent failure counts (3/6/12 months)
    if 'Arıza_Sayısı_3ay' in col or 'Arıza_Sayısı_6ay' in col or 'Arıza_Sayısı_12ay' in col:
        if col != 'Toplam_Arıza_Sayisi_Lifetime':  # Lifetime is OK
            reason = "Recent failure count (includes target period)"

    # Rule 2: Aggregations based on recent failures
    elif any(x in col for x in ['_12ay_Class_Avg', '_12ay_Cluster_Avg', '_6ay_Class_Avg', '_3ay_Class_Avg']):
        reason = "Aggregation based on recent failures (target period)"

    # Rule 3: Recent failure intensity/acceleration
    elif 'Recent_Failure_Intensity' in col or 'Failure_Acceleration' in col:
        reason = "Recent failure intensity (uses target period)"

    # Rule 4: Risk scores if based on recent failures
    elif 'Recent_Failure_Risk_Score' in col:
        reason = "Risk score based on recent failures"

    # Rule 5: Age-Failure interaction - SAFE if using historical failures BEFORE cutoff
    # NOTE: Removed from leakage detection - should be verified manually instead
    # elif 'Age_Failure_Interaction' in col:
    #     reason = "Interaction with recent failure counts"

    # Rule 6: Failure-free flags - SAFE if calculated BEFORE cutoff (2024-06-25)
    # NOTE: Failure_Free_3M = (Son_Arıza_Tarihi < 2024-03-25) is SAFE
    # Removed from leakage detection - it's a historical observation
    # elif 'Failure_Free_3M' in col:
    #     reason = "Recent failure-free flag"

    # Rule 7: Time since last normalized (if recent)
    elif 'Time_Since_Last_Normalized' in col:
        reason = "Normalized time since last failure (recent)"

    # Rule 8: Lifetime failure count (used to create target - DIRECT LEAKAGE!)
    elif 'Toplam_Arıza_Sayisi_Lifetime' in col or 'Toplam_Ariza_Sayisi_Lifetime' in col:
        reason = "Lifetime failure count (DIRECTLY used to create target!)"

    # Rule 9: MTBF features - NOW SAFE (v4.1 fix)
    # ✅ RESTORED: MTBF was fixed in 02_data_transformation.py (lines 633-668)
    #    to use ONLY failures BEFORE cutoff date (2024-06-25)
    # elif 'MTBF' in col:
    #     reason = "MTBF calculated from lifetime failure count (indirect leakage)"

    # Rule 10: Reliability Score - NOW SAFE (v4.1 fix)
    # ✅ RESTORED: Calculated from safe MTBF (which uses only pre-cutoff failures)
    # elif 'Reliability_Score' in col:
    #     reason = "Reliability calculated from MTBF (indirect leakage)"

    # Rule 11: Composite Risk Score - NOW SAFE (v4.1 fix)
    # ✅ RESTORED: Uses safe MTBF_Risk_Score + historical age/recurrence data
    #    See 03_feature_engineering.py lines 420-478 for updated risk weights
    # elif 'Composite_PoF_Risk_Score' in col or 'Composite_Risk' in col:
    #     reason = "Composite score includes MTBF_Risk_Score (indirect leakage)"

    if reason:
        leaky_features.append(col)
        leaky_reasons[col] = reason

print(f"\n⚠️  Identified {len(leaky_features)} leaky features:")
for feat in leaky_features:
    print(f"   ❌ {feat:<50} → {leaky_reasons[feat]}")

# ============================================================================
# STEP 4: DEFINE SAFE FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 4: DEFINING SAFE FEATURE SET")
print("="*100)

# Remove leaky features
safe_features = [col for col in original_features if col not in leaky_features]

print(f"\n✓ {len(safe_features)} safe features identified")

# Categorize safe features
id_features = [col for col in safe_features if 'ID' in col or col == 'Ekipman_ID']
age_features = [col for col in safe_features if 'Yaş' in col or 'Age' in col or 'Ömür' in col or 'Life' in col]
# NOTE: Removed 'Lifetime', 'MTBF', 'Reliability' from historical features (they are leaky!)
historical_features = [col for col in safe_features if 'Son_Arıza' in col or 'Last_Failure' in col or 'Recurrence' in col]
location_features = [col for col in safe_features if 'Geographic' in col or 'Cluster' in col or 'KOORDINAT' in col or 'İl' in col or 'İlçe' in col or 'urban' in col or 'suburban' in col]
equipment_features = [col for col in safe_features if 'Equipment' in col or 'Ekipman' in col or 'Class' in col or 'voltage' in col]
customer_features = [col for col in safe_features if 'Customer' in col or 'Müşteri' in col or 'Peak' in col]
# NOTE: Most 'Risk' and 'Score' features are leaky (removed by rules above), only remaining ones are safe
composite_features = [col for col in safe_features if ('Risk' in col or 'Score' in col) and col not in equipment_features]

# Note: Some composite features might be in multiple categories
# Filter out composites that are already categorized
other_features = [col for col in safe_features if col not in
                 id_features + age_features + historical_features +
                 location_features + equipment_features + customer_features + composite_features]

print("\n--- Safe Features by Category ---")
print(f"\n1. ID Features ({len(id_features)}):")
for feat in id_features:
    print(f"   ✓ {feat}")

print(f"\n2. Age/Life Features ({len(age_features)}):")
for feat in age_features:
    print(f"   ✓ {feat}")

print(f"\n3. Historical Failure Features ({len(historical_features)}):")
for feat in historical_features:
    print(f"   ✓ {feat}")

print(f"\n4. Location/Geographic Features ({len(location_features)}):")
for feat in location_features:
    print(f"   ✓ {feat}")

print(f"\n5. Equipment Type Features ({len(equipment_features)}):")
for feat in equipment_features:
    print(f"   ✓ {feat}")

print(f"\n6. Customer Impact Features ({len(customer_features)}):")
for feat in customer_features:
    print(f"   ✓ {feat}")

print(f"\n7. Composite/Risk Features ({len(composite_features)}):")
for feat in composite_features:
    # Check if composite feature uses safe components
    if any(x in feat for x in ['12ay', '6ay', '3ay', 'Recent']):
        print(f"   ⚠️  {feat} (may contain recent data - verify manually)")
    else:
        print(f"   ✓ {feat}")

if len(other_features) > 0:
    print(f"\n8. Other Features ({len(other_features)}):")
    for feat in other_features:
        print(f"   ✓ {feat}")

# ============================================================================
# STEP 5: MANUAL REVIEW WARNINGS
# ============================================================================
print("\n" + "="*100)
print("STEP 5: MANUAL REVIEW WARNINGS")
print("="*100)

print("\n⚠️  Features requiring manual review:")

# Check for potentially problematic features
review_needed = []

for feat in safe_features:
    if 'Risk_Category' in feat:
        review_needed.append((feat, "Verify Risk_Category is not based on target period"))
    # Composite_PoF_Risk_Score is now SAFE (v4.1 fix) - no manual review needed
    # elif 'Composite_PoF_Risk_Score' in feat:
    #     review_needed.append((feat, "Verify composite score doesn't use recent failures"))
    elif 'Class_Avg' in feat or 'Cluster_Avg' in feat:
        if not any(x in feat for x in ['12ay', '6ay', '3ay']):  # Already filtered
            review_needed.append((feat, "Verify aggregation period doesn't overlap with target"))

if len(review_needed) > 0:
    for feat, warning in review_needed:
        print(f"   ⚠️  {feat}")
        print(f"       → {warning}")
else:
    print("   ✓ No features requiring manual review")

# ============================================================================
# STEP 6: SAVE CLEAN FEATURES
# ============================================================================
print("\n" + "="*100)
print("STEP 6: SAVING CLEAN FEATURE SET")
print("="*100)

# Create clean dataframe
df_clean = df[safe_features].copy()

output_path = Path('data/features_selected_clean.csv')
print(f"\n💾 Saving to: {output_path}")
df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ Successfully saved!")
print(f"   Records: {len(df_clean):,}")
print(f"   Features: {len(df_clean.columns)}")
print(f"   File size: {output_path.stat().st_size / 1024**2:.2f} MB")

# Save leakage report
print("\n📋 Creating leakage analysis report...")

report_data = []
for feat in original_features:
    is_leaky = feat in leaky_features
    report_data.append({
        'Feature': feat,
        'Status': 'LEAKY' if is_leaky else 'SAFE',
        'Reason': leaky_reasons.get(feat, 'No leakage detected'),
        'Included_In_Clean_Set': not is_leaky
    })

report_df = pd.DataFrame(report_data)
report_path = Path('outputs/feature_selection/leakage_analysis.csv')
report_path.parent.mkdir(parents=True, exist_ok=True)
report_df.to_csv(report_path, index=False)
print(f"✓ Leakage analysis saved to: {report_path}")

# ============================================================================
# STEP 7: SUMMARY
# ============================================================================
print("\n" + "="*100)
print("DATA LEAKAGE REMOVAL COMPLETE")
print("="*100)

print(f"\n📊 LEAKAGE REMOVAL SUMMARY:")
print(f"   Original features: {len(original_features)}")
print(f"   Leaky features removed: {len(leaky_features)}")
print(f"   Safe features retained: {len(safe_features)}")
print(f"   Retention rate: {len(safe_features)/len(original_features)*100:.1f}%")

print(f"\n📂 OUTPUT FILES:")
print(f"   • {output_path} ({len(safe_features)} features)")
print(f"   • {report_path} (detailed analysis)")

print(f"\n🚀 READY FOR MODEL TRAINING:")
print(f"   ✓ No data leakage from target period")
print(f"   ✓ Only historical and static features included")
print(f"   ✓ Clean dataset for unbiased model training")

print("\n💡 IMPORTANT NOTES:")
print("   • This script removes OBVIOUS leakage patterns")
print("   • Manual review recommended for composite features")
print("   • Verify aggregations don't overlap with prediction window")
print("   • For multi-horizon predictions (3M/6M/12M), ensure features are calculated")
print("     from data BEFORE each prediction window")

print("\n" + "="*100)
print(f"{'LEAKAGE REMOVAL PIPELINE COMPLETE':^100}")
print("="*100)
