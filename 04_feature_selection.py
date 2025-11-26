"""
FEATURE SELECTION PIPELINE (v6.0 - Smart Selection)
Turkish EDAŞ PoF Prediction Project

Purpose:
- Use smart adaptive feature selection
- Auto-detect problematic features (constants, leakage, correlation, VIF)
- Standardize column names to Turkish
- Generate comprehensive audit trail

Input:  data/features_engineered.csv
Output: data/features_reduced.csv

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Import smart feature selection
from smart_feature_selection import SmartFeatureSelector, SelectionConfig, run_smart_selection

# Import column mapping for EN/TR compatibility
from column_mapping import (
    PROTECTED_FEATURES_EN,
    PROTECTED_FEATURES_TR,
    rename_columns_to_turkish,
    get_turkish_name,
    create_bilingual_report,
    COLUMN_MAP_EN_TO_TR
)

# Import centralized configuration
from config import (
    FEATURES_ENGINEERED_FILE,
    FEATURES_REDUCED_FILE,
    OUTPUT_DIR,
    VIF_THRESHOLD,
    VIF_TARGET,
    CORRELATION_THRESHOLD,
)

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Create output directory
output_dir = OUTPUT_DIR / 'feature_selection'
output_dir.mkdir(parents=True, exist_ok=True)

# Configure smart selection
# NOTE: Models use ENGLISH column names internally for compatibility
#       Turkish names are available via display utilities for reports/UI
selection_config = SelectionConfig(
    # Thresholds (from config.py)
    constant_threshold=0.001,
    near_constant_unique_ratio=0.01,
    correlation_threshold=CORRELATION_THRESHOLD,
    vif_target=VIF_TARGET,
    vif_max_iterations=50,
    min_coverage=0.10,

    # Behavior
    remove_constants=True,
    remove_near_constants=True,
    remove_leaky=True,
    remove_high_correlation=True,
    apply_vif=True,
    standardize_names=False,  # Keep English for model compatibility

    # Protected features (English names for model compatibility)
    protected_features=PROTECTED_FEATURES_EN.copy()
)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*100)
    print(" "*25 + "FEATURE SELECTION PIPELINE v6.0")
    print(" "*20 + "Smart Selection | Turkish Naming | Adaptive")
    print("="*100)

    print("\n📋 Configuration:")
    print(f"   Input: {FEATURES_ENGINEERED_FILE}")
    print(f"   Output: {FEATURES_REDUCED_FILE}")
    print(f"   VIF Target: {VIF_TARGET}")
    print(f"   Correlation Threshold: {CORRELATION_THRESHOLD}")
    print(f"   Protected Features: {len(PROTECTED_FEATURES_EN)}")
    print(f"   Column Names: English (internal) + Turkish (display)")

    # ========================================================================
    # STEP 0: LOAD DATA
    # ========================================================================
    print("\n" + "="*100)
    print("STEP 0: LOADING ENGINEERED FEATURES")
    print("="*100)

    if not FEATURES_ENGINEERED_FILE.exists():
        print(f"\n❌ ERROR: File not found at {FEATURES_ENGINEERED_FILE}")
        print("Please run 03_feature_engineering.py first!")
        exit(1)

    print(f"\n✓ Loading from: {FEATURES_ENGINEERED_FILE}")
    df = pd.read_csv(FEATURES_ENGINEERED_FILE)
    print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

    original_columns = df.columns.tolist()
    original_count = len(original_columns)

    # ========================================================================
    # STEP 1: RUN SMART FEATURE SELECTION
    # ========================================================================
    # The smart selector handles:
    # - Phase 0: Column name standardization (EN → TR)
    # - Phase 1: Constant & near-constant removal
    # - Phase 2: Leakage pattern detection
    # - Phase 3: High correlation removal
    # - Phase 4: VIF optimization

    # Detect target columns
    target_cols = [col for col in df.columns if 'Target' in col or 'Hedef' in col]
    print(f"\n🎯 Detected target columns: {target_cols}")

    # Run smart selection
    selector = SmartFeatureSelector(selection_config)
    df_selected = selector.fit_transform(
        df,
        id_column='Ekipman_ID',
        target_columns=target_cols
    )

    # ========================================================================
    # STEP 2: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*100)
    print("STEP 2: SAVING RESULTS")
    print("="*100)

    # Save selected features
    print(f"\n💾 Saving to: {FEATURES_REDUCED_FILE}")
    df_selected.to_csv(FEATURES_REDUCED_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ Successfully saved!")
    print(f"   Records: {len(df_selected):,}")
    print(f"   Features: {len(df_selected.columns)}")

    # Save comprehensive report
    report_path = output_dir / 'smart_selection_report.csv'
    report_df = selector.get_report()
    # Add Turkish names to report
    report_df['Turkish_Name'] = report_df['Feature'].apply(get_turkish_name)
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n📋 Selection report saved: {report_path}")

    # Save bilingual column reference (EN ↔ TR)
    bilingual_path = output_dir / 'bilingual_column_reference.csv'
    bilingual_df = create_bilingual_report(df_selected, bilingual_path)
    print(f"📋 Bilingual reference saved: {bilingual_path}")

    # ========================================================================
    # STEP 3: FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("FEATURE SELECTION COMPLETE")
    print("="*100)

    # Count by status
    status_counts = report_df['Status'].value_counts()
    print(f"\n📊 Feature Status Summary:")
    for status, count in status_counts.items():
        icon = "✓" if status == 'RETAINED' else "❌"
        print(f"   {icon} {status}: {count}")

    # List retained features by category (show both EN and TR names)
    retained = report_df[report_df['Status'] == 'RETAINED']
    print(f"\n✅ RETAINED FEATURES ({len(retained)}):")
    print(f"\n   {'English Name':<40} {'Turkish Name':<40}")
    print(f"   {'-'*40} {'-'*40}")

    # Group by category
    for category in retained['Category'].unique():
        cat_features = retained[retained['Category'] == category]
        print(f"\n   📁 {category}:")
        for _, row in cat_features.iterrows():
            en_name = row['Feature']
            tr_name = row['Turkish_Name']
            print(f"      {en_name:<38} {tr_name:<38}")

    print(f"\n📂 OUTPUT FILES:")
    print(f"   • {FEATURES_REDUCED_FILE} (English column names)")
    print(f"   • {report_path}")
    print(f"   • {bilingual_path}")

    print(f"\n💡 BENEFITS OF SMART SELECTION:")
    print(f"   ✓ Adaptive - handles dataset changes automatically")
    print(f"   ✓ No hardcoded lists - rule-based detection")
    print(f"   ✓ Bilingual support - EN internal, TR for display")
    print(f"   ✓ Audit trail - comprehensive selection report")
    print(f"   ✓ Domain protection - critical features preserved")

    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Run model training: python 06_temporal_pof_model.py")
    print(f"   2. Review selection report for audit")
    print(f"   3. Check bilingual_column_reference.csv for EN↔TR mapping")

    print(f"\n📝 USAGE NOTE:")
    print(f"   Models use English column names internally.")
    print(f"   For Turkish display in reports, use:")
    print(f"     from column_mapping import get_turkish_name, create_turkish_display_df")

    print("\n" + "="*100)
    print(f"{'FEATURE SELECTION PIPELINE COMPLETE':^100}")
    print("="*100)
