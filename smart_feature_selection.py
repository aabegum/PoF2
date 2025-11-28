"""
SMART FEATURE SELECTION PIPELINE
Turkish EDAŞ PoF Prediction Project (v6.0)

Purpose:
- Adaptive feature selection that handles dataset changes
- Auto-detect and remove problematic features
- No hardcoded feature lists - rule-based detection
- Comprehensive audit trail

Phases:
1. Auto-detect constants & near-constants
2. Auto-detect leakage patterns
3. Auto-detect high correlation pairs
4. Adaptive VIF optimization with domain protection

Input:  data/features_engineered.csv
Output: data/features_reduced.csv

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import sys

# Import column mapping
from column_mapping import (
    PROTECTED_FEATURES_EN,
    PROTECTED_FEATURES_TR,
    detect_leakage_pattern,
    is_protected_feature,
    rename_columns_to_turkish,
    categorize_feature,
    get_turkish_name,
    create_bilingual_report,
    COLUMN_MAP_EN_TO_TR
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

@dataclass
class SelectionConfig:
    """Configuration for smart feature selection"""
    # Thresholds
    constant_threshold: float = 0.001  # Variance < this = constant
    near_constant_unique_ratio: float = 0.01  # < 1% unique values
    correlation_threshold: float = 0.95  # Correlation > this = redundant
    vif_target: float = 10.0  # Target VIF threshold
    vif_max_iterations: int = 50  # Max VIF removal iterations
    min_coverage: float = 0.10  # Features with < 10% non-null are suspicious

    # Behavior
    remove_constants: bool = True
    remove_near_constants: bool = True
    remove_leaky: bool = True
    remove_high_correlation: bool = True
    apply_vif: bool = True
    standardize_names: bool = False  # Keep English names for model compatibility

    # Protected features (never remove) - use English for model compatibility
    protected_features: List[str] = field(default_factory=lambda: PROTECTED_FEATURES_EN.copy())


@dataclass
class FeatureReport:
    """Report for a single feature"""
    name: str
    original_name: str = None
    status: str = 'RETAINED'
    removal_reason: str = None
    phase_removed: int = None
    stats: Dict = field(default_factory=dict)


@dataclass
class SelectionReport:
    """Complete selection report"""
    original_count: int = 0
    final_count: int = 0
    removed_constant: int = 0
    removed_leaky: int = 0
    removed_correlation: int = 0
    removed_vif: int = 0
    features: List[FeatureReport] = field(default_factory=list)


# ============================================================================
# SMART FEATURE SELECTOR CLASS
# ============================================================================

class SmartFeatureSelector:
    """
    Adaptive feature selection pipeline

    Usage:
        selector = SmartFeatureSelector(config)
        df_selected = selector.fit_transform(df)
        report = selector.get_report()
    """

    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        self.report = SelectionReport()
        self.feature_reports = {}
        self.removed_features = set()
        self.id_column = None
        self.target_columns = []

    def fit_transform(self, df: pd.DataFrame,
                      id_column: str = 'Ekipman_ID',
                      target_columns: List[str] = None) -> pd.DataFrame:
        """
        Run complete feature selection pipeline

        Args:
            df: Input DataFrame
            id_column: ID column name (excluded from selection)
            target_columns: Target column names (excluded from selection)

        Returns:
            DataFrame with selected features
        """
        self.id_column = id_column
        self.target_columns = target_columns or []
        self.report.original_count = len(df.columns)

        print("="*100)
        print(" "*25 + "SMART FEATURE SELECTION PIPELINE")
        print(" "*20 + "Adaptive | Rule-Based | Dataset-Agnostic")
        print("="*100)

        # Initialize feature reports
        for col in df.columns:
            self.feature_reports[col] = FeatureReport(
                name=col,
                original_name=col
            )

        # Phase 0: Standardize column names to Turkish
        if self.config.standardize_names:
            df = self._phase0_standardize_names(df)

        # Phase 1: Remove constants and near-constants
        if self.config.remove_constants or self.config.remove_near_constants:
            df = self._phase1_remove_constants(df)

        # Phase 2: Remove leaky features
        if self.config.remove_leaky:
            df = self._phase2_remove_leakage(df)

        # Phase 3: Remove high correlation pairs
        if self.config.remove_high_correlation:
            df = self._phase3_remove_correlation(df)

        # Phase 4: VIF optimization
        if self.config.apply_vif:
            df = self._phase4_vif_optimization(df)

        # Final summary
        self.report.final_count = len(df.columns)
        self._print_summary()

        return df

    def _phase0_standardize_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 0: Optionally standardize column names to Turkish"""
        print("\n" + "="*100)
        print("PHASE 0: COLUMN NAME HANDLING")
        print("="*100)

        if not self.config.standardize_names:
            print("\n✓ Keeping English column names (model compatibility mode)")
            print("  → Turkish names available via display utilities")
            print("  → Use get_turkish_name() or create_bilingual_report() for Turkish output")

            # Show sample mappings for reference
            print(f"\n📋 Sample EN→TR mappings:")
            sample_cols = [col for col in df.columns if col in COLUMN_MAP_EN_TO_TR][:5]
            for col in sample_cols:
                print(f"  {col} → {get_turkish_name(col)}")
            return df

        # Original Turkish renaming logic (if enabled)
        renamed_count = 0
        rename_dict = {}

        for col in df.columns:
            if col in COLUMN_MAP_EN_TO_TR:
                new_name = COLUMN_MAP_EN_TO_TR[col]
                if new_name != col:
                    rename_dict[col] = new_name
                    renamed_count += 1
                    # Update feature report
                    if col in self.feature_reports:
                        self.feature_reports[col].original_name = col
                        # Create new report with Turkish name
                        self.feature_reports[new_name] = self.feature_reports.pop(col)
                        self.feature_reports[new_name].name = new_name

        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"\n✓ Renamed {renamed_count} columns to Turkish")
            print(f"\nSample renames:")
            for old, new in list(rename_dict.items())[:5]:
                print(f"  {old} → {new}")
            if len(rename_dict) > 5:
                print(f"  ... and {len(rename_dict) - 5} more")

        # Update ID column and target columns if renamed
        if self.id_column in rename_dict:
            self.id_column = rename_dict[self.id_column]
        self.target_columns = [rename_dict.get(t, t) for t in self.target_columns]

        # Update protected features list
        self.config.protected_features = [
            rename_dict.get(f, f) for f in self.config.protected_features
        ]

        return df

    def _phase1_remove_constants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 1: Remove constant and near-constant features"""
        print("\n" + "="*100)
        print("PHASE 1: AUTO-DETECT CONSTANTS & NEAR-CONSTANTS")
        print("="*100)

        features_to_check = [
            col for col in df.columns
            if col != self.id_column and col not in self.target_columns
        ]

        constant_features = []
        near_constant_features = []

        print(f"\n📊 Analyzing {len(features_to_check)} features...")

        for col in features_to_check:
            n_unique = df[col].nunique()
            n_total = len(df)
            unique_ratio = n_unique / n_total if n_total > 0 else 0
            variance = df[col].var() if df[col].dtype in ['float64', 'int64'] else None

            # Store stats
            if col in self.feature_reports:
                self.feature_reports[col].stats['unique_values'] = n_unique
                self.feature_reports[col].stats['unique_ratio'] = unique_ratio
                self.feature_reports[col].stats['variance'] = variance

            # Check for constant (single value)
            if n_unique <= 1:
                if not is_protected_feature(col):
                    constant_features.append(col)
                    self._mark_removed(col, 'CONSTANT', 1,
                                       f'Single unique value ({n_unique})')
                else:
                    print(f"  ⚠️  {col}: Constant but PROTECTED - keeping")
                continue

            # Check for near-constant (very low variance or few unique values)
            if self.config.remove_near_constants:
                is_near_constant = False

                # Low unique ratio
                if unique_ratio < self.config.near_constant_unique_ratio:
                    is_near_constant = True

                # Low variance for numeric
                if variance is not None and variance < self.config.constant_threshold:
                    is_near_constant = True

                if is_near_constant and not is_protected_feature(col):
                    near_constant_features.append(col)
                    if variance is not None:
                        reason = f'Low variance ({variance:.6f}) or few unique ({n_unique})'
                    else:
                        reason = f'Few unique values ({n_unique})'
                    self._mark_removed(col, 'NEAR_CONSTANT', 1, reason)

        # Remove identified features
        all_to_remove = constant_features + near_constant_features

        if constant_features:
            print(f"\n❌ Constant features ({len(constant_features)}):")
            for feat in constant_features:
                print(f"   • {feat}")

        if near_constant_features:
            print(f"\n❌ Near-constant features ({len(near_constant_features)}):")
            for feat in near_constant_features:
                print(f"   • {feat}")

        if all_to_remove:
            df = df.drop(columns=all_to_remove)
            self.report.removed_constant = len(all_to_remove)
            print(f"\n✓ Removed {len(all_to_remove)} constant/near-constant features")
        else:
            print("\n✓ No constant features detected")

        print(f"✓ Remaining: {len(df.columns)} features")
        return df

    def _phase2_remove_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 2: Auto-detect and remove leaky features"""
        print("\n" + "="*100)
        print("PHASE 2: AUTO-DETECT DATA LEAKAGE")
        print("="*100)

        features_to_check = [
            col for col in df.columns
            if col != self.id_column and col not in self.target_columns
        ]

        leaky_features = []

        print(f"\n🔍 Scanning {len(features_to_check)} features for leakage patterns...")

        for col in features_to_check:
            is_leaky, pattern_type, is_safe = detect_leakage_pattern(col)

            # PHASE 1.7 FIX (Hybrid Staged Selection - Stage 1: Strict Rules):
            # Removed: and not is_protected_feature(col) condition
            # Reason: PROTECTED_FEATURES override defeats purpose of statistical leakage detection
            # New approach: Apply statistical rules strictly, let domain experts review afterwards
            if is_leaky:
                leaky_features.append((col, pattern_type))
                self._mark_removed(col, 'LEAKAGE', 2,
                                   f'Leakage pattern detected: {pattern_type}')

        if leaky_features:
            print(f"\n❌ Leaky features detected ({len(leaky_features)}):")
            for feat, pattern in leaky_features:
                print(f"   • {feat} [{pattern}]")

            df = df.drop(columns=[f[0] for f in leaky_features])
            self.report.removed_leaky = len(leaky_features)
            print(f"\n✓ Removed {len(leaky_features)} leaky features")
        else:
            print("\n✓ No leakage patterns detected")

        print(f"✓ Remaining: {len(df.columns)} features")
        return df

    def _phase3_remove_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 3: Remove highly correlated features"""
        print("\n" + "="*100)
        print("PHASE 3: AUTO-DETECT HIGH CORRELATION")
        print("="*100)

        # Get numeric features only
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col != self.id_column and col not in self.target_columns
        ]

        if len(numeric_cols) < 2:
            print("\n✓ Insufficient numeric features for correlation analysis")
            return df

        print(f"\n📊 Computing correlation matrix for {len(numeric_cols)} numeric features...")

        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        features_to_remove = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]

                if corr_val > self.config.correlation_threshold:
                    high_corr_pairs.append((col_i, col_j, corr_val))

                    # PHASE 1.7 FIX (Hybrid Staged Selection - Stage 1: Strict Rules):
                    # Removed: is_protected_feature() checks in correlation removal
                    # Reason: Statistical rules should not be overridden by protection list
                    # New approach: Always remove based on data quality (coverage), not domain preferences
                    # Decision rule: Remove the feature with lower coverage
                    coverage_i = df[col_i].notna().mean()
                    coverage_j = df[col_j].notna().mean()
                    to_remove = col_j if coverage_i >= coverage_j else col_i

                    if to_remove not in features_to_remove:
                        features_to_remove.add(to_remove)
                        self._mark_removed(to_remove, 'HIGH_CORRELATION', 3,
                                           f'Corr={corr_val:.3f} with {col_i if to_remove == col_j else col_j}')

        if high_corr_pairs:
            print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated pairs (r > {self.config.correlation_threshold}):")
            for col_i, col_j, corr in high_corr_pairs[:10]:
                print(f"   • {col_i} ↔ {col_j}: r={corr:.3f}")
            if len(high_corr_pairs) > 10:
                print(f"   ... and {len(high_corr_pairs) - 10} more pairs")

        if features_to_remove:
            print(f"\n❌ Removing {len(features_to_remove)} redundant features:")
            for feat in list(features_to_remove)[:10]:
                print(f"   • {feat}")

            df = df.drop(columns=list(features_to_remove))
            self.report.removed_correlation = len(features_to_remove)
        else:
            print("\n✓ No highly correlated pairs detected")

        print(f"✓ Remaining: {len(df.columns)} features")
        return df

    def _phase4_vif_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 4: Iterative VIF-based multicollinearity reduction"""
        print("\n" + "="*100)
        print("PHASE 4: VIF OPTIMIZATION (Multicollinearity)")
        print("="*100)

        # Get features for VIF analysis
        exclude_cols = [self.id_column] + self.target_columns
        vif_candidates = [col for col in df.columns if col not in exclude_cols]

        # Separate numeric and categorical
        numeric_features = df[vif_candidates].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [col for col in vif_candidates if col not in numeric_features]

        print(f"\n📊 VIF Analysis Setup:")
        print(f"   Numeric features: {len(numeric_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        print(f"   Target VIF: < {self.config.vif_target}")

        if len(numeric_features) < 2:
            print("\n✓ Insufficient features for VIF analysis")
            return df

        # Prepare data for VIF
        df_vif = df[numeric_features + categorical_features].copy()

        # Encode categorical features
        for cat_col in categorical_features:
            if df_vif[cat_col].dtype == 'object' or df_vif[cat_col].dtype.name == 'category':
                le = LabelEncoder()
                df_vif[cat_col] = le.fit_transform(df_vif[cat_col].astype(str))

        # Handle missing values and infinities
        df_vif = df_vif.replace([np.inf, -np.inf], np.nan)
        for col in df_vif.columns:
            if df_vif[col].isnull().any():
                median_val = df_vif[col].median()
                df_vif[col] = df_vif[col].fillna(median_val if pd.notna(median_val) else 0)

        # Iterative VIF removal
        vif_features = df_vif.columns.tolist()
        iteration = 0
        removed_vif = []

        print(f"\n--- Iterative VIF Calculation ---")

        while iteration < self.config.vif_max_iterations:
            iteration += 1

            if len(vif_features) < 2:
                break

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data['Feature'] = vif_features
            try:
                vif_data['VIF'] = [
                    variance_inflation_factor(df_vif[vif_features].values, i)
                    for i in range(len(vif_features))
                ]
            except Exception as e:
                print(f"   ⚠️  VIF calculation error: {e}")
                break

            # Find max VIF
            max_vif = vif_data['VIF'].max()
            max_vif_idx = vif_data['VIF'].idxmax()
            max_vif_feature = vif_data.loc[max_vif_idx, 'Feature']

            # Check stopping conditions
            if max_vif <= self.config.vif_target:
                print(f"\n✅ Iteration {iteration}: All VIF ≤ {self.config.vif_target}")
                break

            if max_vif <= 15 and len(vif_features) <= 20:
                print(f"\n✅ Iteration {iteration}: VIF={max_vif:.1f} acceptable with {len(vif_features)} features")
                break

            # PHASE 1.7 FIX (Hybrid Staged Selection - Stage 1: Strict Rules):
            # Removed: is_protected_feature() override logic in VIF removal
            # OLD logic: Skip protected features with high VIF, remove OTHER features instead
            # Problem: Algorithm fought mathematics (VIF=∞ features stayed, others removed)
            # NEW logic: Remove the feature with highest VIF regardless of protection status
            # Impact: Clean feature sets without mathematical fighting

            # No longer checking protection - apply rule strictly
            # (Stage 2 domain review will happen after selection)

            # Remove feature
            print(f"   Iter {iteration}: Removing {max_vif_feature} (VIF={max_vif:.1f})")
            vif_features.remove(max_vif_feature)
            df_vif = df_vif[vif_features]
            removed_vif.append(max_vif_feature)

            self._mark_removed(max_vif_feature, 'HIGH_VIF', 4,
                               f'VIF={max_vif:.1f} > {self.config.vif_target}')

        # Print final VIF
        if len(vif_features) >= 2:
            print(f"\n--- Final VIF Results ({len(vif_features)} features) ---")
            final_vif = pd.DataFrame()
            final_vif['Feature'] = vif_features
            final_vif['VIF'] = [
                variance_inflation_factor(df_vif[vif_features].values, i)
                for i in range(len(vif_features))
            ]
            final_vif = final_vif.sort_values('VIF', ascending=False)
            print(final_vif.head(15).to_string(index=False))

        # Keep only selected features
        if removed_vif:
            self.report.removed_vif = len(removed_vif)
            # Build final column list
            final_cols = [self.id_column] if self.id_column in df.columns else []
            final_cols += [t for t in self.target_columns if t in df.columns]
            final_cols += vif_features
            df = df[[col for col in final_cols if col in df.columns]]

            print(f"\n✓ Removed {len(removed_vif)} high-VIF features")

        print(f"✓ Final: {len(df.columns)} features")
        return df

    def _mark_removed(self, feature: str, reason: str, phase: int, detail: str):
        """Mark a feature as removed"""
        self.removed_features.add(feature)
        if feature in self.feature_reports:
            self.feature_reports[feature].status = f'REMOVED_{reason}'
            self.feature_reports[feature].removal_reason = detail
            self.feature_reports[feature].phase_removed = phase

    def _print_summary(self):
        """Print final selection summary"""
        print("\n" + "="*100)
        print("SMART FEATURE SELECTION COMPLETE")
        print("="*100)

        print(f"\n📊 SELECTION SUMMARY:")
        print(f"   Original features: {self.report.original_count}")
        print(f"   Removed (constant): {self.report.removed_constant}")
        print(f"   Removed (leakage): {self.report.removed_leaky}")
        print(f"   Removed (correlation): {self.report.removed_correlation}")
        print(f"   Removed (VIF): {self.report.removed_vif}")
        print(f"   Final features: {self.report.final_count}")

        total_removed = (self.report.removed_constant + self.report.removed_leaky +
                         self.report.removed_correlation + self.report.removed_vif)
        reduction = (1 - self.report.final_count / self.report.original_count) * 100
        print(f"\n   Total removed: {total_removed}")
        print(f"   Reduction: {reduction:.1f}%")

    def get_report(self) -> pd.DataFrame:
        """Get detailed feature selection report as DataFrame"""
        report_data = []
        for name, report in self.feature_reports.items():
            report_data.append({
                'Feature': name,
                'Original_Name': report.original_name,
                'Status': report.status,
                'Removal_Reason': report.removal_reason,
                'Phase_Removed': report.phase_removed,
                'Category': categorize_feature(name),
            })
        return pd.DataFrame(report_data)

    def get_retained_features(self) -> List[str]:
        """Get list of retained feature names"""
        return [
            name for name, report in self.feature_reports.items()
            if report.status == 'RETAINED'
        ]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_smart_selection(
    input_file: str = 'data/features_engineered.csv',
    output_file: str = 'data/features_reduced.csv',
    report_file: str = 'outputs/feature_selection/smart_selection_report.csv',
    config: SelectionConfig = None
) -> pd.DataFrame:
    """
    Run the smart feature selection pipeline

    Args:
        input_file: Path to input features CSV
        output_file: Path to save selected features
        report_file: Path to save selection report
        config: SelectionConfig instance

    Returns:
        DataFrame with selected features
    """
    config = config or SelectionConfig()

    # Load data
    print(f"\n📂 Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded: {len(df):,} records × {len(df.columns)} features")

    # Detect target columns
    target_cols = [col for col in df.columns if col.startswith('Target_') or col.startswith('Hedef_')]

    # Run selection
    selector = SmartFeatureSelector(config)
    df_selected = selector.fit_transform(
        df,
        id_column='Ekipman_ID',
        target_columns=target_cols
    )

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved selected features: {output_file}")

    # Save selection report
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    report_df = selector.get_report()
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"💾 Saved selection report: {report_file}")

    # Save bilingual column reference (EN/TR)
    bilingual_path = Path(report_file).parent / 'bilingual_column_reference.csv'
    bilingual_df = create_bilingual_report(df_selected, bilingual_path)
    print(f"💾 Saved bilingual reference: {bilingual_path}")

    return df_selected


if __name__ == '__main__':
    # Run with default config
    run_smart_selection()
