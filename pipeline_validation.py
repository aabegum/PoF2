"""
PIPELINE VALIDATION MODULE
Turkish EDAŞ PoF Prediction Pipeline

This module provides validation functions to ensure data integrity
between pipeline steps. Catches errors early before they cascade.

Usage:
    from pipeline_validation import validate_step_output, ValidationError

    try:
        validate_step_output(step=2, check_files=True)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

Author: Data Analytics Team
Date: 2025-11-20
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Import configuration
from config import (
    EQUIPMENT_LEVEL_FILE,
    FEATURES_ENGINEERED_FILE,
    FEATURES_REDUCED_FILE,
    PREDICTION_DIR,
    RESULTS_DIR,
    HORIZONS,
    PROTECTED_FEATURES,
    MIN_EQUIPMENT_RECORDS,
    MAX_EQUIPMENT_RECORDS,
    MIN_PREDICTIONS,
    MAX_PREDICTIONS,
    MIN_FEATURES,
    MAX_FEATURES
)


class ValidationError(Exception):
    """Custom exception for pipeline validation failures."""
    pass


# ============================================================================
# STEP-SPECIFIC VALIDATION SCHEMAS
# ============================================================================

STEP_VALIDATIONS = {
    1: {
        'name': 'Data Profiling',
        'outputs': [],  # Profiling doesn't create files, just validates
        'checks': []
    },
    2: {
        'name': 'Data Transformation',
        'outputs': [EQUIPMENT_LEVEL_FILE],
        'checks': [
            {'file': EQUIPMENT_LEVEL_FILE, 'min_rows': MIN_EQUIPMENT_RECORDS, 'max_rows': MAX_EQUIPMENT_RECORDS,
             'required_columns': ['Ekipman_ID', 'Ekipman_Yaşı_Yıl', 'Toplam_Arıza_Sayisi_Lifetime']}
        ]
    },
    3: {
        'name': 'Feature Engineering',
        'outputs': [FEATURES_ENGINEERED_FILE],
        'checks': [
            {'file': FEATURES_ENGINEERED_FILE, 'min_rows': MIN_EQUIPMENT_RECORDS, 'max_rows': MAX_EQUIPMENT_RECORDS,
             'min_features': 50, 'max_features': 120,  # Dynamic: 50-120 features depending on input schema
             'required_columns': ['Ekipman_ID', 'Ekipman_Yaşı_Yıl']}  # Equipment_Class_Primary may be missing
        ]
    },
    4: {
        'name': 'Feature Selection',
        'outputs': [FEATURES_REDUCED_FILE],
        'checks': [
            {'file': FEATURES_REDUCED_FILE, 'min_rows': MIN_EQUIPMENT_RECORDS, 'max_rows': MAX_EQUIPMENT_RECORDS,
             'min_features': MIN_FEATURES, 'max_features': MAX_FEATURES,
             'no_leakage': True}  # Ensure no future-looking features
        ]
    },
    5: {
        'name': 'Temporal PoF Model',
        'outputs': [
            PREDICTION_DIR / 'predictions_3m.csv',
            PREDICTION_DIR / 'predictions_6m.csv',
            PREDICTION_DIR / 'predictions_12m.csv'
        ],
        'checks': [
            {'file': PREDICTION_DIR / 'predictions_3m.csv', 'min_rows': MIN_EQUIPMENT_RECORDS,
             'required_columns': ['Ekipman_ID', 'PoF_Probability', 'Risk_Class']},
            {'file': PREDICTION_DIR / 'predictions_6m.csv', 'min_rows': MIN_EQUIPMENT_RECORDS,
             'required_columns': ['Ekipman_ID', 'PoF_Probability', 'Risk_Class']},
            {'file': PREDICTION_DIR / 'predictions_12m.csv', 'min_rows': MIN_EQUIPMENT_RECORDS,
             'required_columns': ['Ekipman_ID', 'PoF_Probability', 'Risk_Class']}
        ]
    },
    6: {
        'name': 'Chronic Classifier',
        'outputs': [PREDICTION_DIR / 'chronic_repeaters.csv'],
        'checks': [
            {'file': PREDICTION_DIR / 'chronic_repeaters.csv', 'min_rows': MIN_EQUIPMENT_RECORDS,
             'required_columns': ['Ekipman_ID', 'Chronic_Probability', 'Chronic_Class']}
        ]
    },
    7: {
        'name': 'Model Explainability',
        'outputs': [],  # Creates visualizations, not CSV files
        'checks': []
    },
    8: {
        'name': 'Probability Calibration',
        'outputs': [],  # Updates models in place
        'checks': []
    },
    9: {
        'name': 'Survival Analysis',
        'outputs': [PREDICTION_DIR / 'pof_multi_horizon_predictions.csv'],
        'checks': [
            {'file': PREDICTION_DIR / 'pof_multi_horizon_predictions.csv', 'min_rows': MIN_EQUIPMENT_RECORDS,
             'required_columns': ['Ekipman_ID'] + [f'PoF_Probability_{h}' for h in HORIZONS.keys()]}
        ]
    },
    10: {
        'name': 'Risk Assessment',
        'outputs': [
            RESULTS_DIR / 'risk_assessment_3M.csv',
            RESULTS_DIR / 'risk_assessment_6M.csv',
            RESULTS_DIR / 'risk_assessment_12M.csv',
            RESULTS_DIR / 'capex_priority_list.csv'
        ],
        'checks': [
            {'file': RESULTS_DIR / 'risk_assessment_3M.csv', 'min_rows': MIN_EQUIPMENT_RECORDS,
             'required_columns': ['Ekipman_ID', 'PoF_Probability', 'CoF_Score', 'Risk_Score', 'Risk_Category']},
            {'file': RESULTS_DIR / 'capex_priority_list.csv', 'min_rows': MIN_PREDICTIONS, 'max_rows': MAX_PREDICTIONS,
             'required_columns': ['Ekipman_ID', 'Risk_Score']}  # Equipment_Class_Primary may be missing
        ]
    }
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_file_exists(file_path: Path) -> None:
    """Validate that a file exists."""
    if not file_path.exists():
        raise ValidationError(f"Required file not found: {file_path}")


def validate_dataframe_shape(df: pd.DataFrame, file_path: Path,
                             min_rows: Optional[int] = None,
                             max_rows: Optional[int] = None,
                             min_features: Optional[int] = None,
                             max_features: Optional[int] = None,
                             expected_features: Optional[int] = None) -> None:
    """Validate DataFrame dimensions."""
    rows, cols = df.shape

    if min_rows and rows < min_rows:
        raise ValidationError(f"{file_path.name}: Too few rows ({rows} < {min_rows})")

    if max_rows and rows > max_rows:
        raise ValidationError(f"{file_path.name}: Too many rows ({rows} > {max_rows})")

    if expected_features and cols != expected_features:
        raise ValidationError(
            f"{file_path.name}: Expected {expected_features} features, got {cols}"
        )

    if min_features and cols < min_features:
        raise ValidationError(f"{file_path.name}: Too few features ({cols} < {min_features})")

    if max_features and cols > max_features:
        raise ValidationError(f"{file_path.name}: Too many features ({cols} > {max_features})")


def validate_required_columns(df: pd.DataFrame, file_path: Path,
                              required_columns: List[str]) -> None:
    """Validate that required columns exist."""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValidationError(
            f"{file_path.name}: Missing required columns: {', '.join(missing_cols)}"
        )


def validate_no_leakage(df: pd.DataFrame, file_path: Path) -> None:
    """Validate that no data leakage features are present."""
    # Features that would indicate data leakage
    leakage_patterns = [
        'Future', 'After_Cutoff', 'Prediction_Period',
        'Post_2024', '_Leak', 'Target_'
    ]

    leaky_cols = []
    for col in df.columns:
        if any(pattern.lower() in col.lower() for pattern in leakage_patterns):
            leaky_cols.append(col)

    if leaky_cols:
        raise ValidationError(
            f"{file_path.name}: Potential data leakage in columns: {', '.join(leaky_cols)}"
        )


def validate_no_nulls_in_key_columns(df: pd.DataFrame, file_path: Path,
                                     key_columns: List[str]) -> None:
    """Validate that key columns have no null values."""
    for col in key_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                raise ValidationError(
                    f"{file_path.name}: Column '{col}' has {null_count} null values"
                )


def validate_step_output(step: int, check_files: bool = True, verbose: bool = True) -> Dict:
    """
    Validate output files and data quality for a pipeline step.

    Args:
        step: Step number (1-10)
        check_files: Whether to validate file contents (slower but thorough)
        verbose: Print validation progress

    Returns:
        Dict with validation results

    Raises:
        ValidationError: If validation fails
    """
    if step not in STEP_VALIDATIONS:
        raise ValueError(f"Invalid step number: {step}. Must be 1-10.")

    validation = STEP_VALIDATIONS[step]
    results = {
        'step': step,
        'name': validation['name'],
        'files_validated': [],
        'checks_passed': 0,
        'warnings': []
    }

    if verbose:
        print(f"\n🔍 Validating Step {step}: {validation['name']}...")

    # Validate output files exist
    for output_file in validation['outputs']:
        try:
            validate_file_exists(output_file)
            results['files_validated'].append(str(output_file))
            if verbose:
                print(f"  ✓ File exists: {output_file.name}")
        except ValidationError as e:
            raise ValidationError(f"Step {step} validation failed: {e}")

    # Perform data checks if requested
    if check_files:
        for check in validation['checks']:
            file_path = check['file']

            try:
                # Read file
                df = pd.read_csv(file_path)

                # Shape validation
                validate_dataframe_shape(
                    df, file_path,
                    min_rows=check.get('min_rows'),
                    max_rows=check.get('max_rows'),
                    min_features=check.get('min_features'),
                    max_features=check.get('max_features'),
                    expected_features=check.get('expected_features')
                )

                # Column validation
                if 'required_columns' in check:
                    validate_required_columns(df, file_path, check['required_columns'])

                # Leakage check
                if check.get('no_leakage'):
                    validate_no_leakage(df, file_path)

                # Null check for key columns
                if 'required_columns' in check:
                    validate_no_nulls_in_key_columns(
                        df, file_path,
                        [col for col in check['required_columns'] if col != 'Ekipman_ID']
                    )

                results['checks_passed'] += 1
                if verbose:
                    print(f"  ✓ Data validation passed: {file_path.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

            except ValidationError:
                raise
            except Exception as e:
                results['warnings'].append(f"Could not validate {file_path.name}: {e}")

    if verbose:
        print(f"✅ Step {step} validation complete: {results['checks_passed']} checks passed")

    return results


def validate_pipeline_integrity(steps: List[int] = list(range(1, 11)),
                                verbose: bool = True) -> Dict:
    """
    Validate integrity of entire pipeline or specific steps.

    Args:
        steps: List of step numbers to validate (default: all steps)
        verbose: Print validation progress

    Returns:
        Dict with overall validation results
    """
    results = {
        'total_steps': len(steps),
        'passed': 0,
        'failed': 0,
        'warnings': [],
        'step_results': []
    }

    if verbose:
        print("\n" + "="*80)
        print(f"PIPELINE INTEGRITY VALIDATION ({len(steps)} steps)")
        print("="*80)

    for step in steps:
        try:
            step_result = validate_step_output(step, check_files=True, verbose=verbose)
            results['passed'] += 1
            results['step_results'].append(step_result)
        except ValidationError as e:
            results['failed'] += 1
            results['warnings'].append(f"Step {step} failed: {e}")
            if verbose:
                print(f"❌ Step {step} validation failed: {e}")

    if verbose:
        print("\n" + "="*80)
        print(f"VALIDATION SUMMARY: {results['passed']}/{results['total_steps']} steps passed")
        if results['failed'] > 0:
            print(f"⚠️  {results['failed']} step(s) failed validation")
        print("="*80)

    return results


# ============================================================================
# QUICK VALIDATION HELPERS
# ============================================================================

def quick_validate(step: int) -> bool:
    """Quick validation - just check files exist."""
    try:
        validate_step_output(step, check_files=False, verbose=False)
        return True
    except ValidationError:
        return False


def validate_or_exit(step: int, verbose: bool = True) -> None:
    """Validate step output, exit if validation fails."""
    try:
        validate_step_output(step, check_files=True, verbose=verbose)
    except ValidationError as e:
        print(f"\n❌ VALIDATION FAILED: {e}", file=sys.stderr)
        print(f"Pipeline cannot continue. Fix errors in step {step}.", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate PoF Pipeline outputs')
    parser.add_argument('--step', type=int, help='Validate specific step (1-10)')
    parser.add_argument('--all', action='store_true', help='Validate all steps')
    parser.add_argument('--quick', action='store_true', help='Quick validation (files only)')

    args = parser.parse_args()

    try:
        if args.all:
            results = validate_pipeline_integrity(verbose=True)
            if results['failed'] > 0:
                sys.exit(1)
        elif args.step:
            validate_step_output(args.step, check_files=not args.quick, verbose=True)
        else:
            print("Usage: python pipeline_validation.py --step 2")
            print("       python pipeline_validation.py --all")
            sys.exit(1)

    except ValidationError as e:
        print(f"\n❌ Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
