"""
POF2 PIPELINE RUNNER WITH LOGGING v3.0
Turkish EDAŞ Equipment Failure Prediction Pipeline

This script runs the entire production-ready PoF2 pipeline and captures all outputs.

PIPELINE FLOW (12 STEPS):
    1. Data Profiling          → Validate data quality and temporal coverage
   2a. Healthy Equipment Loader → Load healthy equipment data (OPTIONAL - for balanced training)
    2. Data Transformation     → Transform fault-level to equipment-level + merge healthy
    3. Feature Engineering     → Create optimal feature set
    4. Feature Selection       → Leakage removal + VIF reduction
    5. Equipment ID Audit      → Verify ID consolidation (optional)
    6. Temporal PoF Model      → XGBoost/CatBoost multi-horizon predictions (mixed dataset)
    7. Chronic Classifier      → Identify failure-prone equipment (failed only)
    8. Model Explainability    → SHAP feature importance
    9. Probability Calibration → Calibrate risk estimates
   10. Cox Survival Model      → Multi-horizon survival analysis (censored observations)
   11. Risk Assessment         → PoF × CoF → CAPEX priority list

NEW in v3.0 (MIXED DATASET SUPPORT):
    • Step 2a: Load healthy equipment (optional - enables balanced training)
    • Steps 2, 6-11: Support mixed dataset (failed + healthy equipment)
    • Benefits: Better calibration, reduced false positives, realistic risk scores

OPTIONAL SCRIPTS (in analysis/ folder):
    • analysis/exploratory/04_eda.py - 16 exploratory analyses
    • analysis/diagnostics/06b_logistic_baseline.py - Baseline comparison

Usage:
    python run_pipeline.py

Output:
    - Individual log files for each step in logs/run_TIMESTAMP/
    - Master log file with all outputs combined
    - Summary report with execution times
    - Risk assessment and CAPEX priority CSVs
    - Model files and prediction CSVs
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

# Import validation module for data integrity checks
from pipeline_validation import validate_step_output, ValidationError

# Pipeline configuration (PRODUCTION-READY - Phase 1 Optimized)
# NOTE: 04_eda.py is OPTIONAL - run separately for research/analysis
# NOTE: 06b_logistic_baseline.py is OPTIONAL - baseline comparison only
PIPELINE_STEPS = [
    {
        'step': 1,
        'name': 'Data Profiling',
        'script': '01_data_profiling.py',
        'description': 'Validate data quality and temporal coverage'
    },
    {
        'step': '2a',
        'name': 'Healthy Equipment Loader',
        'script': '02a_healthy_equipment_loader.py',
        'description': 'Load and validate healthy equipment data (OPTIONAL - enables mixed dataset)',
        'optional': True  # Optional - only run if healthy_equipment.xlsx exists
    },
    {
        'step': 2,
        'name': 'Data Transformation',
        'script': '02_data_transformation.py',
        'description': 'Transform fault-level to equipment-level + merge with healthy data'
    },
    {
        'step': 3,
        'name': 'Feature Engineering',
        'script': '03_feature_engineering.py',
        'description': 'Create optimal feature set (TIER 1-8)'
    },
    {
        'step': 4,
        'name': 'Feature Selection',
        'script': '04_feature_selection.py',
        'description': 'Leakage removal + VIF analysis'
    },
    {
        'step': 5,
        'name': 'Equipment ID Audit',
        'script': '05_equipment_id_audit.py',
        'description': 'Verify cbs_id ↔ Ekipman_ID consistency (CRITICAL)',
        'optional': True  # Optional diagnostic - will warn but not fail pipeline
    },
    {
        'step': 6,
        'name': 'Temporal PoF Model',
        'script': '06_temporal_pof_model.py',
        'description': 'Train temporal PoF predictor (3M/6M/12M) - supports mixed dataset'
    },
    {
        'step': 7,
        'name': 'Chronic Classifier',
        'script': '07_chronic_classifier.py',
        'description': 'Train chronic repeater classifier (90-day recurrence) - failed equipment only'
    },
    {
        'step': 8,
        'name': 'Model Explainability',
        'script': '08_explainability.py',
        'description': 'SHAP analysis and feature importance'
    },
    {
        'step': 9,
        'name': 'Probability Calibration',
        'script': '09_calibration.py',
        'description': 'Calibrate model probabilities - improved from mixed dataset'
    },
    {
        'step': 10,
        'name': 'Cox Survival Model',
        'script': '10_survival_model.py',
        'description': 'Cox PH + Kaplan-Meier (multi-horizon) - adds censored observations'
    },
    {
        'step': 11,
        'name': 'Risk Assessment',
        'script': '11_consequence_of_failure.py',
        'description': 'Calculate PoF × CoF = Risk, generate CAPEX priority list'
    }
]

def run_pipeline():
    """Run the complete PoF2 pipeline with logging."""

    # Create logs directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f'logs/run_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)

    master_log_path = log_dir / 'pipeline_master.log'
    summary_path = log_dir / 'pipeline_summary.txt'

    print("="*100)
    print("                    POF2 PIPELINE EXECUTION")
    print("="*100)
    print(f"\n📂 Log directory: {log_dir}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize master log
    with open(master_log_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("POF2 PIPELINE EXECUTION LOG\n")
        f.write("="*100 + "\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

    # Track execution times
    start_time = time.time()
    step_times = []

    # Run each step
    for step_info in PIPELINE_STEPS:
        step_num = step_info['step']
        step_name = step_info['name']
        script = step_info['script']
        description = step_info['description']

        print(f"[STEP {step_num}/{len(PIPELINE_STEPS)}] {step_name}")
        print(f"  → {description}")
        print(f"  → Running {script}...")

        # Individual log file
        log_file = log_dir / f"{script.replace('.py', '.log')}"

        # Run the script
        step_start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            step_duration = time.time() - step_start
            step_times.append({
                'step': step_num,
                'name': step_name,
                'duration': step_duration,
                'status': 'SUCCESS' if result.returncode == 0 else 'FAILED'
            })

            # Save individual log
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"STEP {step_num}: {step_name}\n")
                f.write(f"Script: {script}\n")
                f.write(f"Duration: {step_duration:.1f} seconds\n")
                f.write("="*100 + "\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nSTDERR:\n")
                    f.write(result.stderr)

            # Append to master log
            with open(master_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"STEP {step_num}: {step_name}\n")
                f.write(f"Script: {script}\n")
                f.write(f"Duration: {step_duration:.1f} seconds\n")
                f.write('='*100 + "\n\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nSTDERR:\n")
                    f.write(result.stderr)
                f.write("\n")

            # Check for errors
            if result.returncode != 0:
                # Check if this is an optional step
                is_optional = step_info.get('optional', False)

                if is_optional:
                    print(f"  ⚠️  OPTIONAL STEP FAILED (exit code: {result.returncode})")
                    print(f"  → Check log file: {log_file}")
                    print(f"  → Pipeline will continue (optional diagnostic)")
                else:
                    print(f"  ✗ FAILED (exit code: {result.returncode})")
                    print(f"  → Check log file: {log_file}")
                    print(f"\n❌ Pipeline failed at step {step_num}")
                    print(f"   Check {log_file} for details")
                    return False

            print(f"  ✓ Completed ({step_duration:.1f}s)")

            # Validate output files (for steps that produce data files)
            if step_num in [2, 3, 4, 5, 6, 9, 10]:  # Steps with data outputs
                try:
                    print(f"  → Validating outputs...")
                    validate_step_output(step_num, check_files=True, verbose=False)
                    print(f"  ✓ Validation passed")
                except ValidationError as e:
                    print(f"  ✗ VALIDATION FAILED: {e}")
                    print(f"\n❌ Pipeline stopped - data validation failed at step {step_num}")
                    print(f"   Fix data quality issues before continuing")
                    return False
                except Exception as e:
                    print(f"  ⚠️  Validation warning: {e}")
                    # Continue anyway - validation issues are warnings, not blockers

            print()

        except Exception as e:
            print(f"  ✗ FAILED with exception: {e}")
            step_times.append({
                'step': step_num,
                'name': step_name,
                'duration': time.time() - step_start,
                'status': 'EXCEPTION'
            })
            return False

    # Calculate total time
    total_duration = time.time() - start_time

    # Create summary report
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("POF2 PIPELINE EXECUTION SUMMARY\n")
        f.write("="*100 + "\n\n")

        f.write(f"Started:  {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)\n\n")

        f.write("STEP EXECUTION TIMES:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Step':<6} {'Name':<35} {'Duration (s)':<15} {'Status':<10}\n")
        f.write("-"*100 + "\n")

        for step_time in step_times:
            f.write(f"{step_time['step']:<6} {step_time['name']:<35} "
                   f"{step_time['duration']:>13.1f}s  {step_time['status']:<10}\n")

        f.write("-"*100 + "\n")
        f.write(f"{'TOTAL':<42} {total_duration:>13.1f}s  {'SUCCESS':<10}\n")
        f.write("-"*100 + "\n\n")

        f.write("OUTPUT FILES GENERATED:\n")
        f.write("  DATA OUTPUTS:\n")
        f.write("    • data/equipment_level_data.csv - Equipment-level dataset\n")
        f.write("    • data/features_engineered.csv - Engineered features\n")
        f.write("    • data/features_reduced.csv - Final feature set\n\n")
        f.write("  PREDICTIONS:\n")
        f.write("    • predictions/predictions_3m.csv - 3-month temporal PoF\n")
        f.write("    • predictions/predictions_6m.csv - 6-month temporal PoF\n")
        f.write("    • predictions/predictions_12m.csv - 12-month temporal PoF\n")
        f.write("    • predictions/chronic_repeaters.csv - Chronic repeater classifications\n")
        f.write("    • predictions/pof_multi_horizon_predictions.csv - Multi-horizon survival\n\n")
        f.write("  RISK ASSESSMENT:\n")
        f.write("    • results/risk_assessment_3M.csv - 3-month risk scores\n")
        f.write("    • results/risk_assessment_6M.csv - 6-month risk scores\n")
        f.write("    • results/risk_assessment_12M.csv - 12-month risk scores\n")
        f.write("    • results/capex_priority_list.csv - CAPEX priority list\n\n")
        f.write("  MODELS:\n")
        f.write("    • models/*.pkl - Trained XGBoost/CatBoost models\n\n")
        f.write("  VISUALIZATIONS:\n")
        f.write("    • outputs/feature_selection/*.png - Feature selection analysis\n")
        f.write("    • outputs/explainability/*.png - SHAP visualizations\n")
        f.write("    • outputs/calibration/*.png - Calibration curves\n")
        f.write("    • outputs/survival/*.png - Kaplan-Meier curves\n\n")

        f.write("LOG FILES:\n")
        f.write(f"  • Individual logs: {log_dir}/*.log\n")
        f.write(f"  • Master log: {master_log_path}\n")
        f.write(f"  • This summary: {summary_path}\n")

    # Print success message
    print("="*100)
    print("                    PIPELINE COMPLETED SUCCESSFULLY")
    print("="*100)
    print()
    print(f"⏰ Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print()
    print("📊 STEP EXECUTION TIMES:")
    print("-"*100)
    print(f"{'Step':<6} {'Name':<35} {'Duration (s)':<15} {'Status':<10}")
    print("-"*100)

    for step_time in step_times:
        print(f"{step_time['step']:<6} {step_time['name']:<35} "
              f"{step_time['duration']:>13.1f}s  {step_time['status']:<10}")

    print("-"*100)
    print()
    print("📂 LOG FILES:")
    print(f"   • Individual logs: {log_dir}/*.log")
    print(f"   • Master log: {master_log_path}")
    print(f"   • Summary: {summary_path}")
    print()
    print("📊 KEY OUTPUT FILES:")
    print("   DATA:")
    print("     • data/equipment_level_data.csv")
    print("     • data/features_engineered.csv")
    print("     • data/features_reduced.csv")
    print()
    print("   PREDICTIONS:")
    print("     • predictions/predictions_*.csv (3M, 6M, 12M)")
    print("     • predictions/chronic_repeaters.csv")
    print("     • predictions/pof_multi_horizon_predictions.csv")
    print()
    print("   RISK & CAPEX:")
    print("     • results/risk_assessment_*.csv (3M, 6M, 12M)")
    print("     • results/capex_priority_list.csv")
    print()
    print("   MODELS & VISUALIZATIONS:")
    print("     • models/*.pkl")
    print("     • outputs/*/*.png")
    print()

    return True

if __name__ == '__main__':
    try:
        success = run_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
