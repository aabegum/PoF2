"""
POF2 PIPELINE RUNNER WITH LOGGING
Turkish EDAŞ Equipment Failure Prediction Pipeline

This script runs the entire PoF2 pipeline and captures all outputs to log files.

Usage:
    python run_pipeline.py

Output:
    - Individual log files for each step in logs/run_TIMESTAMP/
    - Master log file with all outputs combined
    - Summary report with execution times
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

# Pipeline configuration
PIPELINE_STEPS = [
    {
        'step': 1,
        'name': 'Data Profiling',
        'script': '01_data_profiling.py',
        'description': 'Loading and profiling raw fault data'
    },
    {
        'step': 2,
        'name': 'Data Transformation',
        'script': '02_data_transformation.py',
        'description': 'Transforming to equipment-level data'
    },
    {
        'step': 3,
        'name': 'Feature Engineering',
        'script': '03_feature_engineering.py',
        'description': 'Creating failure prediction features'
    },
    {
        'step': 4,
        'name': 'Exploratory Data Analysis',
        'script': '04_eda.py',
        'description': 'Analyzing all features (16 analyses)'
    },
    {
        'step': 5,
        'name': 'Feature Selection',
        'script': '05_feature_selection.py',
        'description': 'Selecting relevant features for modeling'
    },
    {
        'step': 6,
        'name': 'Remove Leaky Features',
        'script': '05b_remove_leaky_features.py',
        'description': 'Removing features with data leakage'
    },
    {
        'step': 7,
        'name': 'Model Training (Model 2)',
        'script': '06_model_training.py',
        'description': 'Training chronic repeater classifier'
    },
    {
        'step': 8,
        'name': 'Survival Analysis (Model 1)',
        'script': '09_survival_analysis.py',
        'description': 'Training temporal PoF predictor'
    },
    {
        'step': 9,
        'name': 'Risk Assessment',
        'script': '10_consequence_of_failure.py',
        'description': 'Calculating CoF and Risk scores'
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
                print(f"  ✗ FAILED (exit code: {result.returncode})")
                print(f"  → Check log file: {log_file}")
                print(f"\n❌ Pipeline failed at step {step_num}")
                print(f"   Check {log_file} for details")
                return False

            print(f"  ✓ Completed ({step_duration:.1f}s)")
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
        f.write("  • results/risk_assessment_3M.csv\n")
        f.write("  • results/risk_assessment_12M.csv\n")
        f.write("  • results/risk_assessment_24M.csv\n")
        f.write("  • results/capex_priority_list.csv\n")
        f.write("  • outputs/eda/*.png (16 EDA visualizations)\n")
        f.write("  • outputs/risk_analysis/*.png (6 risk visualizations)\n")
        f.write("  • models/* (trained models)\n")
        f.write("  • predictions/* (PoF predictions)\n\n")

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
    print("📊 OUTPUT FILES:")
    print("   • results/risk_assessment_*.csv")
    print("   • results/capex_priority_list.csv")
    print("   • outputs/eda/*.png")
    print("   • outputs/risk_analysis/*.png")
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
