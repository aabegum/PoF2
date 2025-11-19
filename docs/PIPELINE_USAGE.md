# PoF2 Pipeline Usage Guide

This guide explains how to run the complete PoF2 pipeline with automated logging.

## Quick Start

### Option 1: Windows Batch Script (Recommended for Windows)

```bash
run_pipeline.bat
```

### Option 2: Python Script (Cross-platform)

```bash
python run_pipeline.py
```

### Option 3: Manual Execution (Individual Scripts)

```bash
python 01_data_profiling.py
python 02_data_transformation.py
python 03_feature_engineering.py
python 04_eda.py
python 05_feature_selection.py
python 05b_remove_leaky_features.py
python 06_model_training.py
python 09_survival_analysis.py
python 10_consequence_of_failure.py
```

## Pipeline Steps

| Step | Script | Purpose | Estimated Time |
|------|--------|---------|----------------|
| 1 | `01_data_profiling.py` | Load and profile raw fault data | ~30s |
| 2 | `02_data_transformation.py` | Transform to equipment-level data | ~1min |
| 3 | `03_feature_engineering.py` | Create failure prediction features | ~1min |
| 4 | `04_eda.py` | Generate 16 exploratory analyses | ~3-5min |
| 5 | `05_feature_selection.py` | Select relevant features for modeling | ~1min |
| 6 | `05b_remove_leaky_features.py` | Remove features with data leakage | ~1min |
| 7 | `06_model_training.py` | Train chronic repeater classifier (Model 2) | ~2-3min |
| 8 | `09_survival_analysis.py` | Train temporal PoF predictor (Model 1) | ~2-3min |
| 9 | `10_consequence_of_failure.py` | Calculate CoF & Risk scores | ~1min |

**Total Estimated Time:** 10-20 minutes

## Log Files

When using `run_pipeline.bat` or `run_pipeline.py`, all outputs are saved to:

```
logs/
└── run_YYYYMMDD_HHMMSS/          # Timestamped run directory
    ├── pipeline_master.log        # All outputs combined
    ├── pipeline_summary.txt       # Execution summary with timings
    ├── 01_data_profiling.log
    ├── 02_data_transformation.log
    ├── 03_feature_engineering.log
    ├── 04_eda.log
    ├── 05_feature_selection.log
    ├── 05b_remove_leaky_features.log
    ├── 06_model_training.log
    ├── 09_survival_analysis.log
    └── 10_consequence_of_failure.log
```

### Log File Contents

- **Individual logs** (`01_*.log`, `02_*.log`, etc.): Console output from each script
- **Master log** (`pipeline_master.log`): All outputs combined in chronological order
- **Summary** (`pipeline_summary.txt`): Execution times and status for each step

## Output Files

After successful pipeline execution, you'll have:

### Results (CSV Files)
```
results/
├── risk_assessment_3M.csv         # 3-month risk assessment
├── risk_assessment_12M.csv        # 12-month risk assessment (primary)
├── risk_assessment_24M.csv        # 24-month risk assessment
└── capex_priority_list.csv        # Top 100 equipment for CAPEX
```

### Visualizations
```
outputs/
├── eda/                           # 16 exploratory analysis plots
│   ├── 01_equipment_class_distribution.png
│   ├── 02_temporal_trends.png
│   ├── 03_district_analysis.png
│   ├── 04_failure_frequency_distribution.png
│   ├── 05_age_analysis.png
│   ├── 06_mtbf_analysis.png
│   ├── 07_failure_rate_analysis.png
│   ├── 08_recurrence_patterns.png
│   ├── 09_customer_impact.png
│   ├── 10_model_predictions.png
│   ├── 11_reliability_metrics.png
│   ├── 12_multivariate_relationships.png
│   ├── 13_top_equipment_heatmap.png
│   ├── 14_correlation_matrix.png
│   ├── 15_cause_code_analysis.png
│   └── 16_recurring_failure_analysis.png
│
└── risk_analysis/                 # 6 risk assessment plots
    ├── risk_matrix_3M.png
    ├── risk_matrix_12M.png
    ├── risk_matrix_24M.png
    ├── risk_distribution_by_class_3M.png
    ├── risk_distribution_by_class_12M.png
    └── risk_distribution_by_class_24M.png
```

### Models and Predictions
```
models/                            # Trained ML models
predictions/                       # PoF predictions
└── pof_multi_horizon_predictions.csv
```

## Understanding the Results

### Risk Assessment Files

Each `risk_assessment_*.csv` contains:
- `Ekipman_ID`: Equipment identifier
- `Ekipman_Sınıfı`: Equipment class (Ayırıcı, Trafo, etc.)
- `İlçe`: District (Salihli, Alaşehir, Gördes)
- `PoF_Probability`: Probability of failure (0.0-1.0)
- `CoF_Score`: Consequence of failure score (0-100, percentile-based)
- `Risk_Score`: Combined risk score (0-100, percentile-based)
- `Risk_Category`: DÜŞÜK/ORTA/YÜKSEK/KRİTİK
- `Priority_Rank`: Priority ranking (1 = highest risk)
- `Avg_Outage_Minutes`: Average outage duration
- `Total_Customers_Affected`: Total customers impacted

### Risk Categories (Percentile-Based)

- **DÜŞÜK (Low):** 0-75th percentile (~75% of equipment)
- **ORTA (Medium):** 75-90th percentile (~15% of equipment)
- **YÜKSEK (High):** 90-95th percentile (~5% of equipment)
- **KRİTİK (Critical):** 95-100th percentile (~5% of equipment)

### CAPEX Priority List

`capex_priority_list.csv` contains the top 100 highest-risk equipment with:
- Recommended action (IMMEDIATE REPLACEMENT, PRIORITY REPLACEMENT, etc.)
- Risk scores and components (PoF, CoF)
- Customer impact metrics

## Troubleshooting

### Pipeline Fails at a Specific Step

1. Check the individual log file for that step:
   ```
   logs/run_YYYYMMDD_HHMMSS/0X_script_name.log
   ```

2. Look for error messages (usually marked with `ERROR:` or `❌`)

3. Common issues:
   - **Missing data files:** Ensure `data/combined_data.xlsx` exists
   - **Python package missing:** Run `pip install -r requirements.txt`
   - **Memory error:** Close other applications, increase system RAM
   - **Encoding error:** Check that data files use UTF-8 encoding

### Checking Log Files

**Windows:**
```bash
type logs\run_YYYYMMDD_HHMMSS\pipeline_summary.txt
notepad logs\run_YYYYMMDD_HHMMSS\pipeline_master.log
```

**Linux/Mac:**
```bash
cat logs/run_YYYYMMDD_HHMMSS/pipeline_summary.txt
less logs/run_YYYYMMDD_HHMMSS/pipeline_master.log
```

### Re-running from a Specific Step

If a step fails, you can manually run just that step:

```bash
python 0X_script_name.py
```

Then continue with subsequent steps.

## Performance Tips

- **Faster execution:** Close unnecessary applications to free RAM
- **Parallel processing:** Some scripts use multi-core processing (automatic)
- **Disk space:** Ensure ~500MB free space for logs and outputs

## Advanced Usage

### Running Only Specific Steps

You can comment out steps in the batch/Python script or run individual scripts:

```bash
# Just update the risk assessment without retraining models
python 10_consequence_of_failure.py
```

### Customizing Risk Thresholds

Edit `10_consequence_of_failure.py` lines 373-377 to change percentile thresholds.

### Modifying Log Directory

Edit the `LOG_DIR` or `log_dir` variable in the runner scripts to change the log location.

## Support

For issues or questions, check:
1. Individual log files for detailed error messages
2. `pipeline_summary.txt` for execution time anomalies
3. `pipeline_master.log` for complete console output

---

**Last Updated:** 2025-01-14
**Pipeline Version:** 2.0
**Author:** Data Analytics Team
