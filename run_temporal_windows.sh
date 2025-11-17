#!/bin/bash

###############################################################################
# WALK-FORWARD TEMPORAL VALIDATION
# Turkish EDAŞ PoF Prediction Project
#
# Purpose:
#   - Generate multiple training windows with proper temporal cutoffs
#   - Prevent temporal leakage in model validation
#   - Enable walk-forward validation for production readiness
#
# Usage:
#   bash run_temporal_windows.sh
#
# Output:
#   - data/equipment_level_data_temporal_window1_2023.csv
#   - data/equipment_level_data_temporal_window2_2024Q1.csv
#   - data/equipment_level_data_temporal_window3_2024Q2.csv
#   - (+ metadata and documentation files)
#
# Author: Data Analytics Team
# Date: 2025-11-17
###############################################################################

set -e  # Exit on error

echo "================================================================================"
echo "          WALK-FORWARD TEMPORAL VALIDATION - GENERATING TRAINING WINDOWS"
echo "================================================================================"
echo ""

# Check if script exists
if [ ! -f "02_data_transformation_temporal.py" ]; then
    echo "❌ ERROR: 02_data_transformation_temporal.py not found!"
    echo "   Please ensure you're running this from the project root directory."
    exit 1
fi

# Check if data file exists
if [ ! -f "data/combined_data.xlsx" ]; then
    echo "❌ ERROR: data/combined_data.xlsx not found!"
    echo "   Please ensure the source data file exists."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

echo "✓ Prerequisites checked"
echo ""

###############################################################################
# WINDOW 1: 2023-06-25 (Historical Validation)
###############################################################################
echo "================================================================================"
echo "WINDOW 1: Historical Validation (Cutoff: 2023-06-25)"
echo "================================================================================"
echo ""
echo "  Training data: All faults before 2023-06-25"
echo "  Target period: 2023-06-26 to 2024-06-25 (12 months)"
echo "  Purpose: Validate model on older time period (known outcomes)"
echo ""

python 02_data_transformation_temporal.py \
    --cutoff_date 2023-06-25 \
    --prediction_horizon 365 \
    --output_suffix "_window1_2023"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Window 1 completed successfully"
    echo ""
else
    echo ""
    echo "❌ Window 1 failed!"
    exit 1
fi

###############################################################################
# WINDOW 2: 2024-01-01 (Recent Validation)
###############################################################################
echo "================================================================================"
echo "WINDOW 2: Recent Validation (Cutoff: 2024-01-01)"
echo "================================================================================"
echo ""
echo "  Training data: All faults before 2024-01-01"
echo "  Target period: 2024-01-02 to 2025-01-01 (12 months)"
echo "  Purpose: Validate performance on recent data (known outcomes)"
echo ""

python 02_data_transformation_temporal.py \
    --cutoff_date 2024-01-01 \
    --prediction_horizon 365 \
    --output_suffix "_window2_2024Q1"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Window 2 completed successfully"
    echo ""
else
    echo ""
    echo "❌ Window 2 failed!"
    exit 1
fi

###############################################################################
# WINDOW 3: 2024-06-25 (Production Deployment)
###############################################################################
echo "================================================================================"
echo "WINDOW 3: Production Deployment (Cutoff: 2024-06-25)"
echo "================================================================================"
echo ""
echo "  Training data: All faults before 2024-06-25"
echo "  Target period: 2024-06-26 to 2025-06-25 (12 months)"
echo "  Purpose: Train final production model for current deployment"
echo ""

python 02_data_transformation_temporal.py \
    --cutoff_date 2024-06-25 \
    --prediction_horizon 365 \
    --output_suffix "_window3_2024Q2"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Window 3 completed successfully"
    echo ""
else
    echo ""
    echo "❌ Window 3 failed!"
    exit 1
fi

###############################################################################
# SUMMARY
###############################################################################
echo "================================================================================"
echo "                         ALL WINDOWS COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📄 data/equipment_level_data_temporal_window1_2023.csv"
echo "  📄 data/equipment_level_data_temporal_window2_2024Q1.csv"
echo "  📄 data/equipment_level_data_temporal_window3_2024Q2.csv"
echo "  📄 (+ metadata and documentation files)"
echo ""
echo "Next steps:"
echo "  1. Validate outputs using TEMPORAL_VALIDATION_GUIDE.md"
echo "  2. Run walk-forward validation to test model generalization"
echo "  3. Train production model on Window 3 (most recent data)"
echo ""
echo "Walk-forward validation strategy:"
echo "  • Train on Window 1 → Test on Window 2 (out-of-sample)"
echo "  • Train on Window 2 → Test on Window 3 (out-of-sample)"
echo "  • Train on Window 3 → Deploy to production"
echo ""
echo "================================================================================"
echo "                         TEMPORAL PIPELINE READY FOR ML"
echo "================================================================================"
