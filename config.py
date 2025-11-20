"""
CENTRALIZED CONFIGURATION
Turkish EDAŞ PoF Prediction Pipeline

This file contains all configuration parameters used across the pipeline.
Update values here instead of modifying individual scripts.

Author: Data Analytics Team
Date: 2025-11-19
Version: 1.0
"""

import pandas as pd
from pathlib import Path

# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================

# Reference/Cutoff Date (split between historical and prediction period)
CUTOFF_DATE = pd.Timestamp('2024-06-25')
REFERENCE_DATE = CUTOFF_DATE  # Alias for backward compatibility

# Prediction Horizons (days)
HORIZONS = {
    '3M': 90,    # 3 months
    '6M': 180,   # 6 months
    '12M': 365,  # 12 months
    '24M': 730   # 24 months
}

# Date Validation
MIN_VALID_YEAR = 1950
MAX_VALID_YEAR = pd.Timestamp.now().year + 1

# ============================================================================
# PIPELINE VALIDATION
# ============================================================================

# Data size validation (lenient thresholds to catch catastrophic failures only)
MIN_EQUIPMENT_RECORDS = 100  # Minimum equipment records for statistical validity
MAX_EQUIPMENT_RECORDS = 5000  # Maximum expected (sanity check for data errors)

# Prediction validation
MIN_PREDICTIONS = 50  # Minimum high-risk predictions expected
MAX_PREDICTIONS = 500  # Maximum high-risk predictions (sanity check)

# Feature validation
MIN_FEATURES = 25  # After feature selection/VIF removal
MAX_FEATURES = 35  # Should have ~30 optimal features

# ============================================================================
# FILE PATHS
# ============================================================================

# Data directories
DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
MODEL_DIR = Path('models')
PREDICTION_DIR = Path('predictions')
RESULTS_DIR = Path('results')

# Input files
INPUT_FILE = DATA_DIR / 'combined_data.xlsx'

# Intermediate files
EQUIPMENT_LEVEL_FILE = DATA_DIR / 'equipment_level_data.csv'
FEATURES_ENGINEERED_FILE = DATA_DIR / 'features_engineered.csv'
FEATURES_REDUCED_FILE = DATA_DIR / 'features_reduced.csv'

# Output files
FEATURE_DOCS_FILE = DATA_DIR / 'feature_documentation.csv'
FEATURE_CATALOG_FILE = DATA_DIR / 'feature_catalog.csv'
HIGH_RISK_FILE = DATA_DIR / 'high_risk_equipment.csv'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.30

# Cross-validation
N_FOLDS = 3

# Class imbalance handling
USE_CLASS_WEIGHTS = True

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Age calculation
USE_FIRST_WORKORDER_FALLBACK = True  # Use first work order if installation date missing

# Geographic clustering
ENABLE_GEOGRAPHIC_CLUSTERING = True
MIN_EQUIPMENT_FOR_CLUSTERING = 10
EQUIPMENT_PER_CLUSTER = 50  # Target equipment per cluster

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# ============================================================================

# VIF thresholds
VIF_THRESHOLD = 10  # Features with VIF > 10 are highly collinear
VIF_TARGET = 10     # Target VIF after iterative removal

# Correlation threshold
CORRELATION_THRESHOLD = 0.85  # Remove features with correlation > 0.85

# Feature importance threshold
IMPORTANCE_THRESHOLD = 0.001  # Keep features contributing > 0.1%

# Protected features (OPTIMAL 30-FEATURE SET - never remove)
# These features form the core of the optimized PoF prediction model
PROTECTED_FEATURES = [
    # Essential ID
    'Ekipman_ID',

    # TIER 1: Equipment Characteristics (3 features)
    'Equipment_Class_Primary',     # Equipment type (most important predictor)
    'component_voltage',            # Operating voltage level
    'Voltage_Class',                # Voltage class (AG/OG/YG)

    # TIER 2: Age & Lifecycle (3 features)
    'Ekipman_Yaşı_Yıl',            # Equipment age in years
    'Yas_Beklenen_Omur_Orani',     # Age as % of expected life
    'Beklenen_Ömür_Yıl',           # Expected equipment lifetime

    # TIER 3: Failure History - Temporal (5 features)
    'Son_Arıza_Gun_Sayisi',        # Days since last failure (recency)
    'Ilk_Arizaya_Kadar_Yil',       # Years until first failure
    'Toplam_Arıza_Sayisi_Lifetime', # Total lifetime faults
    'Time_To_Repair_Hours_mean',   # Average repair time
    'Time_To_Repair_Hours_max',    # Maximum repair time

    # TIER 4: MTBF & Reliability (5 features)
    'MTBF_InterFault_Gün',         # Mean time between consecutive faults
    'MTBF_Lifetime_Gün',           # Overall reliability quality
    'MTBF_ActiveLife_Gün',         # Post-activation reliability
    'MTBF_InterFault_Trend',       # Degradation detector (NEW - TIER 3 enhancement)
    'MTBF_InterFault_StdDev',      # Predictability measure (NEW - TIER 3 enhancement)

    # TIER 5: Failure Cause Patterns (4 features)
    'Arıza_Nedeni_Çeşitlilik',     # Number of different fault causes
    'Arıza_Nedeni_Tutarlılık',     # Consistency of fault causes
    'Neden_Değişim_Flag',          # Whether fault causes changed
    'Tek_Neden_Flag',              # Single dominant cause flag

    # TIER 6: Customer Impact & Loading (5 features)
    'Urban_Customer_Ratio_mean',   # Urban customer density
    'urban_lv_Avg',                # Low voltage urban customers
    'urban_mv_Avg',                # Medium voltage urban customers
    'MV_Customer_Ratio_mean',      # Industrial customer ratio
    'total_customer_count_Avg',    # Total affected customers

    # TIER 7: Geographic & Environmental (3 features)
    'İlçe',                        # District location
    'Bölge_Tipi',                  # Urban vs Rural
    'Summer_Peak_Flag_sum',        # Seasonal stress pattern

    # TIER 8: Derived Interactions (2 features)
    'Overdue_Factor',              # Imminent failure risk (NEW - TIER 3 enhancement)
    'AgeRatio_Recurrence_Interaction', # Compound aging + use risk

    # TARGET (for chronic repeater model)
    'Tekrarlayan_Arıza_90gün_Flag',
]

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

# XGBoost base parameters
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'scale_pos_weight': 1.0  # Will be calculated based on class balance
}

# XGBoost GridSearchCV parameter grid
XGBOOST_GRID = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
    'min_child_weight': [3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 0.1],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [1.0, 2.0]
}

# CatBoost base parameters
CATBOOST_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'auto_class_weights': 'Balanced',
    'task_type': 'CPU',
    'thread_count': -1
}

# CatBoost GridSearchCV parameter grid
CATBOOST_GRID = {
    'iterations': [100, 150],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 5, 6],
    'l2_leaf_reg': [1, 3],
    'border_count': [64]
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = 'INFO'

# Log file location
LOG_DIR = Path('logs')
LOG_FILE = LOG_DIR / 'pipeline.log'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Console logging
CONSOLE_LOG_LEVEL = 'INFO'

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Figure size (inches)
FIGURE_SIZE = (12, 6)

# DPI for saved figures
FIGURE_DPI = 300

# ============================================================================
# EQUIPMENT TYPE MAPPING
# ============================================================================

EQUIPMENT_CLASS_MAPPING = {
    'aghat': 'AG Hat',
    'AG Hat': 'AG Hat',
    'REKORTMAN': 'Rekortman',
    'Rekortman': 'Rekortman',
    'agdirek': 'AG Direk',
    'AG Direk': 'AG Direk',
    'OGAGTRF': 'OG/AG Trafo',
    'OG/AG Trafo': 'OG/AG Trafo',
    'Trafo Bina Tip': 'Trafo Bina Tip',
    'SDK': 'AG Pano Box',
    'AG Pano': 'AG Pano',
    'AG Pano Box': 'AG Pano Box',
    'Ayırıcı': 'Ayırıcı',
    'anahtar': 'AG Anahtar',
    'AG Anahtar': 'AG Anahtar',
    'KESİCİ': 'Kesici',
    'Kesici': 'Kesici',
    'OGHAT': 'OG Hat',
    'PANO': 'Pano',
    'Bina': 'Bina',
    'Armatür': 'Armatür',
    'ENHDirek': 'ENH Direk',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        OUTPUT_DIR,
        MODEL_DIR,
        PREDICTION_DIR,
        RESULTS_DIR,
        LOG_DIR,
        OUTPUT_DIR / 'feature_selection',
        OUTPUT_DIR / 'eda',
        OUTPUT_DIR / 'chronic_repeater',
        OUTPUT_DIR / 'explainability',
        OUTPUT_DIR / 'calibration',
        OUTPUT_DIR / 'survival',
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_target_columns():
    """Get list of target column names based on configured horizons."""
    return [f'Arıza_Olacak_{horizon}' for horizon in HORIZONS.keys()]


def print_config_summary():
    """Print configuration summary."""
    print("="*80)
    print(" "*25 + "PIPELINE CONFIGURATION")
    print("="*80)
    print(f"\nTemporal:")
    print(f"  Cutoff Date: {CUTOFF_DATE.date()}")
    print(f"  Horizons: {', '.join(HORIZONS.keys())}")
    print(f"\nFiles:")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {FEATURES_REDUCED_FILE}")
    print(f"\nModeling:")
    print(f"  Random State: {RANDOM_STATE}")
    print(f"  Test Size: {TEST_SIZE*100:.0f}%")
    print(f"  Cross-Validation: {N_FOLDS} folds")
    print(f"\nFeature Selection:")
    print(f"  VIF Threshold: {VIF_THRESHOLD}")
    print(f"  Correlation Threshold: {CORRELATION_THRESHOLD}")
    print(f"  Protected Features: {len(PROTECTED_FEATURES)}")
    print("="*80)


if __name__ == '__main__':
    # Test configuration by printing summary
    print_config_summary()

    # Create directories
    print("\nCreating directories...")
    create_directories()
    print("✓ All directories created")

    # Verify paths
    print(f"\n✓ Configuration loaded successfully")
    print(f"✓ Cutoff date: {CUTOFF_DATE}")
    print(f"✓ Random state: {RANDOM_STATE}")
    print(f"✓ Horizons: {list(HORIZONS.keys())}")
