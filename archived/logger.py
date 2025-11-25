"""
LOGGING UTILITY
Turkish EDAŞ PoF Prediction Pipeline

Centralized logging configuration for all pipeline scripts.
Provides consistent logging to both console and file.

Author: Data Analytics Team
Date: 2025-11-19
Version: 1.0

Usage:
    from logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.warning("High AUC detected")
    logger.error("Failed to load file")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config import LOG_DIR, LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, CONSOLE_LOG_LEVEL


def setup_logging(log_file=None, log_level=None, console_level=None):
    """
    Setup logging configuration for the pipeline.

    Parameters:
    -----------
    log_file : str or Path, optional
        Path to log file. If None, uses config.LOG_FILE
    log_level : str, optional
        File logging level. If None, uses config.LOG_LEVEL
    console_level : str, optional
        Console logging level. If None, uses config.CONSOLE_LOG_LEVEL

    Returns:
    --------
    logging.Logger : Root logger
    """
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Use defaults from config if not specified
    if log_file is None:
        log_file = LOG_FILE
    if log_level is None:
        log_level = LOG_LEVEL
    if console_level is None:
        console_level = CONSOLE_LOG_LEVEL

    # Convert string levels to logging constants
    file_level = getattr(logging, log_level.upper())
    console_log_level = getattr(logging, console_level.upper())

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, filters apply at handlers

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )

    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name):
    """
    Get a logger instance for a specific module.

    Parameters:
    -----------
    name : str
        Logger name (typically __name__ of the module)

    Returns:
    --------
    logging.Logger : Logger instance

    Examples:
    ---------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    >>> logger.debug("Variable value: %s", some_var)
    >>> logger.warning("High correlation detected: %.2f", correlation)
    >>> logger.error("Failed to process: %s", error_msg)
    """
    return logging.getLogger(name)


def log_script_start(logger, script_name, script_version="1.0"):
    """
    Log script start with standard format.

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    script_name : str
        Name of the script
    script_version : str, optional
        Version of the script
    """
    logger.info("="*80)
    logger.info(f"{script_name} v{script_version} - STARTED")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


def log_script_end(logger, script_name, start_time=None):
    """
    Log script end with standard format.

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    script_name : str
        Name of the script
    start_time : datetime, optional
        Script start time for duration calculation
    """
    logger.info("="*80)
    logger.info(f"{script_name} - COMPLETED")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if start_time:
        duration = datetime.now() - start_time
        logger.info(f"Duration: {duration}")

    logger.info("="*80)


def log_dataframe_info(logger, df, name="DataFrame"):
    """
    Log DataFrame information.

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    df : pd.DataFrame
        DataFrame to log info about
    name : str, optional
        Name of the DataFrame
    """
    logger.info(f"{name}: {len(df):,} rows × {len(df.columns)} columns")

    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.debug(f"{name} memory usage: {memory_mb:.2f} MB")


def log_feature_stats(logger, df, feature_name):
    """
    Log statistics for a specific feature.

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    df : pd.DataFrame
        DataFrame containing the feature
    feature_name : str
        Name of the feature
    """
    if feature_name not in df.columns:
        logger.warning(f"Feature '{feature_name}' not found in DataFrame")
        return

    series = df[feature_name]

    # Basic stats
    logger.debug(f"{feature_name} - Non-null: {series.notna().sum():,}/{len(series):,} ({series.notna().sum()/len(series)*100:.1f}%)")

    if series.dtype in ['int64', 'float64']:
        logger.debug(f"{feature_name} - Mean: {series.mean():.2f}, Median: {series.median():.2f}, Std: {series.std():.2f}")
        logger.debug(f"{feature_name} - Min: {series.min():.2f}, Max: {series.max():.2f}")


def log_model_metrics(logger, model_name, metrics_dict):
    """
    Log model performance metrics.

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model
    metrics_dict : dict
        Dictionary of metric names and values

    Examples:
    ---------
    >>> metrics = {'AUC': 0.73, 'Precision': 0.25, 'Recall': 0.85}
    >>> log_model_metrics(logger, 'XGBoost', metrics)
    """
    logger.info(f"{model_name} Performance:")
    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


# Initialize logging when module is imported (optional - can be disabled)
_logging_initialized = False

def initialize_logging():
    """Initialize logging on first import."""
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
        _logging_initialized = True


if __name__ == '__main__':
    # Test logging functionality
    print("Testing logging functionality...\n")

    # Setup logging
    setup_logging()

    # Get logger
    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("This is a DEBUG message (detailed info)")
    logger.info("This is an INFO message (general info)")
    logger.warning("This is a WARNING message (potential issues)")
    logger.error("This is an ERROR message (errors)")
    logger.critical("This is a CRITICAL message (critical issues)")

    # Test helper functions
    log_script_start(logger, "Test Script", "1.0")

    import pandas as pd
    test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    log_dataframe_info(logger, test_df, "Test DataFrame")

    metrics = {'AUC': 0.7388, 'Precision': 0.1875, 'Recall': 0.1667}
    log_model_metrics(logger, 'XGBoost', metrics)

    log_script_end(logger, "Test Script", datetime.now())

    print(f"\n✓ Logging test complete. Check log file: {LOG_FILE}")
