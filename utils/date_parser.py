"""
DATE PARSING UTILITIES
======================
Centralized date parsing functions for PoF2 pipeline.

This module provides flexible date parsing that handles:
- ISO format (YYYY-MM-DD)
- Turkish/European formats (DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY)
- Excel serial dates (numeric)
- Mixed format data in a single column

Usage:
    from utils.date_parser import parse_date_flexible

    df['date_column'] = df['date_column'].apply(parse_date_flexible)
"""

import pandas as pd
from datetime import datetime


def parse_date_flexible(value):
    """
    Parse date with multiple format support - handles mixed format data.

    Supports:
    - ISO format: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS
    - Turkish/European: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY (with/without time)
    - Excel serial dates: numeric values between 1 and 100000
    - US format (as fallback): MM/DD/YYYY

    Args:
        value: Date value to parse (str, int, float, datetime, or pd.Timestamp)

    Returns:
        pd.Timestamp: Parsed timestamp, or pd.NaT if parsing fails

    Examples:
        >>> parse_date_flexible('2021-01-15')
        Timestamp('2021-01-15 00:00:00')

        >>> parse_date_flexible('15/01/2021')
        Timestamp('2021-01-15 00:00:00')

        >>> parse_date_flexible(44208)  # Excel serial date
        Timestamp('2021-01-15 00:00:00')

        >>> parse_date_flexible(None)
        NaT
    """
    # Already a timestamp/datetime
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)

    # Handle NaN/None
    if pd.isna(value):
        return pd.NaT

    # Excel serial date (numeric)
    if isinstance(value, (int, float)):
        # Excel serial dates are typically between 1 (1900-01-01) and 100000
        if 1 <= value <= 100000:
            try:
                # Excel's epoch: 1899-12-30 (note: Excel incorrectly treats 1900 as leap year)
                return pd.Timestamp('1899-12-30') + pd.Timedelta(days=value)
            except:
                return pd.NaT
        else:
            return pd.NaT

    # String parsing with multiple format attempts
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return pd.NaT

        # Try multiple formats in order of likelihood
        formats = [
            '%Y-%m-%d %H:%M:%S',     # 2021-01-15 12:30:45 (ISO with time)
            '%d-%m-%Y %H:%M:%S',     # 15-01-2021 12:30:45 (Turkish/European dash with time)
            '%d/%m/%Y %H:%M:%S',     # 15/01/2021 12:30:45 (Turkish/European slash with time)
            '%Y-%m-%d',              # 2021-01-15 (ISO date only)
            '%d-%m-%Y',              # 15-01-2021 (Turkish/European dash date only)
            '%d/%m/%Y',              # 15/01/2021 (Turkish/European slash date only)
            '%d.%m.%Y %H:%M:%S',     # 15.01.2021 12:30:45 (Turkish dot format with time)
            '%d.%m.%Y',              # 15.01.2021 (Turkish dot format date only)
            '%m/%d/%Y %H:%M:%S',     # 01/15/2021 12:30:45 (US format with time - try last)
            '%m/%d/%Y',              # 01/15/2021 (US format date only - try last)
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except:
                continue

        # Last resort: let pandas infer format (with Turkish/European preference)
        try:
            return pd.to_datetime(value, infer_datetime_format=True, dayfirst=True)
        except:
            return pd.NaT

    return pd.NaT
