"""
Date processing functionality for SLA Predictor
GREEN phase: Minimal implementation to make tests pass
"""
import pandas as pd
import logging
from datetime import datetime, date
from typing import Union, Optional


class DateProcessor:
    """Handles date standardization and validation for SLA prediction."""
    
    def __init__(self):
        """Initialize DateProcessor with logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = [
            "%Y-%m-%d",      # 2023-01-01
            "%m/%d/%Y",      # 01/01/2023
            "%m-%d-%Y",      # 01-01-2023
            "%d/%m/%Y",      # 01/01/2023 (DD/MM/YYYY)
        ]
    
    def standardize_date(self, date_str: Union[str, None], format_hint: Optional[str] = None) -> date:
        """
        Standardize date string to date object.
        
        Args:
            date_str: Date string to standardize
            format_hint: Optional format hint for parsing
            
        Returns:
            date: Standardized date object
            
        Raises:
            ValueError: If date is invalid or cannot be parsed
        """
        if date_str is None:
            raise ValueError("Date cannot be None")
        
        if not date_str or not date_str.strip():
            raise ValueError("Empty date string")
        
        date_str = date_str.strip()
        
        # Try different formats
        formats_to_try = self.supported_formats.copy()
        
        # If format hint provided, try that first
        if format_hint == "DD/MM/YYYY":
            formats_to_try.insert(0, "%d/%m/%Y")
        
        for fmt in formats_to_try:
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                self.logger.debug(f"Successfully parsed date '{date_str}' with format '{fmt}'")
                return parsed_date
            except ValueError:
                continue
        
        # If no format worked, raise error
        self.logger.error(f"Could not parse date: {date_str}")
        raise ValueError(f"Invalid date format: {date_str}")
    
    def standardize_dataframe_dates(self, df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """
        Standardize date column in DataFrame.
        
        Args:
            df: DataFrame with date column
            date_column: Name of date column to standardize
            
        Returns:
            pd.DataFrame: DataFrame with standardized date column
            
        Raises:
            ValueError: If date column missing or contains invalid dates
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        df_copy = df.copy()
        
        try:
            # Try pandas built-in date parsing first
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            self.logger.info(f"Successfully standardized {len(df_copy)} dates")
            return df_copy
        except Exception as e:
            self.logger.error(f"Error parsing dates in column '{date_column}': {e}")
            raise ValueError("Invalid date found in DataFrame")
    
    def validate_date_range(self, check_date: date, min_date: date, max_date: date) -> bool:
        """
        Validate that date falls within acceptable range.
        
        Args:
            check_date: Date to validate
            min_date: Minimum allowed date
            max_date: Maximum allowed date
            
        Returns:
            bool: True if date is valid
            
        Raises:
            ValueError: If date is outside allowed range
        """
        if check_date < min_date:
            raise ValueError(f"Date {check_date} is before minimum allowed date {min_date}")
        
        if check_date > max_date:
            raise ValueError(f"Date {check_date} is after maximum allowed date {max_date}")
        
        return True