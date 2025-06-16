"""
Feature Engineering Module for SLA Predictor.

This module provides feature extraction functionality for the SLA prediction system,
including calendar features, temporal features, and holiday integration.
"""

import logging
from datetime import datetime, date
from typing import Dict, Union, Any
import pandas as pd
import holidays


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for extracting calendar, temporal, and holiday features.
    
    This class provides methods to extract various features from dates that are useful
    for SLA prediction, including day of week, temporal patterns, and holiday information.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer with US federal holidays."""
        self.us_holidays = holidays.US()
        logger.info("FeatureEngineer initialized with US federal holidays")
    
    def extract_calendar_features(self, input_date: Union[date, datetime]) -> Dict[str, int]:
        """
        Extract calendar-based features from a date.
        
        Args:
            input_date: Date or datetime object to extract features from
            
        Returns:
            Dictionary containing calendar features:
            - day_of_week: Day of week (0=Monday, 6=Sunday)
            - day_of_month: Day of the month (1-31)
            - month: Month (1-12)
            
        Raises:
            ValueError: If input_date is not a valid date or datetime object
        """
        try:
            if isinstance(input_date, datetime):
                date_obj = input_date.date()
            elif isinstance(input_date, date):
                date_obj = input_date
            else:
                raise ValueError(f"Invalid date input: {input_date}")
            
            features = {
                'day_of_week': date_obj.weekday(),  # 0=Monday, 6=Sunday
                'day_of_month': date_obj.day,
                'month': date_obj.month
            }
            
            logger.debug(f"Extracted calendar features for {date_obj}: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting calendar features from {input_date}: {e}")
            raise ValueError(f"Invalid date input: {input_date}")
    
    def extract_temporal_features(self, input_date: Union[date, datetime]) -> Dict[str, int]:
        """
        Extract temporal features from a date.
        
        Args:
            input_date: Date or datetime object to extract features from
            
        Returns:
            Dictionary containing temporal features:
            - week_of_year: Week number in year (1-53)
            - day_of_year: Day number in year (1-366)
            - is_weekend: 1 if weekend (Saturday/Sunday), 0 otherwise
            
        Raises:
            ValueError: If input_date is not a valid date or datetime object
        """
        try:
            if isinstance(input_date, datetime):
                date_obj = input_date.date()
            elif isinstance(input_date, date):
                date_obj = input_date
            else:
                raise ValueError(f"Invalid date input: {input_date}")
            
            # Calculate week of year using ISO format
            week_of_year = date_obj.isocalendar()[1]
            
            # Check if weekend (Saturday=5, Sunday=6 in weekday())
            weekday = date_obj.weekday()
            is_weekend = 1 if weekday >= 5 else 0
            
            features = {
                'week_of_year': week_of_year,
                'day_of_year': date_obj.timetuple().tm_yday,
                'is_weekend': is_weekend
            }
            
            logger.debug(f"Extracted temporal features for {date_obj}: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting temporal features from {input_date}: {e}")
            raise ValueError(f"Invalid date input: {input_date}")
    
    def extract_holiday_features(self, input_date: Union[date, datetime]) -> Dict[str, int]:
        """
        Extract holiday-related features from a date.
        
        Args:
            input_date: Date or datetime object to extract features from
            
        Returns:
            Dictionary containing holiday features:
            - is_holiday: 1 if US federal holiday, 0 otherwise
            - days_since_last_holiday: Number of days since last holiday
            
        Raises:
            ValueError: If input_date is not a valid date or datetime object
        """
        try:
            if isinstance(input_date, datetime):
                date_obj = input_date.date()
            elif isinstance(input_date, date):
                date_obj = input_date
            else:
                raise ValueError(f"Invalid date input: {input_date}")
            
            # Check if current date is a holiday
            is_holiday = 1 if date_obj in self.us_holidays else 0
            
            # Calculate days since last holiday
            days_since_last_holiday = 0
            if not is_holiday:
                # Look backwards to find the most recent holiday
                from datetime import timedelta
                check_date = date_obj
                while days_since_last_holiday < 365:  # Max 1 year back
                    check_date = check_date - timedelta(days=1)
                    days_since_last_holiday += 1
                    if check_date in self.us_holidays:
                        break
                else:
                    # If no holiday found in past year, set to 365
                    days_since_last_holiday = 365
            
            features = {
                'is_holiday': is_holiday,
                'days_since_last_holiday': days_since_last_holiday
            }
            
            logger.debug(f"Extracted holiday features for {date_obj}: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting holiday features from {input_date}: {e}")
            raise ValueError(f"Invalid date input: {input_date}")
    
    def extract_all_features(self, input_date: Union[date, datetime]) -> Dict[str, int]:
        """
        Extract all features (calendar, temporal, and holiday) from a date.
        
        Args:
            input_date: Date or datetime object to extract features from
            
        Returns:
            Dictionary containing all 8 features combined
            
        Raises:
            ValueError: If input_date is not a valid date or datetime object
        """
        all_features = {}
        
        # Extract all feature types
        all_features.update(self.extract_calendar_features(input_date))
        all_features.update(self.extract_temporal_features(input_date))
        all_features.update(self.extract_holiday_features(input_date))
        
        logger.info(f"Extracted {len(all_features)} features for {input_date}")
        return all_features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Extract features from all dates in a DataFrame.
        
        Args:
            df: DataFrame containing a date column
            date_column: Name of the column containing dates
            
        Returns:
            DataFrame with feature columns added
            
        Raises:
            ValueError: If date_column doesn't exist or contains invalid dates
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        features_list = []
        for idx, row in df.iterrows():
            date_value = row[date_column]
            features = self.extract_all_features(date_value)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted features for {len(features_df)} dates from DataFrame")
        
        return features_df
    
