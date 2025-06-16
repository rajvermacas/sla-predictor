"""
Advanced Feature Engineering Module for SLA Predictor.

This module provides advanced feature extraction functionality including
lag variables, consecutive analysis, and rolling statistics for SLA prediction.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Union
import pandas as pd


logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering class for extracting lag-based features.
    
    This class provides methods to extract advanced features from historical
    SLA data including lag variables, consecutive patterns, and rolling statistics.
    """
    
    def __init__(self):
        """Initialize AdvancedFeatureEngineer."""
        logger.info("AdvancedFeatureEngineer initialized")
    
    def extract_lag_features(self, target_date: Union[date, datetime], 
                           historical_data: pd.DataFrame) -> Dict[str, int]:
        """
        Extract lag-based features from historical data.
        
        Args:
            target_date: Date to extract features for
            historical_data: DataFrame with 'Date' and 'SLA_Outcome' columns
            
        Returns:
            Dictionary containing lag features:
            - previous_day_sla_missed: 1 if previous day missed SLA, 0 otherwise
            - previous_day_sla_met: 1 if previous day met SLA, 0 otherwise
            
        Raises:
            ValueError: If target_date is not a valid date or datetime object
        """
        try:
            if isinstance(target_date, datetime):
                date_obj = target_date.date()
            elif isinstance(target_date, date):
                date_obj = target_date
            else:
                raise ValueError(f"Invalid date input: {target_date}")
            
            # Default values when no data available
            features = {
                'previous_day_sla_missed': 0,
                'previous_day_sla_met': 0
            }
            
            if len(historical_data) == 0:
                logger.debug(f"No historical data available for {date_obj}")
                return features
            
            # Find previous day's SLA outcome
            previous_day = date_obj - timedelta(days=1)
            
            # Filter data for previous day
            prev_day_data = historical_data[historical_data['Date'] == previous_day]
            
            if len(prev_day_data) > 0:
                sla_outcome = prev_day_data.iloc[0]['SLA_Outcome']
                if sla_outcome == 0:  # Missed
                    features['previous_day_sla_missed'] = 1
                    features['previous_day_sla_met'] = 0
                else:  # Met
                    features['previous_day_sla_missed'] = 0
                    features['previous_day_sla_met'] = 1
            
            logger.debug(f"Extracted lag features for {date_obj}: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting lag features for {target_date}: {e}")
            raise ValueError(f"Invalid date input: {target_date}")
    
    def extract_consecutive_features(self, target_date: Union[date, datetime],
                                   historical_data: pd.DataFrame) -> Dict[str, int]:
        """
        Extract consecutive miss patterns from historical data.
        
        Args:
            target_date: Date to extract features for
            historical_data: DataFrame with 'Date' and 'SLA_Outcome' columns
            
        Returns:
            Dictionary containing consecutive features:
            - consecutive_misses: Number of consecutive SLA misses leading up to target date
            
        Raises:
            ValueError: If target_date is not a valid date or datetime object
        """
        try:
            if isinstance(target_date, datetime):
                date_obj = target_date.date()
            elif isinstance(target_date, date):
                date_obj = target_date
            else:
                raise ValueError(f"Invalid date input: {target_date}")
            
            features = {'consecutive_misses': 0}
            
            if len(historical_data) == 0:
                logger.debug(f"No historical data available for {date_obj}")
                return features
            
            # Sort historical data by date in descending order
            sorted_data = historical_data.sort_values('Date', ascending=False)
            
            # Count consecutive misses from most recent date backwards
            consecutive_count = 0
            check_date = date_obj - timedelta(days=1)
            
            while True:
                day_data = sorted_data[sorted_data['Date'] == check_date]
                
                if len(day_data) == 0:
                    # No data for this date, stop counting
                    break
                
                sla_outcome = day_data.iloc[0]['SLA_Outcome']
                if sla_outcome == 0:  # Missed
                    consecutive_count += 1
                    check_date = check_date - timedelta(days=1)
                else:
                    # SLA met, break the streak
                    break
            
            features['consecutive_misses'] = consecutive_count
            
            logger.debug(f"Extracted consecutive features for {date_obj}: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting consecutive features for {target_date}: {e}")
            raise ValueError(f"Invalid date input: {target_date}")
    
    def extract_rolling_statistics(self, target_date: Union[date, datetime],
                                 historical_data: pd.DataFrame,
                                 window_size: int = 7) -> Dict[str, int]:
        """
        Extract rolling statistics from historical data.
        
        Args:
            target_date: Date to extract features for
            historical_data: DataFrame with 'Date' and 'SLA_Outcome' columns
            window_size: Number of days to include in rolling window
            
        Returns:
            Dictionary containing rolling statistics:
            - rolling_7day_miss_rate: Miss rate percentage over rolling window
            
        Raises:
            ValueError: If target_date is not a valid date or datetime object
        """
        try:
            if isinstance(target_date, datetime):
                date_obj = target_date.date()
            elif isinstance(target_date, date):
                date_obj = target_date
            else:
                raise ValueError(f"Invalid date input: {target_date}")
            
            features = {'rolling_7day_miss_rate': 0}
            
            if len(historical_data) == 0:
                logger.debug(f"No historical data available for {date_obj}")
                return features
            
            # Define the rolling window (ending day before target date)
            end_date = date_obj - timedelta(days=1)
            start_date = end_date - timedelta(days=window_size - 1)
            
            # Filter data within the rolling window
            window_data = historical_data[
                (historical_data['Date'] >= start_date) & 
                (historical_data['Date'] <= end_date)
            ]
            
            if len(window_data) == 0:
                logger.debug(f"No data in rolling window for {date_obj}")
                return features
            
            # Calculate miss rate
            total_days = len(window_data)
            missed_days = len(window_data[window_data['SLA_Outcome'] == 0])
            miss_rate = (missed_days / total_days) * 100
            
            features['rolling_7day_miss_rate'] = int(round(miss_rate))
            
            logger.debug(f"Extracted rolling statistics for {date_obj}: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting rolling statistics for {target_date}: {e}")
            raise ValueError(f"Invalid date input: {target_date}")
    
    def extract_all_advanced_features(self, target_date: Union[date, datetime],
                                    historical_data: pd.DataFrame) -> Dict[str, int]:
        """
        Extract all advanced features (lag, consecutive, rolling) from historical data.
        
        Args:
            target_date: Date to extract features for
            historical_data: DataFrame with 'Date' and 'SLA_Outcome' columns
            
        Returns:
            Dictionary containing all advanced features combined
            
        Raises:
            ValueError: If target_date is not a valid date or datetime object
        """
        all_features = {}
        
        # Extract all feature types
        all_features.update(self.extract_lag_features(target_date, historical_data))
        all_features.update(self.extract_consecutive_features(target_date, historical_data))
        all_features.update(self.extract_rolling_statistics(target_date, historical_data))
        
        logger.info(f"Extracted {len(all_features)} advanced features for {target_date}")
        return all_features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                      historical_data: pd.DataFrame,
                                      date_column: str) -> pd.DataFrame:
        """
        Extract advanced features from all dates in a DataFrame.
        
        Args:
            df: DataFrame containing dates to extract features for
            historical_data: DataFrame with historical SLA data
            date_column: Name of the column containing dates
            
        Returns:
            DataFrame with advanced feature columns added
            
        Raises:
            ValueError: If date_column doesn't exist or contains invalid dates
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        features_list = []
        for idx, row in df.iterrows():
            date_value = row[date_column]
            features = self.extract_all_advanced_features(date_value, historical_data)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted advanced features for {len(features_df)} dates from DataFrame")
        
        return features_df