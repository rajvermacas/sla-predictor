"""
Time Series Splitting module for SLA Predictor.

This module provides functionality for chronological train/test splitting
to ensure temporal order is preserved in time series analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """
    Splits time series data chronologically for machine learning.
    
    Maintains temporal order by splitting data so that training data
    comes before test data chronologically.
    """
    
    def __init__(self, test_size: float = 0.2):
        """
        Initialize TimeSeriesSplitter.
        
        Args:
            test_size: Proportion of data to use for testing (0.0 to 1.0)
            
        Raises:
            ValueError: If test_size is not between 0 and 1
        """
        if not 0 < test_size < 1:
            logger.error(f"Invalid test_size: {test_size}. Must be between 0 and 1")
            raise ValueError("test_size must be between 0 and 1")
        
        self.test_size = test_size
        logger.info(f"TimeSeriesSplitter initialized with test_size={test_size}")
    
    def get_split_indices(self, data_length: int) -> Tuple[List[int], List[int]]:
        """
        Get indices for chronological train/test split.
        
        Args:
            data_length: Total length of the dataset
            
        Returns:
            Tuple of (train_indices, test_indices)
            
        Raises:
            ValueError: If data is insufficient for splitting
        """
        if data_length < 5:
            logger.error(f"Insufficient data: {data_length} samples (minimum 5 required)")
            raise ValueError("Insufficient data for splitting (minimum 5 samples required)")
        
        # Calculate split point
        train_size = int(data_length * (1 - self.test_size))
        
        # Ensure minimum samples in both sets
        if train_size < 2:
            train_size = 2
        
        test_size = data_length - train_size
        if test_size < 2:
            train_size = data_length - 2
        
        # Generate indices
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, data_length))
        
        logger.info(f"Split indices: {len(train_indices)} train, {len(test_indices)} test")
        return train_indices, test_indices
    
    def split(self, 
              X: pd.DataFrame, 
              y: pd.Series, 
              dates: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split features and target chronologically.
        
        Args:
            X: Feature data
            y: Target variable
            dates: Date information for chronological sorting (optional but recommended)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If data is insufficient or dates are missing when needed
        """
        if dates is None:
            logger.error("Date information is required for chronological splitting")
            raise ValueError("Date information is required for chronological splitting")
        
        if len(X) != len(y) or len(X) != len(dates):
            logger.error(f"Data length mismatch: X={len(X)}, y={len(y)}, dates={len(dates)}")
            raise ValueError("All input data must have the same length")
        
        # Convert dates to datetime if needed
        from datetime import datetime
        if not isinstance(dates.iloc[0], (pd.Timestamp, datetime)):
            dates = pd.to_datetime(dates)
        
        # Create combined dataframe for sorting
        combined_data = pd.DataFrame({
            'date': dates,
            'y': y
        })
        
        # Add feature columns
        for col in X.columns:
            combined_data[col] = X[col].values
        
        # Sort by date to ensure chronological order
        combined_data = combined_data.sort_values('date').reset_index(drop=True)
        
        # Get split indices
        train_indices, test_indices = self.get_split_indices(len(combined_data))
        
        # Split data
        train_data = combined_data.iloc[train_indices]
        test_data = combined_data.iloc[test_indices]
        
        # Extract features and targets
        X_train = train_data.drop(['date', 'y'], axis=1)
        X_test = test_data.drop(['date', 'y'], axis=1)
        y_train = train_data['y']
        y_test = test_data['y']
        
        logger.info(f"Chronological split completed: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test