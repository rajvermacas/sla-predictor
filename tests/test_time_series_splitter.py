"""
Test suite for TimeSeriesSplitter class.
Tests chronological data splitting functionality.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.sla_predictor.time_series_splitter import TimeSeriesSplitter


class TestTimeSeriesSplitter:
    """Test cases for TimeSeriesSplitter functionality."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'day_of_week': [d.weekday() + 1 for d in dates],
            'month': [d.month for d in dates],
            'sla_outcome': [(i % 3) for i in range(len(dates))]  # Some pattern
        })
    
    def test_time_series_splitter_initialization(self):
        """Test TimeSeriesSplitter can be initialized properly."""
        splitter = TimeSeriesSplitter()
        
        assert splitter is not None
        assert hasattr(splitter, 'test_size')
        assert splitter.test_size == 0.2  # Default test size
    
    def test_time_series_splitter_custom_test_size(self):
        """Test TimeSeriesSplitter initialization with custom test size."""
        splitter = TimeSeriesSplitter(test_size=0.3)
        
        assert splitter.test_size == 0.3
    
    def test_split_chronological_data(self, sample_time_series_data):
        """Test splitting data chronologically."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        
        # This should fail initially
        X_train, X_test, y_train, y_test = splitter.split(
            sample_time_series_data.drop(['date', 'sla_outcome'], axis=1),
            sample_time_series_data['sla_outcome'],
            sample_time_series_data['date']
        )
        
        # Check that data is split correctly
        total_size = len(sample_time_series_data)
        expected_train_size = int(total_size * 0.8)
        expected_test_size = total_size - expected_train_size
        
        assert len(X_train) == expected_train_size
        assert len(X_test) == expected_test_size
        assert len(y_train) == expected_train_size
        assert len(y_test) == expected_test_size
    
    def test_chronological_order_preserved(self, sample_time_series_data):
        """Test that chronological order is preserved in split."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        
        X_train, X_test, y_train, y_test = splitter.split(
            sample_time_series_data.drop(['date', 'sla_outcome'], axis=1),
            sample_time_series_data['sla_outcome'],
            sample_time_series_data['date']
        )
        
        # Get the split indices to verify chronological order
        train_end_idx = len(X_train)
        test_start_idx = train_end_idx
        
        # Training data should come before test data chronologically
        train_end_date = sample_time_series_data.iloc[train_end_idx - 1]['date']
        test_start_date = sample_time_series_data.iloc[test_start_idx]['date']
        
        assert train_end_date < test_start_date
    
    def test_split_without_date_column_raises_error(self, sample_time_series_data):
        """Test that splitting without date column raises error."""
        splitter = TimeSeriesSplitter()
        
        with pytest.raises(ValueError, match="Date information is required"):
            splitter.split(
                sample_time_series_data.drop(['date', 'sla_outcome'], axis=1),
                sample_time_series_data['sla_outcome']
            )
    
    def test_split_with_insufficient_data(self):
        """Test splitting with insufficient data raises error."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        
        # Create very small dataset
        small_data = pd.DataFrame({
            'feature1': [1, 2],
            'date': pd.date_range('2023-01-01', periods=2),
            'target': [0, 1]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            splitter.split(
                small_data.drop(['date', 'target'], axis=1),
                small_data['target'],
                small_data['date']
            )
    
    def test_invalid_test_size_raises_error(self):
        """Test that invalid test_size raises error."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            TimeSeriesSplitter(test_size=1.5)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            TimeSeriesSplitter(test_size=-0.1)
    
    def test_split_returns_indices(self, sample_time_series_data):
        """Test that split can return indices instead of data."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        
        # This should fail initially
        train_idx, test_idx = splitter.get_split_indices(
            len(sample_time_series_data)
        )
        
        total_size = len(sample_time_series_data)
        expected_train_size = int(total_size * 0.8)
        
        assert len(train_idx) == expected_train_size
        assert len(test_idx) == total_size - expected_train_size
        assert max(train_idx) < min(test_idx)  # No overlap, chronological order
    
    def test_split_with_custom_date_format(self):
        """Test splitting with different date formats."""
        splitter = TimeSeriesSplitter(test_size=0.3)
        
        # Create data with string dates
        data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Convert date strings to datetime
        dates = pd.to_datetime(data['date'])
        
        X_train, X_test, y_train, y_test = splitter.split(
            data.drop(['date', 'target'], axis=1),
            data['target'],
            dates
        )
        
        assert len(X_train) == 3  # 70% of 5 = 3.5 -> 3
        assert len(X_test) == 2   # 30% of 5 = 1.5 -> 2