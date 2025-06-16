"""
Test-First Date Standardization Tests
RED phase: Write failing tests for date format validation and parsing
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

from src.sla_predictor.date_processor import DateProcessor


class TestDateProcessor:
    """Test class for date processing functionality"""
    
    def test_date_processor_initialization(self):
        """Test DateProcessor can be initialized"""
        processor = DateProcessor()
        assert processor is not None
    
    def test_standardize_date_string_valid_format(self):
        """Test standardizing valid date string (YYYY-MM-DD)"""
        processor = DateProcessor()
        result = processor.standardize_date("2023-01-01")
        assert result == datetime(2023, 1, 1).date()
    
    def test_standardize_date_string_alternative_formats(self):
        """Test standardizing alternative date formats"""
        processor = DateProcessor()
        
        # Test MM/DD/YYYY format
        result = processor.standardize_date("01/01/2023")
        assert result == datetime(2023, 1, 1).date()
        
        # Test MM-DD-YYYY format
        result = processor.standardize_date("01-01-2023")
        assert result == datetime(2023, 1, 1).date()
        
        # Test DD/MM/YYYY format
        result = processor.standardize_date("01/01/2023", format_hint="DD/MM/YYYY")
        assert result == datetime(2023, 1, 1).date()
    
    def test_standardize_date_invalid_format_raises_error(self):
        """Test invalid date format raises ValueError"""
        processor = DateProcessor()
        
        with pytest.raises(ValueError, match="Invalid date format"):
            processor.standardize_date("invalid-date")
    
    def test_standardize_date_empty_string_raises_error(self):
        """Test empty date string raises ValueError"""
        processor = DateProcessor()
        
        with pytest.raises(ValueError, match="Empty date string"):
            processor.standardize_date("")
    
    def test_standardize_date_none_raises_error(self):
        """Test None date raises ValueError"""
        processor = DateProcessor()
        
        with pytest.raises(ValueError, match="Date cannot be None"):
            processor.standardize_date(None)
    
    def test_standardize_dataframe_date_column(self):
        """Test standardizing date column in DataFrame"""
        processor = DateProcessor()
        
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'SLA_Outcome': [1, 0, 1]
        })
        
        result_df = processor.standardize_dataframe_dates(df, 'Date')
        
        # Check that Date column is now datetime type
        assert pd.api.types.is_datetime64_any_dtype(result_df['Date'])
        
        # Check specific values
        expected_dates = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']), name='Date')
        pd.testing.assert_series_equal(result_df['Date'], expected_dates)
    
    def test_standardize_dataframe_missing_column_raises_error(self):
        """Test standardizing missing date column raises ValueError"""
        processor = DateProcessor()
        
        df = pd.DataFrame({
            'Wrong_Column': ['2023-01-01', '2023-01-02'],
            'SLA_Outcome': [1, 0]
        })
        
        with pytest.raises(ValueError, match="Date column 'Date' not found"):
            processor.standardize_dataframe_dates(df, 'Date')
    
    def test_standardize_dataframe_invalid_dates_raises_error(self):
        """Test DataFrame with invalid dates raises ValueError"""
        processor = DateProcessor()
        
        df = pd.DataFrame({
            'Date': ['2023-01-01', 'invalid-date', '2023-01-03'],
            'SLA_Outcome': [1, 0, 1]
        })
        
        with pytest.raises(ValueError, match="Invalid date found"):
            processor.standardize_dataframe_dates(df, 'Date')
    
    def test_validate_date_range(self):
        """Test date range validation"""
        processor = DateProcessor()
        
        # Valid date range
        result = processor.validate_date_range(
            datetime(2023, 1, 1).date(),
            datetime(2020, 1, 1).date(),
            datetime(2025, 1, 1).date()
        )
        assert result is True
        
        # Date too early
        with pytest.raises(ValueError, match="Date .* is before minimum allowed date"):
            processor.validate_date_range(
                datetime(2019, 1, 1).date(),
                datetime(2020, 1, 1).date(),
                datetime(2025, 1, 1).date()
            )
        
        # Date too late
        with pytest.raises(ValueError, match="Date .* is after maximum allowed date"):
            processor.validate_date_range(
                datetime(2026, 1, 1).date(),
                datetime(2020, 1, 1).date(),
                datetime(2025, 1, 1).date()
            )