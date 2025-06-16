"""
Tests for Advanced Feature Engineering with Lag Variables.

This module tests lag-based features, consecutive analysis, and rolling statistics
for the SLA prediction system.
"""

import pytest
import pandas as pd
from datetime import date, timedelta
from src.sla_predictor.advanced_features import AdvancedFeatureEngineer


class TestAdvancedFeatureEngineer:
    """Test suite for AdvancedFeatureEngineer class."""

    def test_extract_lag_features_basic(self):
        """Test basic lag feature extraction with previous day SLA data."""
        engineer = AdvancedFeatureEngineer()
        
        # Create sample data with historical SLA outcomes
        historical_data = pd.DataFrame({
            'Date': [date(2023, 6, 13), date(2023, 6, 14), date(2023, 6, 15)],
            'SLA_Outcome': [1, 0, 1]  # Met, Missed, Met
        })
        
        # Test extracting lag features for June 16, 2023
        test_date = date(2023, 6, 16)
        features = engineer.extract_lag_features(test_date, historical_data)
        
        # Previous day (June 15) had SLA outcome of 1 (met)
        assert features['previous_day_sla_missed'] == 0  # 0 = not missed
        assert features['previous_day_sla_met'] == 1    # 1 = met
    
    def test_extract_lag_features_missing_data(self):
        """Test lag features when no historical data is available."""
        engineer = AdvancedFeatureEngineer()
        
        # Empty historical data
        historical_data = pd.DataFrame({
            'Date': [],
            'SLA_Outcome': []
        })
        
        test_date = date(2023, 6, 16)
        features = engineer.extract_lag_features(test_date, historical_data)
        
        # Should default to 0 when no data available
        assert features['previous_day_sla_missed'] == 0
        assert features['previous_day_sla_met'] == 0
    
    def test_extract_consecutive_features_basic(self):
        """Test consecutive miss counting functionality."""
        engineer = AdvancedFeatureEngineer()
        
        # Create data with consecutive misses
        historical_data = pd.DataFrame({
            'Date': [date(2023, 6, 12), date(2023, 6, 13), 
                    date(2023, 6, 14), date(2023, 6, 15)],
            'SLA_Outcome': [1, 0, 0, 0]  # Met, Miss, Miss, Miss
        })
        
        test_date = date(2023, 6, 16)
        features = engineer.extract_consecutive_features(test_date, historical_data)
        
        # Should count 3 consecutive misses leading up to test date
        assert features['consecutive_misses'] == 3
    
    def test_extract_consecutive_features_no_misses(self):
        """Test consecutive features when there are no recent misses."""
        engineer = AdvancedFeatureEngineer()
        
        # All SLAs met recently
        historical_data = pd.DataFrame({
            'Date': [date(2023, 6, 13), date(2023, 6, 14), date(2023, 6, 15)],
            'SLA_Outcome': [1, 1, 1]  # All met
        })
        
        test_date = date(2023, 6, 16)
        features = engineer.extract_consecutive_features(test_date, historical_data)
        
        assert features['consecutive_misses'] == 0
    
    def test_extract_rolling_statistics_basic(self):
        """Test 7-day rolling miss rate calculation."""
        engineer = AdvancedFeatureEngineer()
        
        # Create 7 days of data with 3 misses out of 7 days
        dates = [date(2023, 6, 9) + timedelta(days=i) for i in range(7)]
        historical_data = pd.DataFrame({
            'Date': dates,
            'SLA_Outcome': [1, 0, 1, 0, 1, 0, 1]  # 3 misses, 4 met
        })
        
        test_date = date(2023, 6, 16)
        features = engineer.extract_rolling_statistics(test_date, historical_data, window_size=7)
        
        # Miss rate should be 3/7 ≈ 0.43 (rounded to 2 decimal places = 43)
        expected_miss_rate = int(round(3/7 * 100))  # Convert to percentage integer
        assert features['rolling_7day_miss_rate'] == expected_miss_rate
    
    def test_extract_rolling_statistics_insufficient_data(self):
        """Test rolling statistics with insufficient historical data."""
        engineer = AdvancedFeatureEngineer()
        
        # Only 3 days of data when requesting 7-day window
        historical_data = pd.DataFrame({
            'Date': [date(2023, 6, 13), date(2023, 6, 14), date(2023, 6, 15)],
            'SLA_Outcome': [1, 0, 1]
        })
        
        test_date = date(2023, 6, 16)
        features = engineer.extract_rolling_statistics(test_date, historical_data, window_size=7)
        
        # Should calculate miss rate based on available data (1 miss out of 3 = 33%)
        expected_miss_rate = int(round(1/3 * 100))
        assert features['rolling_7day_miss_rate'] == expected_miss_rate
    
    def test_extract_all_advanced_features(self):
        """Test extracting all advanced features together."""
        engineer = AdvancedFeatureEngineer()
        
        # Create comprehensive historical data
        historical_data = pd.DataFrame({
            'Date': [date(2023, 6, 9) + timedelta(days=i) for i in range(7)],
            'SLA_Outcome': [1, 0, 0, 1, 1, 0, 1]  # Mix of met/missed
        })
        
        test_date = date(2023, 6, 16)
        features = engineer.extract_all_advanced_features(test_date, historical_data)
        
        # Should contain all feature types
        expected_keys = ['previous_day_sla_missed', 'previous_day_sla_met', 
                        'consecutive_misses', 'rolling_7day_miss_rate']
        
        for key in expected_keys:
            assert key in features
        
        # Verify feature values are reasonable
        assert isinstance(features['previous_day_sla_missed'], int)
        assert isinstance(features['consecutive_misses'], int)
        assert isinstance(features['rolling_7day_miss_rate'], int)
        assert 0 <= features['rolling_7day_miss_rate'] <= 100
    
    def test_dataframe_integration(self):
        """Test advanced feature extraction on DataFrame."""
        engineer = AdvancedFeatureEngineer()
        
        # Historical data for feature calculation
        historical_data = pd.DataFrame({
            'Date': [date(2023, 6, 10) + timedelta(days=i) for i in range(10)],
            'SLA_Outcome': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        })
        
        # New data to add features to
        new_data = pd.DataFrame({
            'Date': [date(2023, 6, 21), date(2023, 6, 22)],
            'SLA_Outcome': [1, 0]
        })
        
        features_df = engineer.extract_features_from_dataframe(
            new_data, historical_data, date_column='Date'
        )
        
        # Should return DataFrame with all advanced features
        expected_columns = ['previous_day_sla_missed', 'previous_day_sla_met',
                           'consecutive_misses', 'rolling_7day_miss_rate']
        
        for col in expected_columns:
            assert col in features_df.columns
        
        assert len(features_df) == len(new_data)
    
    def test_invalid_date_input(self):
        """Test error handling for invalid date inputs."""
        engineer = AdvancedFeatureEngineer()
        
        historical_data = pd.DataFrame({
            'Date': [date(2023, 6, 15)],
            'SLA_Outcome': [1]
        })
        
        with pytest.raises(ValueError):
            engineer.extract_lag_features("invalid_date", historical_data)
    
    def test_empty_historical_data(self):
        """Test handling of completely empty historical data."""
        engineer = AdvancedFeatureEngineer()
        
        empty_data = pd.DataFrame({'Date': [], 'SLA_Outcome': []})
        
        test_date = date(2023, 6, 16)
        features = engineer.extract_all_advanced_features(test_date, empty_data)
        
        # All features should default to 0
        assert features['previous_day_sla_missed'] == 0
        assert features['previous_day_sla_met'] == 0
        assert features['consecutive_misses'] == 0
        assert features['rolling_7day_miss_rate'] == 0