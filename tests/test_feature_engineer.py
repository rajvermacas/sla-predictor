"""
Test suite for FeatureEngineer class.

This module tests the feature engineering functionality including:
- Calendar features (day_of_week, day_of_month, month)
- Temporal features (week_of_year, day_of_year, is_weekend)
- Holiday integration (is_holiday, days_since_last_holiday)
"""

import pytest
import pandas as pd
from datetime import datetime, date
from src.sla_predictor.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer can be initialized."""
        feature_engineer = FeatureEngineer()
        assert feature_engineer is not None
    
    def test_extract_calendar_features_basic(self):
        """Test extraction of basic calendar features."""
        feature_engineer = FeatureEngineer()
        test_date = date(2023, 6, 15)  # Thursday, June 15, 2023
        
        features = feature_engineer.extract_calendar_features(test_date)
        
        assert features['day_of_week'] == 3  # Thursday (0=Monday)
        assert features['day_of_month'] == 15
        assert features['month'] == 6
    
    def test_extract_calendar_features_datetime_input(self):
        """Test calendar features with datetime input."""
        feature_engineer = FeatureEngineer()
        test_datetime = datetime(2023, 12, 25, 14, 30)  # Christmas 2023
        
        features = feature_engineer.extract_calendar_features(test_datetime)
        
        assert features['day_of_week'] == 0  # Monday
        assert features['day_of_month'] == 25
        assert features['month'] == 12
    
    def test_extract_temporal_features_basic(self):
        """Test extraction of temporal features."""
        feature_engineer = FeatureEngineer()
        test_date = date(2023, 6, 15)  # Thursday, June 15, 2023
        
        features = feature_engineer.extract_temporal_features(test_date)
        
        assert features['week_of_year'] == 24  # 15th week of 2023
        assert features['day_of_year'] == 166  # 166th day of 2023
        assert features['is_weekend'] == 0  # Thursday is not weekend
    
    def test_extract_temporal_features_weekend(self):
        """Test temporal features for weekend dates."""
        feature_engineer = FeatureEngineer()
        saturday = date(2023, 6, 17)  # Saturday
        sunday = date(2023, 6, 18)    # Sunday
        
        saturday_features = feature_engineer.extract_temporal_features(saturday)
        sunday_features = feature_engineer.extract_temporal_features(sunday)
        
        assert saturday_features['is_weekend'] == 1
        assert sunday_features['is_weekend'] == 1
    
    def test_extract_holiday_features_non_holiday(self):
        """Test holiday features for non-holiday dates."""
        feature_engineer = FeatureEngineer()
        test_date = date(2023, 6, 15)  # Regular Thursday
        
        features = feature_engineer.extract_holiday_features(test_date)
        
        assert features['is_holiday'] == 0
        assert isinstance(features['days_since_last_holiday'], int)
        assert features['days_since_last_holiday'] >= 0
    
    def test_extract_holiday_features_independence_day(self):
        """Test holiday features for Independence Day."""
        feature_engineer = FeatureEngineer()
        july_4th = date(2023, 7, 4)  # Independence Day 2023
        
        features = feature_engineer.extract_holiday_features(july_4th)
        
        assert features['is_holiday'] == 1
        assert features['days_since_last_holiday'] == 0
    
    def test_extract_all_features_comprehensive(self):
        """Test extraction of all features at once."""
        feature_engineer = FeatureEngineer()
        test_date = date(2023, 6, 15)
        
        all_features = feature_engineer.extract_all_features(test_date)
        
        # Check calendar features
        assert 'day_of_week' in all_features
        assert 'day_of_month' in all_features
        assert 'month' in all_features
        
        # Check temporal features
        assert 'week_of_year' in all_features
        assert 'day_of_year' in all_features
        assert 'is_weekend' in all_features
        
        # Check holiday features
        assert 'is_holiday' in all_features
        assert 'days_since_last_holiday' in all_features
        
        # Should have 8 features total
        assert len(all_features) == 8
    
    def test_extract_features_from_dataframe(self):
        """Test feature extraction from a pandas DataFrame."""
        feature_engineer = FeatureEngineer()
        
        # Create test DataFrame
        dates = [date(2023, 6, 15), date(2023, 7, 4), date(2023, 12, 25)]
        df = pd.DataFrame({'Date': dates})
        
        features_df = feature_engineer.extract_features_from_dataframe(df, 'Date')
        
        assert len(features_df) == 3
        assert len(features_df.columns) == 8  # 8 feature columns
        
        # Check first row (regular Thursday)
        assert features_df.iloc[0]['day_of_week'] == 3
        assert features_df.iloc[0]['is_weekend'] == 0
        
        # Check second row (July 4th)
        assert features_df.iloc[1]['is_holiday'] == 1
    
    def test_extract_features_leap_year_edge_case(self):
        """Test feature extraction handles leap year correctly."""
        feature_engineer = FeatureEngineer()
        leap_day = date(2024, 2, 29)  # Leap day
        
        temporal_features = feature_engineer.extract_temporal_features(leap_day)
        calendar_features = feature_engineer.extract_calendar_features(leap_day)
        
        assert temporal_features['day_of_year'] == 60  # 60th day of leap year
        assert calendar_features['month'] == 2
        assert calendar_features['day_of_month'] == 29
    
    def test_invalid_date_input_raises_error(self):
        """Test that invalid date input raises appropriate error."""
        feature_engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="Invalid date"):
            feature_engineer.extract_calendar_features("invalid_date")
        
        with pytest.raises(ValueError, match="Invalid date"):
            feature_engineer.extract_temporal_features(None)