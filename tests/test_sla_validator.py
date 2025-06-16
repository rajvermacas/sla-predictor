"""
Test-First SLA Outcome Validation Tests
RED phase: Write failing tests for binary outcome validation
"""
import pytest
import pandas as pd
import numpy as np

from src.sla_predictor.sla_validator import SLAValidator


class TestSLAValidator:
    """Test class for SLA outcome validation functionality"""
    
    def test_sla_validator_initialization(self):
        """Test SLAValidator can be initialized"""
        validator = SLAValidator()
        assert validator is not None
    
    def test_validate_binary_outcome_valid_integers(self):
        """Test validating valid binary integers (0, 1)"""
        validator = SLAValidator()
        
        assert validator.validate_binary_outcome(0) == 0
        assert validator.validate_binary_outcome(1) == 1
    
    def test_validate_binary_outcome_valid_strings(self):
        """Test validating valid string representations"""
        validator = SLAValidator()
        
        # Test string numbers
        assert validator.validate_binary_outcome("0") == 0
        assert validator.validate_binary_outcome("1") == 1
        
        # Test Yes/No
        assert validator.validate_binary_outcome("Yes") == 1
        assert validator.validate_binary_outcome("No") == 0
        assert validator.validate_binary_outcome("yes") == 1
        assert validator.validate_binary_outcome("no") == 0
        
        # Test True/False
        assert validator.validate_binary_outcome("True") == 1
        assert validator.validate_binary_outcome("False") == 0
        assert validator.validate_binary_outcome("true") == 1
        assert validator.validate_binary_outcome("false") == 0
    
    def test_validate_binary_outcome_valid_booleans(self):
        """Test validating boolean values"""
        validator = SLAValidator()
        
        assert validator.validate_binary_outcome(True) == 1
        assert validator.validate_binary_outcome(False) == 0
    
    def test_validate_binary_outcome_invalid_values_raise_error(self):
        """Test invalid values raise ValueError"""
        validator = SLAValidator()
        
        # Invalid integers
        with pytest.raises(ValueError, match="Invalid SLA outcome"):
            validator.validate_binary_outcome(2)
        
        with pytest.raises(ValueError, match="Invalid SLA outcome"):
            validator.validate_binary_outcome(-1)
        
        # Invalid strings
        with pytest.raises(ValueError, match="Invalid SLA outcome"):
            validator.validate_binary_outcome("invalid")
        
        with pytest.raises(ValueError, match="Invalid SLA outcome"):
            validator.validate_binary_outcome("maybe")
        
        # None value
        with pytest.raises(ValueError, match="SLA outcome cannot be None"):
            validator.validate_binary_outcome(None)
        
        # Empty string
        with pytest.raises(ValueError, match="SLA outcome cannot be empty"):
            validator.validate_binary_outcome("")
    
    def test_validate_dataframe_sla_column(self):
        """Test validating SLA outcome column in DataFrame"""
        validator = SLAValidator()
        
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'SLA_Outcome': [1, 0, "Yes", "No"]
        })
        
        result_df = validator.validate_dataframe_sla_outcomes(df, 'SLA_Outcome')
        
        # Check that all values are converted to integers
        assert result_df['SLA_Outcome'].dtype == 'int64'
        
        # Check specific values
        expected_outcomes = [1, 0, 1, 0]
        assert list(result_df['SLA_Outcome']) == expected_outcomes
    
    def test_validate_dataframe_missing_column_raises_error(self):
        """Test validating missing SLA column raises ValueError"""
        validator = SLAValidator()
        
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'Wrong_Column': [1, 0]
        })
        
        with pytest.raises(ValueError, match="SLA outcome column 'SLA_Outcome' not found"):
            validator.validate_dataframe_sla_outcomes(df, 'SLA_Outcome')
    
    def test_validate_dataframe_invalid_outcomes_raises_error(self):
        """Test DataFrame with invalid SLA outcomes raises ValueError"""
        validator = SLAValidator()
        
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'SLA_Outcome': [1, 0, "invalid"]
        })
        
        with pytest.raises(ValueError, match="Invalid SLA outcome found"):
            validator.validate_dataframe_sla_outcomes(df, 'SLA_Outcome')
    
    def test_convert_flexible_formats(self):
        """Test converting various flexible input formats"""
        validator = SLAValidator()
        
        # Test mixed case
        assert validator.validate_binary_outcome("YES") == 1
        assert validator.validate_binary_outcome("NO") == 0
        assert validator.validate_binary_outcome("TRUE") == 1
        assert validator.validate_binary_outcome("FALSE") == 0
        
        # Test with whitespace
        assert validator.validate_binary_outcome(" 1 ") == 1
        assert validator.validate_binary_outcome(" 0 ") == 0
        assert validator.validate_binary_outcome(" Yes ") == 1
        assert validator.validate_binary_outcome(" No ") == 0
    
    def test_get_statistics(self):
        """Test getting statistics for SLA outcomes"""
        validator = SLAValidator()
        
        outcomes = [1, 0, 1, 1, 0, 1, 0, 0]
        stats = validator.get_statistics(outcomes)
        
        assert stats['total_count'] == 8
        assert stats['sla_met_count'] == 4
        assert stats['sla_missed_count'] == 4
        assert stats['sla_met_percentage'] == 50.0
        assert stats['sla_missed_percentage'] == 50.0
    
    def test_get_statistics_empty_list(self):
        """Test statistics with empty outcomes list"""
        validator = SLAValidator()
        
        with pytest.raises(ValueError, match="Outcomes list cannot be empty"):
            validator.get_statistics([])