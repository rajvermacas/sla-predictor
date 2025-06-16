"""
Integration tests for the complete SLA Predictor workflow.

This module contains end-to-end tests that validate the complete
prediction pipeline from data loading to CLI output.
"""

import unittest
import tempfile
from pathlib import Path
from datetime import date
import pandas as pd
from unittest.mock import patch, Mock

from src.sla_predictor.prediction_engine import PredictionEngine
from src.sla_predictor.cli_interface import CLIInterface
from src.sla_predictor.data_loader import DataLoader
from src.sla_predictor.model_trainer import ModelTrainer
from src.sla_predictor.model_persistence import ModelPersistence


class TestIntegration(unittest.TestCase):
    """Integration test cases for complete workflow."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Create comprehensive sample data
        self.data_file = Path(self.temp_dir) / "sla_data.csv"
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample historical SLA data for testing."""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        outcomes = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 6  # 60% success rate
        
        data = pd.DataFrame({
            'Date': dates,
            'SLA_Outcome': outcomes
        })
        
        data.to_csv(self.data_file, index=False)
    
    def test_end_to_end_prediction_workflow(self):
        """Test complete workflow from data loading to prediction."""
        # Load data
        data_loader = DataLoader()
        historical_data = data_loader.load_csv(self.data_file)
        
        # Initialize prediction engine
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=historical_data
        )
        
        # Test feature extraction
        prediction_date = date(2023, 3, 15)
        features = engine.extract_features(prediction_date)
        
        # Verify features are extracted correctly
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)
        
        # Verify all expected feature columns are present
        expected_features = [
            'day_of_week', 'day_of_month', 'month', 'week_of_year',
            'day_of_year', 'is_weekend', 'is_holiday', 'days_since_last_holiday',
            'previous_day_sla_missed', 'previous_day_sla_met',
            'consecutive_misses', 'rolling_7day_miss_rate'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Test basic prediction
        result = engine.predict(prediction_date)
        self.assertIn(result, ['Yes', 'No'])
        
        # Test prediction with confidence
        result_with_confidence = engine.predict(prediction_date, include_confidence=True)
        self.assertIsInstance(result_with_confidence, dict)
        self.assertIn('prediction', result_with_confidence)
        self.assertIn('confidence', result_with_confidence)
    
    @patch('src.sla_predictor.model_persistence.ModelPersistence.load_model')
    def test_cli_interface_integration(self, mock_load_model):
        """Test CLI interface integration with mocked model."""
        # Mock a trained model
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_load_model.return_value = mock_model
        
        # Test CLI with specific model
        cli = CLIInterface()
        
        with patch('sys.stdout') as mock_stdout:
            cli.run([
                '--date', '2023-03-15',
                '--data', str(self.data_file),
                '--models-dir', str(self.models_dir),
                '--model', 'logistic_regression'
            ])
            
        # Verify model was called for loading
        mock_load_model.assert_called_once_with('logistic_regression')
    
    def test_feature_consistency_across_dates(self):
        """Test that feature extraction is consistent across different dates."""
        data_loader = DataLoader()
        historical_data = data_loader.load_csv(self.data_file)
        
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=historical_data
        )
        
        # Test multiple dates
        test_dates = [
            date(2023, 3, 15),  # Wednesday
            date(2023, 3, 18),  # Saturday (weekend)
            date(2023, 3, 20),  # Monday
        ]
        
        for test_date in test_dates:
            features = engine.extract_features(test_date)
            
            # Verify feature structure is consistent
            self.assertEqual(len(features), 1)
            self.assertGreaterEqual(len(features.columns), 12)
            
            # Verify no NaN values (should have defaults)
            self.assertFalse(features.isnull().any().any())
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        data_loader = DataLoader()
        historical_data = data_loader.load_csv(self.data_file)
        
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=historical_data
        )
        
        # Test invalid date input
        with self.assertRaises(ValueError):
            engine.predict(None)
        
        # Test invalid date string
        with self.assertRaises(ValueError):
            engine.predict("invalid-date")
        
        # CLI should handle errors gracefully
        cli = CLIInterface()
        
        with self.assertRaises(SystemExit):
            cli.parse_arguments(['--date', 'invalid-date'])


if __name__ == '__main__':
    unittest.main()