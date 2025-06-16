"""
Test suite for the SLA Prediction Engine.

This module contains comprehensive tests for the prediction engine that handles
single date predictions using trained models and feature extraction.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, date
import tempfile
import numpy as np
from pathlib import Path

from src.sla_predictor.prediction_engine import PredictionEngine
from src.sla_predictor.model_persistence import ModelPersistence


class TestPredictionEngine(unittest.TestCase):
    """Test cases for the PredictionEngine class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Create sample historical data for feature extraction
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=30, freq='D'),
            'SLA_Outcome': [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 3
        })
    
    def test_prediction_engine_initialization(self):
        """Test that PredictionEngine initializes correctly."""
        # This should fail initially (RED phase)
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=self.sample_data
        )
        
        self.assertIsNotNone(engine)
        self.assertEqual(engine.models_dir, self.models_dir)
        self.assertIsInstance(engine.historical_data, pd.DataFrame)
    
    def test_predict_single_date_basic(self):
        """Test basic single date prediction functionality."""
        # This should fail initially (RED phase)
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=self.sample_data
        )
        
        prediction_date = datetime(2023, 2, 15).date()
        result = engine.predict(prediction_date)
        
        # Should return a clear Yes/No prediction
        self.assertIn(result, ['Yes', 'No'])
    
    def test_predict_single_date_with_confidence(self):
        """Test single date prediction with confidence score."""
        # This should fail initially (RED phase)
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=self.sample_data
        )
        
        prediction_date = datetime(2023, 2, 15).date()
        result = engine.predict(prediction_date, include_confidence=True)
        
        # Should return dict with prediction and confidence
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn(result['prediction'], ['Yes', 'No'])
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_predict_with_actual_model_loading(self):
        """Test prediction with actual model loading."""
        # Mock model persistence to return a trained model
        with patch.object(ModelPersistence, 'load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_load.return_value = mock_model
            
            engine = PredictionEngine(
                models_dir=self.models_dir,
                historical_data=self.sample_data
            )
            
            prediction_date = datetime(2023, 2, 15).date()
            result = engine.predict_with_model(prediction_date, model_name='logistic_regression')
            
            self.assertIn(result, ['Yes', 'No'])
    
    def test_extract_features_for_prediction(self):
        """Test feature extraction for a prediction date."""
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=self.sample_data
        )
        
        prediction_date = datetime(2023, 2, 15).date()
        features = engine.extract_features(prediction_date)
        
        # Should return a DataFrame with extracted features
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)  # Single row for single date
        
        # Should contain expected feature columns
        expected_features = [
            'day_of_week', 'day_of_month', 'month', 'week_of_year',
            'day_of_year', 'is_weekend', 'is_holiday', 'days_since_last_holiday',
            'previous_day_sla_missed', 'previous_day_sla_met',
            'consecutive_misses', 'rolling_7day_miss_rate'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_predict_with_invalid_date(self):
        """Test prediction with invalid date input."""
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=self.sample_data
        )
        
        # Test with None
        with self.assertRaises(ValueError):
            engine.predict(None)
        
        # Test with string that can't be converted
        with self.assertRaises(ValueError):
            engine.predict("invalid-date")
    
    def test_predict_with_no_historical_data(self):
        """Test prediction when no historical data is available."""
        empty_data = pd.DataFrame(columns=['Date', 'SLA_Outcome'])
        
        engine = PredictionEngine(
            models_dir=self.models_dir,
            historical_data=empty_data
        )
        
        prediction_date = datetime(2023, 2, 15).date()
        # Should handle gracefully with defaults
        result = engine.predict(prediction_date)
        
        self.assertIn(result, ['Yes', 'No'])


if __name__ == '__main__':
    unittest.main()