"""
Test suite for ModelTrainer class.
Tests model training functionality with different scikit-learn algorithms.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.sla_predictor.model_trainer import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer functionality."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing."""
        return pd.DataFrame({
            'day_of_week': [1, 2, 3, 4, 5, 1, 2],
            'day_of_month': [15, 16, 17, 18, 19, 20, 21],
            'month': [6, 6, 6, 6, 6, 6, 6],
            'is_weekend': [0, 0, 0, 0, 0, 0, 0],
            'is_holiday': [0, 0, 0, 0, 0, 0, 0],
            'sla_outcome': [1, 0, 1, 1, 0, 1, 0]
        })
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer can be initialized properly."""
        trainer = ModelTrainer()
        
        assert trainer is not None
        assert hasattr(trainer, 'models')
        assert isinstance(trainer.models, dict)
        assert len(trainer.models) == 0
    
    def test_train_logistic_regression_model(self, sample_training_data):
        """Test training a logistic regression model."""
        trainer = ModelTrainer()
        
        # This should fail initially
        model = trainer.train_logistic_regression(
            sample_training_data.drop('sla_outcome', axis=1),
            sample_training_data['sla_outcome']
        )
        
        assert model is not None
        assert isinstance(model, LogisticRegression)
        assert 'logistic_regression' in trainer.models
    
    def test_train_decision_tree_model(self, sample_training_data):
        """Test training a decision tree model.""" 
        trainer = ModelTrainer()
        
        # This should fail initially
        model = trainer.train_decision_tree(
            sample_training_data.drop('sla_outcome', axis=1),
            sample_training_data['sla_outcome']
        )
        
        assert model is not None
        assert isinstance(model, DecisionTreeClassifier)
        assert 'decision_tree' in trainer.models
    
    def test_train_random_forest_model(self, sample_training_data):
        """Test training a random forest model."""
        trainer = ModelTrainer()
        
        # This should fail initially  
        model = trainer.train_random_forest(
            sample_training_data.drop('sla_outcome', axis=1),
            sample_training_data['sla_outcome']
        )
        
        assert model is not None
        assert isinstance(model, RandomForestClassifier)
        assert 'random_forest' in trainer.models
    
    def test_train_all_models(self, sample_training_data):
        """Test training all three models at once."""
        trainer = ModelTrainer()
        
        # This should fail initially
        models = trainer.train_all_models(
            sample_training_data.drop('sla_outcome', axis=1),
            sample_training_data['sla_outcome']  
        )
        
        assert isinstance(models, dict)
        assert len(models) == 3
        assert 'logistic_regression' in models
        assert 'decision_tree' in models
        assert 'random_forest' in models
    
    def test_get_model_by_name(self, sample_training_data):
        """Test retrieving a trained model by name."""
        trainer = ModelTrainer()
        trainer.train_all_models(
            sample_training_data.drop('sla_outcome', axis=1),
            sample_training_data['sla_outcome']
        )
        
        # This should fail initially
        model = trainer.get_model('logistic_regression')
        assert model is not None
        assert isinstance(model, LogisticRegression)
        
        # Test invalid model name
        with pytest.raises(ValueError):
            trainer.get_model('invalid_model')
    
    def test_model_training_with_insufficient_data(self):
        """Test model training with insufficient data raises appropriate error."""
        trainer = ModelTrainer()
        
        # Create insufficient data (only 1 row)
        insufficient_data = pd.DataFrame({
            'feature1': [1],
            'sla_outcome': [1]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            trainer.train_logistic_regression(
                insufficient_data.drop('sla_outcome', axis=1),
                insufficient_data['sla_outcome']
            )
    
    def test_model_training_with_invalid_features(self, sample_training_data):
        """Test model training with invalid feature data."""
        trainer = ModelTrainer()
        
        # Create data with NaN values
        invalid_features = sample_training_data.drop('sla_outcome', axis=1).copy()
        invalid_features.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="Invalid feature data"):
            trainer.train_logistic_regression(
                invalid_features,
                sample_training_data['sla_outcome']
            )
    
    def test_model_training_with_single_class_target(self):
        """Test model training when target has only one class."""
        trainer = ModelTrainer()
        
        # Create data with only one class
        single_class_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'sla_outcome': [1, 1, 1, 1, 1]  # All same class
        })
        
        with pytest.raises(ValueError, match="Target variable must have at least 2 classes"):
            trainer.train_logistic_regression(
                single_class_data.drop('sla_outcome', axis=1),
                single_class_data['sla_outcome']
            )