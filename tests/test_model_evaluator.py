"""
Test suite for ModelEvaluator class.
Tests model evaluation functionality including metrics calculation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from src.sla_predictor.model_evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test cases for ModelEvaluator functionality."""
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data for evaluation."""
        return pd.DataFrame({
            'day_of_week': [1, 2, 3, 4, 5],
            'month': [6, 6, 6, 6, 6],
            'is_weekend': [0, 0, 0, 0, 0]
        })
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return np.array([1, 0, 1, 1, 0])
    
    @pytest.fixture
    def sample_true_labels(self):
        """Create sample true labels."""
        return np.array([1, 0, 1, 0, 1])
    
    @pytest.fixture
    def trained_model(self):
        """Create a mock trained model."""
        model = Mock(spec=LogisticRegression)
        model.predict.return_value = np.array([1, 0, 1, 1, 0])
        model.predict_proba.return_value = np.array([
            [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.3, 0.7], [0.6, 0.4]
        ])
        return model
    
    def test_model_evaluator_initialization(self):
        """Test ModelEvaluator can be initialized properly."""
        evaluator = ModelEvaluator()
        
        assert evaluator is not None
        assert hasattr(evaluator, 'evaluation_results')
        assert isinstance(evaluator.evaluation_results, dict)
    
    def test_calculate_accuracy(self, sample_true_labels, sample_predictions):
        """Test accuracy calculation."""
        evaluator = ModelEvaluator()
        
        # This should fail initially
        accuracy = evaluator.calculate_accuracy(sample_true_labels, sample_predictions)
        
        expected_accuracy = accuracy_score(sample_true_labels, sample_predictions)
        assert accuracy == expected_accuracy
        assert 0.0 <= accuracy <= 1.0
    
    def test_calculate_recall(self, sample_true_labels, sample_predictions):
        """Test recall calculation."""
        evaluator = ModelEvaluator()
        
        # This should fail initially
        recall = evaluator.calculate_recall(sample_true_labels, sample_predictions)
        
        expected_recall = recall_score(sample_true_labels, sample_predictions)
        assert recall == expected_recall
        assert 0.0 <= recall <= 1.0
    
    def test_calculate_auc_roc(self, sample_true_labels):
        """Test AUC-ROC calculation."""
        evaluator = ModelEvaluator()
        
        # Create sample probabilities
        sample_probabilities = np.array([0.8, 0.3, 0.9, 0.7, 0.4])
        
        # This should fail initially
        auc_roc = evaluator.calculate_auc_roc(sample_true_labels, sample_probabilities)
        
        expected_auc = roc_auc_score(sample_true_labels, sample_probabilities)
        assert auc_roc == expected_auc
        assert 0.0 <= auc_roc <= 1.0
    
    def test_evaluate_model_comprehensive(self, trained_model, sample_test_data, sample_true_labels):
        """Test comprehensive model evaluation."""
        evaluator = ModelEvaluator()
        
        # This should fail initially
        results = evaluator.evaluate_model(trained_model, sample_test_data, sample_true_labels)
        
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'recall' in results
        assert 'auc_roc' in results
        assert 'predictions' in results
        
        # Check that all metrics are valid
        assert 0.0 <= results['accuracy'] <= 1.0
        assert 0.0 <= results['recall'] <= 1.0
        assert 0.0 <= results['auc_roc'] <= 1.0
        assert len(results['predictions']) == len(sample_true_labels)
    
    def test_evaluate_multiple_models(self, sample_test_data, sample_true_labels):
        """Test evaluation of multiple models."""
        evaluator = ModelEvaluator()
        
        # Create multiple mock models
        models = {
            'logistic_regression': Mock(spec=LogisticRegression),
            'decision_tree': Mock(spec=LogisticRegression),
            'random_forest': Mock(spec=LogisticRegression)
        }
        
        # Configure mock predictions
        for model in models.values():
            model.predict.return_value = np.array([1, 0, 1, 1, 0])
            model.predict_proba.return_value = np.array([
                [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.3, 0.7], [0.6, 0.4]
            ])
        
        # This should fail initially
        results = evaluator.evaluate_all_models(models, sample_test_data, sample_true_labels)
        
        assert isinstance(results, dict)
        assert len(results) == 3
        assert 'logistic_regression' in results
        assert 'decision_tree' in results
        assert 'random_forest' in results
        
        # Check each model's results
        for model_name, model_results in results.items():
            assert 'accuracy' in model_results
            assert 'recall' in model_results
            assert 'auc_roc' in model_results
    
    def test_get_best_model(self, sample_test_data, sample_true_labels):
        """Test finding the best model based on accuracy."""
        evaluator = ModelEvaluator()
        
        # Create models with different performance
        models = {
            'model_a': Mock(spec=LogisticRegression),
            'model_b': Mock(spec=LogisticRegression),
            'model_c': Mock(spec=LogisticRegression)
        }
        
        # Configure different accuracies
        models['model_a'].predict.return_value = np.array([1, 0, 1, 1, 0])  # 3/5 = 0.6
        models['model_b'].predict.return_value = np.array([1, 0, 1, 0, 1])  # 5/5 = 1.0
        models['model_c'].predict.return_value = np.array([0, 1, 0, 1, 0])  # 1/5 = 0.2
        
        for model in models.values():
            model.predict_proba.return_value = np.array([
                [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.3, 0.7], [0.6, 0.4]
            ])
        
        # Evaluate all models first
        evaluator.evaluate_all_models(models, sample_test_data, sample_true_labels)
        
        # This should fail initially
        best_model_name, best_accuracy = evaluator.get_best_model()
        
        assert best_model_name == 'model_b'
        assert best_accuracy == 1.0
    
    def test_evaluation_with_invalid_data(self, trained_model):
        """Test evaluation with invalid data raises appropriate errors."""
        evaluator = ModelEvaluator()
        
        # Test with mismatched data lengths
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array([1, 0])  # Different length
        
        with pytest.raises(ValueError, match="Data length mismatch"):
            evaluator.evaluate_model(trained_model, X_test, y_test)
    
    def test_evaluation_with_empty_data(self, trained_model):
        """Test evaluation with empty data raises error."""
        evaluator = ModelEvaluator()
        
        X_test = pd.DataFrame()
        y_test = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            evaluator.evaluate_model(trained_model, X_test, y_test)
    
    def test_auc_roc_with_single_class(self):
        """Test AUC-ROC calculation with single class labels."""
        evaluator = ModelEvaluator()
        
        # All labels are the same class
        y_true = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.8, 0.7, 0.9, 0.6, 0.5])
        
        with pytest.raises(ValueError, match="AUC-ROC cannot be calculated"):
            evaluator.calculate_auc_roc(y_true, y_prob)