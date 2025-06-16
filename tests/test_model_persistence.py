"""
Test suite for ModelPersistence class.
Tests model saving and loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.sla_predictor.model_persistence import ModelPersistence


class TestModelPersistence:
    """Test cases for ModelPersistence functionality."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        # Create simple training data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        # Train a simple model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model
    
    def test_model_persistence_initialization(self, temp_directory):
        """Test ModelPersistence can be initialized properly."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        assert persistence is not None
        assert persistence.models_dir == temp_directory
        assert temp_directory.exists()
    
    def test_model_persistence_default_directory(self):
        """Test ModelPersistence with default directory."""
        persistence = ModelPersistence()
        
        assert persistence is not None
        assert persistence.models_dir.name == 'models'
    
    def test_save_single_model(self, temp_directory, trained_model):
        """Test saving a single model."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        # This should fail initially
        file_path = persistence.save_model(trained_model, 'test_model')
        
        assert file_path.exists()
        assert file_path.suffix == '.pkl'
        assert 'test_model' in file_path.name
    
    def test_load_single_model(self, temp_directory, trained_model):
        """Test loading a single model."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        # Save model first
        file_path = persistence.save_model(trained_model, 'test_model')
        
        # This should fail initially
        loaded_model = persistence.load_model('test_model')
        
        assert loaded_model is not None
        assert type(loaded_model) == type(trained_model)
        
        # Test that loaded model can make predictions
        test_data = np.array([[2, 3], [6, 7]])
        original_predictions = trained_model.predict(test_data)
        loaded_predictions = loaded_model.predict(test_data)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_save_multiple_models(self, temp_directory):
        """Test saving multiple models at once."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        # Create multiple trained models
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Train models
        for model in models.values():
            model.fit(X, y)
        
        # This should fail initially
        saved_paths = persistence.save_all_models(models)
        
        assert isinstance(saved_paths, dict)
        assert len(saved_paths) == 2
        assert 'logistic_regression' in saved_paths
        assert 'decision_tree' in saved_paths
        
        # Check files exist
        for path in saved_paths.values():
            assert path.exists()
            assert path.suffix == '.pkl'
    
    def test_load_multiple_models(self, temp_directory):
        """Test loading multiple models."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        # Create and save multiple models
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        original_models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        
        for model in original_models.values():
            model.fit(X, y)
        
        persistence.save_all_models(original_models)
        
        # This should fail initially
        loaded_models = persistence.load_all_models(['logistic_regression', 'decision_tree'])
        
        assert isinstance(loaded_models, dict)
        assert len(loaded_models) == 2
        assert 'logistic_regression' in loaded_models
        assert 'decision_tree' in loaded_models
        
        # Test predictions match
        test_data = np.array([[2, 3], [6, 7]])
        for model_name in original_models.keys():
            original_pred = original_models[model_name].predict(test_data)
            loaded_pred = loaded_models[model_name].predict(test_data)
            np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_save_model_with_metadata(self, temp_directory, trained_model):
        """Test saving model with metadata."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        metadata = {
            'accuracy': 0.95,
            'recall': 0.87,
            'training_date': '2023-12-01',
            'features': ['feature1', 'feature2']
        }
        
        # This should fail initially
        file_path = persistence.save_model_with_metadata(trained_model, 'test_model_meta', metadata)
        
        assert file_path.exists()
        
        # Check metadata file exists
        metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
        assert metadata_file.exists()
    
    def test_load_model_with_metadata(self, temp_directory, trained_model):
        """Test loading model with metadata."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        metadata = {
            'accuracy': 0.95,
            'recall': 0.87,
            'training_date': '2023-12-01'
        }
        
        # Save model with metadata
        persistence.save_model_with_metadata(trained_model, 'test_model_meta', metadata)
        
        # This should fail initially
        loaded_model, loaded_metadata = persistence.load_model_with_metadata('test_model_meta')
        
        assert loaded_model is not None
        assert loaded_metadata is not None
        assert loaded_metadata['accuracy'] == 0.95
        assert loaded_metadata['recall'] == 0.87
        assert loaded_metadata['training_date'] == '2023-12-01'
    
    def test_list_saved_models(self, temp_directory, trained_model):
        """Test listing all saved models."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        # Save multiple models
        persistence.save_model(trained_model, 'model1')
        persistence.save_model(trained_model, 'model2')
        persistence.save_model(trained_model, 'model3')
        
        # This should fail initially
        model_list = persistence.list_saved_models()
        
        assert isinstance(model_list, list)
        assert len(model_list) == 3
        assert 'model1' in model_list
        assert 'model2' in model_list
        assert 'model3' in model_list
    
    def test_model_exists(self, temp_directory, trained_model):
        """Test checking if a model exists."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        # Model doesn't exist yet
        assert not persistence.model_exists('nonexistent_model')
        
        # Save model
        persistence.save_model(trained_model, 'existing_model')
        
        # This should fail initially
        assert persistence.model_exists('existing_model')
    
    def test_delete_model(self, temp_directory, trained_model):
        """Test deleting a saved model."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        # Save model
        persistence.save_model(trained_model, 'model_to_delete')
        assert persistence.model_exists('model_to_delete')
        
        # This should fail initially
        success = persistence.delete_model('model_to_delete')
        
        assert success
        assert not persistence.model_exists('model_to_delete')
    
    def test_load_nonexistent_model_raises_error(self, temp_directory):
        """Test loading nonexistent model raises error."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        with pytest.raises(FileNotFoundError):
            persistence.load_model('nonexistent_model')
    
    def test_save_model_invalid_name_raises_error(self, temp_directory, trained_model):
        """Test saving model with invalid name raises error."""
        persistence = ModelPersistence(models_dir=temp_directory)
        
        with pytest.raises(ValueError, match="Invalid model name"):
            persistence.save_model(trained_model, "")
        
        with pytest.raises(ValueError, match="Invalid model name"):
            persistence.save_model(trained_model, None)