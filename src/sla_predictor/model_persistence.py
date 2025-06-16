"""
Model Persistence module for SLA Predictor.

This module provides functionality to save and load trained machine learning models
with optional metadata for version tracking and model information.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPersistence:
    """
    Handles saving and loading of trained machine learning models.
    
    Supports model serialization with pickle and optional metadata storage.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize ModelPersistence.
        
        Args:
            models_dir: Directory to store models (default: ./models)
        """
        if models_dir is None:
            models_dir = Path('models')
        
        self.models_dir = Path(models_dir)
        
        # Create directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelPersistence initialized with directory: {self.models_dir}")
    
    def _validate_model_name(self, model_name: str) -> None:
        """
        Validate model name for file system safety.
        
        Args:
            model_name: Name of the model
            
        Raises:
            ValueError: If model name is invalid
        """
        if not model_name or not isinstance(model_name, str):
            logger.error(f"Invalid model name: {model_name}")
            raise ValueError("Invalid model name: must be a non-empty string")
        
        # Check for dangerous characters
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        if any(char in model_name for char in dangerous_chars):
            logger.error(f"Model name contains dangerous characters: {model_name}")
            raise ValueError(f"Invalid model name: contains dangerous characters")
    
    def save_model(self, model: Any, model_name: str) -> Path:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            model_name: Name for the saved model
            
        Returns:
            Path to the saved model file
            
        Raises:
            ValueError: If model name is invalid
        """
        self._validate_model_name(model_name)
        
        file_path = self.models_dir / f"{model_name}.pkl"
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model saved successfully: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the saved model
            
        Returns:
            Loaded model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model name is invalid
        """
        self._validate_model_name(model_name)
        
        file_path = self.models_dir / f"{model_name}.pkl"
        
        if not file_path.exists():
            logger.error(f"Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Model loaded successfully: {file_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def save_all_models(self, models: Dict[str, Any]) -> Dict[str, Path]:
        """
        Save multiple models to disk.
        
        Args:
            models: Dictionary of model_name -> model
            
        Returns:
            Dictionary of model_name -> saved_file_path
        """
        saved_paths = {}
        
        logger.info(f"Saving {len(models)} models")
        
        for model_name, model in models.items():
            try:
                file_path = self.save_model(model, model_name)
                saved_paths[model_name] = file_path
            except Exception as e:
                logger.error(f"Failed to save model {model_name}: {e}")
                # Continue with other models
                continue
        
        logger.info(f"Successfully saved {len(saved_paths)} models")
        return saved_paths
    
    def load_all_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Load multiple models from disk.
        
        Args:
            model_names: List of model names to load
            
        Returns:
            Dictionary of model_name -> loaded_model
        """
        loaded_models = {}
        
        logger.info(f"Loading {len(model_names)} models")
        
        for model_name in model_names:
            try:
                model = self.load_model(model_name)
                loaded_models[model_name] = model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Continue with other models
                continue
        
        logger.info(f"Successfully loaded {len(loaded_models)} models")
        return loaded_models
    
    def save_model_with_metadata(self, model: Any, model_name: str, metadata: Dict[str, Any]) -> Path:
        """
        Save a model with associated metadata.
        
        Args:
            model: Trained model to save
            model_name: Name for the saved model
            metadata: Dictionary containing model metadata
            
        Returns:
            Path to the saved model file
        """
        # Save the model
        model_path = self.save_model(model, model_name)
        
        # Save metadata
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving metadata for {model_name}: {e}")
            # Don't fail if metadata saving fails
            pass
        
        return model_path
    
    def load_model_with_metadata(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model with its associated metadata.
        
        Args:
            model_name: Name of the saved model
            
        Returns:
            Tuple of (loaded_model, metadata_dict)
        """
        # Load the model
        model = self.load_model(model_name)
        
        # Load metadata
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        metadata = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                logger.info(f"Model metadata loaded: {metadata_path}")
                
            except Exception as e:
                logger.error(f"Error loading metadata for {model_name}: {e}")
                # Return empty metadata if loading fails
                metadata = {}
        else:
            logger.warning(f"No metadata file found for model: {model_name}")
        
        return model, metadata
    
    def list_saved_models(self) -> List[str]:
        """
        List all saved models in the models directory.
        
        Returns:
            List of saved model names
        """
        model_files = list(self.models_dir.glob("*.pkl"))
        model_names = [f.stem for f in model_files if not f.stem.endswith('_metadata')]
        
        logger.info(f"Found {len(model_names)} saved models")
        return sorted(model_names)
    
    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists on disk.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            self._validate_model_name(model_name)
            file_path = self.models_dir / f"{model_name}.pkl"
            return file_path.exists()
        except ValueError:
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model and its metadata.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self._validate_model_name(model_name)
            
            # Delete model file
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted model file: {model_path}")
            
            # Delete metadata file if it exists
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"Deleted metadata file: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False