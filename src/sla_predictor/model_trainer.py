"""
Model Training module for SLA Predictor.

This module provides functionality to train machine learning models for SLA prediction.
Supports Logistic Regression, Decision Tree, and Random Forest algorithms.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains machine learning models for SLA prediction.
    
    Supports training of multiple scikit-learn algorithms:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    """
    
    def __init__(self):
        """Initialize ModelTrainer with empty models dictionary."""
        self.models: Dict[str, Any] = {}
        logger.info("ModelTrainer initialized")
    
    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate training data before model training.
        
        Args:
            X: Feature data
            y: Target variable
            
        Raises:
            ValueError: If data is insufficient or invalid
        """
        # Check data size
        if len(X) < 2:
            logger.error(f"Insufficient data: only {len(X)} samples provided")
            raise ValueError("Insufficient data for model training (minimum 2 samples required)")
        
        # Check for NaN values in features
        if X.isnull().any().any():
            logger.error("Invalid feature data: contains NaN values")
            raise ValueError("Invalid feature data: features cannot contain NaN values")
        
        # Check for NaN values in target
        if y.isnull().any():
            logger.error("Invalid target data: contains NaN values")
            raise ValueError("Invalid target data: target cannot contain NaN values")
        
        # Check target has at least 2 classes
        unique_classes = y.unique()
        if len(unique_classes) < 2:
            logger.error(f"Target variable has only {len(unique_classes)} unique class(es): {unique_classes}")
            raise ValueError("Target variable must have at least 2 classes for classification")
        
        logger.info(f"Training data validation passed: {len(X)} samples, {len(X.columns)} features, {len(unique_classes)} classes")
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
        """
        Train a Logistic Regression model.
        
        Args:
            X: Feature data
            y: Target variable (SLA outcomes)
            
        Returns:
            Trained LogisticRegression model
            
        Raises:
            ValueError: If training data is invalid
        """
        logger.info("Starting Logistic Regression training")
        
        # Validate data
        self._validate_training_data(X, y)
        
        # Create and train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        # Store model
        self.models['logistic_regression'] = model
        
        logger.info("Logistic Regression training completed successfully")
        return model
    
    def train_decision_tree(self, X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
        """
        Train a Decision Tree model.
        
        Args:
            X: Feature data
            y: Target variable (SLA outcomes)
            
        Returns:
            Trained DecisionTreeClassifier model
            
        Raises:
            ValueError: If training data is invalid
        """
        logger.info("Starting Decision Tree training")
        
        # Validate data
        self._validate_training_data(X, y)
        
        # Create and train model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        
        # Store model
        self.models['decision_tree'] = model
        
        logger.info("Decision Tree training completed successfully")
        return model
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        Train a Random Forest model.
        
        Args:
            X: Feature data
            y: Target variable (SLA outcomes)
            
        Returns:
            Trained RandomForestClassifier model
            
        Raises:
            ValueError: If training data is invalid
        """
        logger.info("Starting Random Forest training")
        
        # Validate data
        self._validate_training_data(X, y)
        
        # Create and train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Store model
        self.models['random_forest'] = model
        
        logger.info("Random Forest training completed successfully")
        return model
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train all three models (Logistic Regression, Decision Tree, Random Forest).
        
        Args:
            X: Feature data  
            y: Target variable (SLA outcomes)
            
        Returns:
            Dictionary containing all trained models
            
        Raises:
            ValueError: If training data is invalid
        """
        logger.info("Starting training of all models")
        
        # Train each model
        self.train_logistic_regression(X, y)
        self.train_decision_tree(X, y)
        self.train_random_forest(X, y)
        
        logger.info("All models trained successfully")
        return self.models.copy()
    
    def get_model(self, model_name: str) -> Any:
        """
        Retrieve a trained model by name.
        
        Args:
            model_name: Name of the model ('logistic_regression', 'decision_tree', 'random_forest')
            
        Returns:
            The requested trained model
            
        Raises:
            ValueError: If model name is invalid or model not trained
        """
        valid_models = ['logistic_regression', 'decision_tree', 'random_forest']
        
        if model_name not in valid_models:
            logger.error(f"Invalid model name: {model_name}. Valid options: {valid_models}")
            raise ValueError(f"Invalid model name: {model_name}. Valid options: {valid_models}")
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} has not been trained yet")
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        logger.info(f"Retrieved model: {model_name}")
        return self.models[model_name]