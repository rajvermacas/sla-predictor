"""
Model Evaluation module for SLA Predictor.

This module provides functionality to evaluate machine learning models
using various metrics including accuracy, recall, and AUC-ROC.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings


# Configure logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates machine learning models for SLA prediction.
    
    Provides comprehensive evaluation metrics including:
    - Accuracy
    - Recall
    - AUC-ROC
    """
    
    def __init__(self):
        """Initialize ModelEvaluator with empty results."""
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        logger.info("ModelEvaluator initialized")
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f"Accuracy calculated: {accuracy:.4f}")
        return accuracy
    
    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        recall = recall_score(y_true, y_pred)
        logger.info(f"Recall calculated: {recall:.4f}")
        return recall
    
    def calculate_auc_roc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate AUC-ROC score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for positive class
            
        Returns:
            AUC-ROC score (0.0 to 1.0)
            
        Raises:
            ValueError: If AUC-ROC cannot be calculated (e.g., single class)
        """
        try:
            # Check if we have both classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.error(f"AUC-ROC requires at least 2 classes, got {len(unique_classes)}")
                raise ValueError("AUC-ROC cannot be calculated with only one class present")
            
            auc_roc = roc_auc_score(y_true, y_prob)
            logger.info(f"AUC-ROC calculated: {auc_roc:.4f}")
            return auc_roc
            
        except ValueError as e:
            logger.error(f"Error calculating AUC-ROC: {e}")
            raise
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ValueError: If data is invalid or mismatched
        """
        # Validate input data
        if len(X_test) == 0 or len(y_test) == 0:
            logger.error("Empty data provided for evaluation")
            raise ValueError("Empty data provided for evaluation")
        
        if len(X_test) != len(y_test):
            logger.error(f"Data length mismatch: X_test={len(X_test)}, y_test={len(y_test)}")
            raise ValueError("Data length mismatch between features and labels")
        
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Get predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]  # Positive class probabilities
        
        # Calculate metrics
        accuracy = self.calculate_accuracy(y_test, predictions)
        recall = self.calculate_recall(y_test, predictions)
        auc_roc = self.calculate_auc_roc(y_test, probabilities)
        
        results = {
            'accuracy': accuracy,
            'recall': recall,
            'auc_roc': auc_roc,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        logger.info("Model evaluation completed successfully")
        return results
    
    def evaluate_all_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of model_name -> trained_model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of model_name -> evaluation_results
        """
        logger.info(f"Evaluating {len(models)} models")
        
        all_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                results = self.evaluate_model(model, X_test, y_test)
                all_results[model_name] = results
                
                # Store in class attribute
                self.evaluation_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                # Continue with other models
                continue
        
        logger.info(f"Completed evaluation of {len(all_results)} models")
        return all_results
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, float]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison ('accuracy', 'recall', 'auc_roc')
            
        Returns:
            Tuple of (best_model_name, best_metric_value)
            
        Raises:
            ValueError: If no models have been evaluated or invalid metric
        """
        if not self.evaluation_results:
            logger.error("No models have been evaluated yet")
            raise ValueError("No models have been evaluated yet")
        
        valid_metrics = ['accuracy', 'recall', 'auc_roc']
        if metric not in valid_metrics:
            logger.error(f"Invalid metric: {metric}. Valid options: {valid_metrics}")
            raise ValueError(f"Invalid metric: {metric}. Valid options: {valid_metrics}")
        
        best_model = None
        best_score = -1
        
        for model_name, results in self.evaluation_results.items():
            score = results[metric]
            if score > best_score:
                best_score = score
                best_model = model_name
        
        logger.info(f"Best model: {best_model} with {metric}={best_score:.4f}")
        return best_model, best_score
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model evaluations as a DataFrame.
        
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'model': model_name,
                'accuracy': results['accuracy'],
                'recall': results['recall'],
                'auc_roc': results['auc_roc']
            })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info(f"Generated evaluation summary for {len(summary_data)} models")
        
        return summary_df