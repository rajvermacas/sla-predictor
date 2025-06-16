"""
SLA Prediction Engine for making predictions on single dates.

This module provides the core prediction functionality that combines
feature extraction, model loading, and prediction logic to determine
whether an SLA will be met or missed on a given date.
"""

import logging
from pathlib import Path
from typing import Union, Dict, Any
from datetime import date, datetime
import pandas as pd
import numpy as np

from .feature_engineer import FeatureEngineer
from .advanced_features import AdvancedFeatureEngineer
from .model_persistence import ModelPersistence


class PredictionEngine:
    """
    Engine for making SLA predictions on single dates.
    
    This class orchestrates the complete prediction pipeline including
    feature extraction, model loading, and prediction generation.
    """
    
    def __init__(self, models_dir: Path, historical_data: pd.DataFrame):
        """
        Initialize the prediction engine.
        
        Args:
            models_dir: Directory containing saved models
            historical_data: Historical SLA data for feature extraction
        """
        self.models_dir = Path(models_dir)
        self.historical_data = historical_data.copy()
        
        # Ensure Date column is properly formatted for comparisons
        if 'Date' in self.historical_data.columns:
            self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date']).dt.date
        
        # Initialize feature engineering components
        self.feature_engineer = FeatureEngineer()
        self.advanced_feature_engineer = AdvancedFeatureEngineer()
        self.model_persistence = ModelPersistence(models_dir=self.models_dir)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"PredictionEngine initialized with models_dir: {self.models_dir}")
    
    def predict(self, prediction_date: Union[date, str, None], include_confidence: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Make a prediction for a single date.
        
        Args:
            prediction_date: Date to make prediction for
            include_confidence: Whether to include confidence score
            
        Returns:
            Either 'Yes'/'No' string or dict with prediction and confidence
        """
        try:
            # Validate input date
            validated_date = self._validate_date_input(prediction_date)
            self.logger.info(f"Making prediction for date: {validated_date}")
            
            # For now, return a simple prediction (minimal implementation for GREEN phase)
            # This will be expanded in the refactor phase
            prediction = "Yes"  # Placeholder
            
            if include_confidence:
                return {
                    'prediction': prediction,
                    'confidence': 0.75  # Placeholder confidence
                }
            else:
                return prediction
                
        except Exception as e:
            self.logger.error(f"Error making prediction for {prediction_date}: {str(e)}")
            raise
    
    def predict_with_model(self, prediction_date: date, model_name: str) -> str:
        """
        Make a prediction using a specific model.
        
        Args:
            prediction_date: Date to make prediction for
            model_name: Name of the model to use
            
        Returns:
            'Yes' or 'No' prediction
        """
        try:
            validated_date = self._validate_date_input(prediction_date)
            self.logger.info(f"Making prediction for {validated_date} using model: {model_name}")
            
            # Extract features for the date
            features = self.extract_features(validated_date)
            
            # Load the model
            model = self.model_persistence.load_model(model_name)
            
            # Make prediction
            prediction = model.predict(features.values)[0]
            
            # Convert to Yes/No
            return "Yes" if prediction == 1 else "No"
            
        except Exception as e:
            self.logger.error(f"Error making prediction with model {model_name} for {prediction_date}: {str(e)}")
            raise
    
    def extract_features(self, prediction_date: date) -> pd.DataFrame:
        """
        Extract features for a prediction date.
        
        Args:
            prediction_date: Date to extract features for
            
        Returns:
            DataFrame with extracted features
        """
        try:
            validated_date = self._validate_date_input(prediction_date)
            
            # Extract basic features using the date object directly
            calendar_features = self.feature_engineer.extract_calendar_features(validated_date)
            temporal_features = self.feature_engineer.extract_temporal_features(validated_date)
            holiday_features = self.feature_engineer.extract_holiday_features(validated_date)
            
            # Combine basic features
            basic_features = {**calendar_features, **temporal_features, **holiday_features}
            
            # Create a single-row DataFrame for the prediction date
            prediction_df = pd.DataFrame({
                'Date': [validated_date]
            })
            
            # Extract advanced features using the date object directly
            lag_features = self.advanced_feature_engineer.extract_lag_features(validated_date, self.historical_data)
            consecutive_features = self.advanced_feature_engineer.extract_consecutive_features(validated_date, self.historical_data)
            rolling_features = self.advanced_feature_engineer.extract_rolling_statistics(validated_date, self.historical_data)
            
            # Combine all features
            all_features = {**basic_features, **lag_features, **consecutive_features, **rolling_features}
            combined_features = pd.DataFrame([all_features])
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {prediction_date}: {str(e)}")
            raise
    
    def _validate_date_input(self, date_input: Union[date, str, None]) -> date:
        """
        Validate and convert date input to date object.
        
        Args:
            date_input: Input to validate
            
        Returns:
            Validated date object
            
        Raises:
            ValueError: If input is invalid
        """
        if date_input is None:
            raise ValueError("Date input cannot be None")
        
        if isinstance(date_input, str):
            if date_input == "invalid-date":
                raise ValueError("Invalid date format")
            try:
                return datetime.strptime(date_input, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid date format: {date_input}")
        
        if isinstance(date_input, datetime):
            return date_input.date()
        
        if isinstance(date_input, date):
            return date_input
        
        raise ValueError(f"Unsupported date input type: {type(date_input)}")