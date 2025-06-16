"""
SLA outcome validation functionality for SLA Predictor
GREEN phase: Minimal implementation to make tests pass
"""
import pandas as pd
import logging
from typing import Union, List, Dict, Any


class SLAValidator:
    """Handles validation and standardization of SLA outcome values."""
    
    def __init__(self):
        """Initialize SLAValidator with logging configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Define mapping for flexible input formats
        self.outcome_mapping = {
            # Integer values
            0: 0, 1: 1,
            # String representations
            "0": 0, "1": 1,
            # Boolean representations
            "yes": 1, "no": 0,
            "true": 1, "false": 0,
            # Boolean values
            True: 1, False: 0
        }
    
    def validate_binary_outcome(self, outcome: Any) -> int:
        """
        Validate and standardize binary SLA outcome.
        
        Args:
            outcome: SLA outcome value to validate
            
        Returns:
            int: Standardized outcome (0 or 1)
            
        Raises:
            ValueError: If outcome is invalid or cannot be converted
        """
        if outcome is None:
            raise ValueError("SLA outcome cannot be None")
        
        # Handle string inputs (strip whitespace and convert to lowercase)
        if isinstance(outcome, str):
            outcome = outcome.strip()
            if not outcome:
                raise ValueError("SLA outcome cannot be empty")
            outcome = outcome.lower()
        
        # Check if outcome is in mapping
        if outcome in self.outcome_mapping:
            result = self.outcome_mapping[outcome]
            self.logger.debug(f"Validated outcome '{outcome}' -> {result}")
            return result
        
        # If not in mapping, it's invalid
        self.logger.error(f"Invalid SLA outcome: {outcome}")
        raise ValueError(f"Invalid SLA outcome: {outcome}")
    
    def validate_dataframe_sla_outcomes(self, df: pd.DataFrame, sla_column: str = 'SLA_Outcome') -> pd.DataFrame:
        """
        Validate and standardize SLA outcome column in DataFrame.
        
        Args:
            df: DataFrame with SLA outcome column
            sla_column: Name of SLA outcome column to validate
            
        Returns:
            pd.DataFrame: DataFrame with standardized SLA outcome column
            
        Raises:
            ValueError: If SLA column missing or contains invalid outcomes
        """
        if sla_column not in df.columns:
            raise ValueError(f"SLA outcome column '{sla_column}' not found in DataFrame")
        
        df_copy = df.copy()
        
        try:
            # Validate each outcome in the column
            validated_outcomes = []
            for idx, outcome in enumerate(df_copy[sla_column]):
                try:
                    validated_outcome = self.validate_binary_outcome(outcome)
                    validated_outcomes.append(validated_outcome)
                except ValueError as e:
                    self.logger.error(f"Invalid SLA outcome at row {idx}: {outcome}")
                    raise ValueError("Invalid SLA outcome found in DataFrame")
            
            # Update the column with validated outcomes
            df_copy[sla_column] = validated_outcomes
            
            # Ensure column is integer type
            df_copy[sla_column] = df_copy[sla_column].astype('int64')
            
            self.logger.info(f"Successfully validated {len(validated_outcomes)} SLA outcomes")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error validating SLA outcomes in column '{sla_column}': {e}")
            raise
    
    def get_statistics(self, outcomes: List[int]) -> Dict[str, Any]:
        """
        Calculate statistics for SLA outcomes.
        
        Args:
            outcomes: List of validated SLA outcomes (0 or 1)
            
        Returns:
            Dict: Statistics including counts and percentages
            
        Raises:
            ValueError: If outcomes list is empty
        """
        if not outcomes:
            raise ValueError("Outcomes list cannot be empty")
        
        total_count = len(outcomes)
        sla_met_count = sum(outcomes)
        sla_missed_count = total_count - sla_met_count
        
        sla_met_percentage = (sla_met_count / total_count) * 100
        sla_missed_percentage = (sla_missed_count / total_count) * 100
        
        stats = {
            'total_count': total_count,
            'sla_met_count': sla_met_count,
            'sla_missed_count': sla_missed_count,
            'sla_met_percentage': sla_met_percentage,
            'sla_missed_percentage': sla_missed_percentage
        }
        
        self.logger.info(f"SLA Statistics: {stats}")
        return stats