"""
Command Line Interface for SLA Predictor.

This module provides a command-line interface for making SLA predictions,
allowing users to specify dates and get Yes/No predictions with optional
confidence scores.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Optional

from .prediction_engine import PredictionEngine
from .data_loader import DataLoader


class CLIInterface:
    """
    Command Line Interface for SLA predictions.
    
    This class handles command-line argument parsing and orchestrates
    the prediction process through the PredictionEngine.
    """
    
    def __init__(self):
        """Initialize the CLI interface with argument parser."""
        self.parser = self._create_argument_parser()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """
        Create and configure the argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description='SLA Predictor - Predict whether a daily feed will meet its SLA deadline',
            prog='sla-predictor'
        )
        
        parser.add_argument(
            '--date',
            required=True,
            help='Date to make prediction for (YYYY-MM-DD format)',
            type=self._validate_date_format
        )
        
        parser.add_argument(
            '--data',
            required=True,
            help='Path to historical SLA data CSV file',
            type=Path
        )
        
        parser.add_argument(
            '--models-dir',
            required=True,
            help='Directory containing trained models',
            type=Path
        )
        
        parser.add_argument(
            '--model',
            help='Specific model to use (logistic_regression, decision_tree, random_forest)',
            choices=['logistic_regression', 'decision_tree', 'random_forest'],
            default=None
        )
        
        parser.add_argument(
            '--confidence',
            action='store_true',
            help='Include confidence score in output'
        )
        
        parser.add_argument(
            '--version',
            action='version',
            version='SLA Predictor 1.0.0'
        )
        
        return parser
    
    def _validate_date_format(self, date_string: str) -> str:
        """
        Validate date format.
        
        Args:
            date_string: Date string to validate
            
        Returns:
            Validated date string
            
        Raises:
            argparse.ArgumentTypeError: If date format is invalid
        """
        try:
            datetime.strptime(date_string, '%Y-%m-%d')
            return date_string
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Expected YYYY-MM-DD")
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Args:
            args: Optional list of arguments (for testing)
            
        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
    
    def run(self, args: Optional[List[str]] = None) -> None:
        """
        Run the CLI interface.
        
        Args:
            args: Optional list of arguments (for testing)
        """
        try:
            parsed_args = self.parse_arguments(args)
            
            # Load historical data
            self.logger.info(f"Loading historical data from: {parsed_args.data}")
            data_loader = DataLoader()
            historical_data = data_loader.load_csv(parsed_args.data)
            
            # Initialize prediction engine
            prediction_engine = PredictionEngine(
                models_dir=parsed_args.models_dir,
                historical_data=historical_data
            )
            
            # Convert date string to date object
            prediction_date = datetime.strptime(parsed_args.date, '%Y-%m-%d').date()
            
            # Make prediction
            if parsed_args.model:
                # Use specific model
                result = prediction_engine.predict_with_model(prediction_date, parsed_args.model)
                if parsed_args.confidence:
                    print(f"Prediction: {result} (using {parsed_args.model} model)")
                else:
                    print(result)
            else:
                # Use default prediction
                result = prediction_engine.predict(prediction_date, include_confidence=parsed_args.confidence)
                
                if parsed_args.confidence:
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                else:
                    print(result)
            
            self.logger.info(f"Prediction completed for {parsed_args.date}")
            
        except Exception as e:
            self.logger.error(f"Error running CLI: {str(e)}")
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli = CLIInterface()
    cli.run()


if __name__ == '__main__':
    main()