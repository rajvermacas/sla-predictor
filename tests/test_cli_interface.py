"""
Test suite for the SLA Predictor CLI Interface.

This module contains comprehensive tests for the command-line interface
that allows users to make predictions via command-line arguments.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import argparse
import sys
from io import StringIO

from src.sla_predictor.cli_interface import CLIInterface


class TestCLIInterface(unittest.TestCase):
    """Test cases for the CLIInterface class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Create sample historical data file
        self.data_file = Path(self.temp_dir) / "historical_data.csv"
        with open(self.data_file, 'w') as f:
            f.write("Date,SLA_Outcome\n")
            f.write("2023-01-01,1\n")
            f.write("2023-01-02,0\n")
            f.write("2023-01-03,1\n")
    
    def test_cli_interface_initialization(self):
        """Test that CLIInterface initializes correctly."""
        # This should fail initially (RED phase)
        cli = CLIInterface()
        
        self.assertIsNotNone(cli)
        self.assertIsNotNone(cli.parser)
    
    def test_parse_valid_arguments(self):
        """Test parsing valid command line arguments."""
        # This should fail initially (RED phase)
        cli = CLIInterface()
        
        args = cli.parse_arguments([
            '--date', '2023-02-15',
            '--data', str(self.data_file),
            '--models-dir', str(self.models_dir)
        ])
        
        self.assertEqual(args.date, '2023-02-15')
        self.assertEqual(str(args.data), str(self.data_file))
        self.assertEqual(str(args.models_dir), str(self.models_dir))
    
    def test_parse_arguments_with_confidence(self):
        """Test parsing arguments with confidence flag."""
        # This should fail initially (RED phase)
        cli = CLIInterface()
        
        args = cli.parse_arguments([
            '--date', '2023-02-15',
            '--data', str(self.data_file),
            '--models-dir', str(self.models_dir),
            '--confidence'
        ])
        
        self.assertTrue(args.confidence)
    
    def test_parse_arguments_with_model_selection(self):
        """Test parsing arguments with specific model selection."""
        # This should fail initially (RED phase)
        cli = CLIInterface()
        
        args = cli.parse_arguments([
            '--date', '2023-02-15',
            '--data', str(self.data_file),
            '--models-dir', str(self.models_dir),
            '--model', 'logistic_regression'
        ])
        
        self.assertEqual(args.model, 'logistic_regression')
    
    def test_run_prediction_basic(self):
        """Test running a basic prediction through CLI."""
        # Mock the prediction engine
        with patch('src.sla_predictor.cli_interface.PredictionEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.predict.return_value = "Yes"
            mock_engine_class.return_value = mock_engine
            
            cli = CLIInterface()
            
            # Capture stdout
            captured_output = StringIO()
            with patch('sys.stdout', captured_output):
                cli.run([
                    '--date', '2023-02-15',
                    '--data', str(self.data_file),
                    '--models-dir', str(self.models_dir)
                ])
            
            output = captured_output.getvalue()
            self.assertIn("Yes", output)
    
    def test_help_message_available(self):
        """Test that help message is available."""
        # This should fail initially (RED phase)
        cli = CLIInterface()
        
        with self.assertRaises(SystemExit):
            cli.parse_arguments(['--help'])
    
    def test_invalid_date_format_error(self):
        """Test error handling for invalid date format."""
        # This should fail initially (RED phase)
        cli = CLIInterface()
        
        with self.assertRaises(SystemExit):
            cli.parse_arguments([
                '--date', 'invalid-date',
                '--data', str(self.data_file),
                '--models-dir', str(self.models_dir)
            ])
    
    def test_missing_required_arguments_error(self):
        """Test error handling for missing required arguments."""
        # This should fail initially (RED phase)
        cli = CLIInterface()
        
        with self.assertRaises(SystemExit):
            cli.parse_arguments(['--date', '2023-02-15'])


if __name__ == '__main__':
    unittest.main()