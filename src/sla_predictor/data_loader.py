"""
Data loading functionality for SLA Predictor
REFACTOR phase: Improved implementation with better error handling and logging
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Union, List


class DataLoader:
    """
    Handles loading and validation of CSV data files for SLA prediction.
    
    This class provides functionality to load historical SLA data from CSV files
    with proper validation and error handling.
    
    Attributes:
        required_columns (List[str]): List of required column names
        logger: Logger instance for this class
    """
    
    def __init__(self, required_columns: List[str] = None):
        """
        Initialize DataLoader with configurable required columns.
        
        Args:
            required_columns: List of required column names. 
                            Defaults to ['Date', 'SLA_Outcome']
        """
        self.logger = logging.getLogger(__name__)
        self.required_columns = required_columns or ['Date', 'SLA_Outcome']
        self.logger.debug(f"DataLoader initialized with required columns: {self.required_columns}")
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load CSV file and validate required columns.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            pandas.DataFrame: Loaded and validated data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or missing required columns
        """
        file_path = Path(file_path)
        self.logger.debug(f"Attempting to load CSV from: {file_path}")
        
        # Validate file existence and size
        self._validate_file(file_path)
        
        try:
            # Load CSV with explicit error handling
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded CSV with shape {df.shape} from {file_path}")
            
            # Validate required columns
            self._validate_columns(df)
            
            self.logger.info(f"Data validation completed successfully")
            return df
            
        except pd.errors.EmptyDataError:
            error_msg = f"Empty CSV file: {file_path}"
            self.logger.error(error_msg)
            raise ValueError("Empty CSV file")
        except pd.errors.ParserError as e:
            error_msg = f"CSV parsing error in {file_path}: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            self.logger.error(f"Unexpected error loading CSV {file_path}: {e}")
            raise
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate file existence and basic properties.
        
        Args:
            file_path: Path to file to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty
        """
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if file_path.stat().st_size == 0:
            error_msg = f"Empty CSV file: {file_path}"
            self.logger.error(error_msg)
            raise ValueError("Empty CSV file")
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.debug(f"All required columns present: {self.required_columns}")