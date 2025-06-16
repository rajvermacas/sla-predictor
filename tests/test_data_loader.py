"""
Test-First Data Loading Tests
RED phase: Write failing tests for CSV data loading with expected columns
"""
import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path

from src.sla_predictor.data_loader import DataLoader


class TestDataLoader:
    """Test class for data loading functionality"""
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing"""
        return """Date,SLA_Outcome
2023-01-01,1
2023-01-02,0
2023-01-03,1
2023-01-04,0"""
    
    @pytest.fixture
    def sample_csv_file(self, sample_csv_content):
        """Create a temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_data_loader_initialization(self):
        """Test DataLoader can be initialized"""
        loader = DataLoader()
        assert loader is not None
    
    def test_load_csv_returns_dataframe(self, sample_csv_file):
        """Test loading CSV returns pandas DataFrame"""
        loader = DataLoader()
        df = loader.load_csv(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
    
    def test_load_csv_has_expected_columns(self, sample_csv_file):
        """Test loaded CSV has Date and SLA_Outcome columns"""
        loader = DataLoader()
        df = loader.load_csv(sample_csv_file)
        expected_columns = ['Date', 'SLA_Outcome']
        assert list(df.columns) == expected_columns
    
    def test_load_csv_correct_data_types(self, sample_csv_file):
        """Test loaded CSV has correct data types"""
        loader = DataLoader()
        df = loader.load_csv(sample_csv_file)
        # Date should be string initially, SLA_Outcome should be int
        assert df['Date'].dtype == 'object'
        assert df['SLA_Outcome'].dtype == 'int64'
    
    def test_load_csv_correct_shape(self, sample_csv_file):
        """Test loaded CSV has correct shape"""
        loader = DataLoader()
        df = loader.load_csv(sample_csv_file)
        assert df.shape == (4, 2)  # 4 rows, 2 columns
    
    def test_load_csv_file_not_found_raises_error(self):
        """Test loading non-existent file raises FileNotFoundError"""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_csv("non_existent_file.csv")
    
    def test_load_csv_invalid_columns_raises_error(self):
        """Test loading CSV with invalid columns raises ValueError"""
        invalid_content = """Wrong,Columns
1,2
3,4"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(invalid_content)
            f.flush()
            
            loader = DataLoader()
            with pytest.raises(ValueError, match="Missing required columns"):
                loader.load_csv(f.name)
        
        os.unlink(f.name)
    
    def test_load_csv_empty_file_raises_error(self):
        """Test loading empty CSV raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")
            f.flush()
            
            loader = DataLoader()
            with pytest.raises(ValueError, match="Empty CSV file"):
                loader.load_csv(f.name)
        
        os.unlink(f.name)