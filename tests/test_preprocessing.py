import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.utils import load_config

class TestDataPreprocessing:
    """Test cases for data preprocessing"""
    
    @pytest.fixture
    def config(self):
        return load_config("config/params.yaml")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'tenure': [1, 34, 2],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': ['29.85', '1889.5', '108.15'],
            'Churn': ['No', 'No', 'Yes']
        })
    
    def test_data_loading(self, config):
        """Test data loading functionality"""
        preprocessor = DataPreprocessor(config)
        
        # This will fail if data file doesn't exist, which is expected
        try:
            df = preprocessor.load_data()
            assert df is not None
            assert len(df) > 0
        except FileNotFoundError:
            pytest.skip("Data file not available for testing")
    
    def test_data_cleaning(self, config, sample_data):
        """Test data cleaning functionality"""
        preprocessor = DataPreprocessor(config)
        cleaned_data = preprocessor.clean_data(sample_data)
        
        # Test TotalCharges conversion
        assert pd.api.types.is_numeric_dtype(cleaned_data['TotalCharges'])
        
        # Test Churn conversion
        assert set(cleaned_data['Churn'].unique()).issubset({0, 1})
        
        # Test SeniorCitizen conversion
        assert set(cleaned_data['SeniorCitizen'].unique()).issubset({'0', '1'})
    
    def test_preprocessor_creation(self, config):
        """Test preprocessor pipeline creation"""
        preprocessor = DataPreprocessor(config)
        
        numeric_features = config['data']['numeric_features']
        categorical_features = config['data']['categorical_features']
        
        pipeline = preprocessor.create_preprocessor(numeric_features, categorical_features)
        
        assert pipeline is not None
        assert 'num' in pipeline.named_transformers_
        assert 'cat' in pipeline.named_transformers_
    
    def test_missing_value_handling(self, config):
        """Test missing value handling"""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'tenure': [1, np.nan, 3],
            'MonthlyCharges': [29.85, 56.95, np.nan],
            'gender': ['Male', np.nan, 'Female'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        preprocessor = DataPreprocessor(config)
        cleaned_data = preprocessor.clean_data(data_with_missing)
        
        # Check that missing values are handled
        assert cleaned_data.isnull().sum().sum() == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])