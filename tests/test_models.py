import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_training import ModelTrainer
from src.utils import load_config

class TestModelTraining:
    """Test cases for model training"""
    
    @pytest.fixture
    def config(self):
        return load_config("config/params.yaml")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        return X, y
    
    def test_model_initialization(self, config):
        """Test model trainer initialization"""
        trainer = ModelTrainer(config)
        
        assert trainer is not None
        assert isinstance(trainer.models, dict)
        assert len(trainer.models) == 0
    
    def test_get_model(self, config):
        """Test getting model instances"""
        trainer = ModelTrainer(config)
        
        model_names = ['RandomForest', 'XGBoost', 'LightGBM']
        
        for name in model_names:
            model = trainer.get_model(name)
            assert model is not None
    
    def test_hyperparameter_loading(self, config):
        """Test hyperparameter configuration loading"""
        trainer = ModelTrainer(config)
        
        for model_name in config['models']['algorithms']:
            params = trainer.get_hyperparameters(model_name)
            assert isinstance(params, dict)
    
    def test_single_model_training(self, config, sample_data):
        """Test training a single model"""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        trainer = ModelTrainer(config)
        metrics = trainer.train_single_model('RandomForest', X_train, y_train, X_test, y_test)
        
        assert metrics is not None
        assert 'model_name' in metrics
        assert 'train_score' in metrics
        assert metrics['model_name'] == 'RandomForest'
    
    def test_model_evaluation(self, config, sample_data):
        """Test model evaluation functionality"""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        trainer = ModelTrainer(config)
        trainer.train_single_model('RandomForest', X_train, y_train, X_test, y_test)
        
        evaluation_df = trainer.evaluate_models(X_test, y_test)
        
        assert not evaluation_df.empty
        assert 'Model' in evaluation_df.columns
        assert 'Accuracy' in evaluation_df.columns
        assert 'ROC-AUC' in evaluation_df.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])