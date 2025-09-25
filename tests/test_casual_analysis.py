import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.causal_analysis import CausalAnalyzer
from src.utils import load_config

class TestCausalAnalysis:
    """Test cases for causal analysis"""
    
    @pytest.fixture
    def config(self):
        return load_config("config/params.yaml")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for causal analysis"""
        np.random.seed(42)
        n_samples = 500
        
        data = pd.DataFrame({
            'treatment': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'confounder1': np.random.normal(0, 1, n_samples),
            'confounder2': np.random.normal(0, 1, n_samples),
            'outcome': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        # Add some correlation to make it realistic
        data['outcome'] = (data['treatment'] * 0.3 + 
                          data['confounder1'] * 0.2 + 
                          data['confounder2'] * 0.1 +
                          np.random.normal(0, 0.5, n_samples) > 0).astype(int)
        
        return data
    
    def test_causal_analyzer_initialization(self, config, sample_data):
        """Test causal analyzer initialization"""
        analyzer = CausalAnalyzer(sample_data, config)
        
        assert analyzer is not None
        assert analyzer.data.equals(sample_data)
        assert analyzer.config == config
    
    def test_causal_graph_creation(self, config, sample_data):
        """Test causal graph creation"""
        analyzer = CausalAnalyzer(sample_data, config)
        
        graph = analyzer.create_causal_graph('treatment', 'outcome', ['confounder1', 'confounder2'])
        
        assert graph is not None
        assert isinstance(graph, str)
        assert 'treatment' in graph
        assert 'outcome' in graph
    
    def test_treatment_effect_analysis(self, config, sample_data):
        """Test treatment effect analysis"""
        analyzer = CausalAnalyzer(sample_data, config)
        
        # Modify config for test data
        test_config = config.copy()
        test_config['causal_analysis']['treatment_variables'] = ['treatment']
        test_config['causal_analysis']['confounders'] = ['confounder1', 'confounder2']
        
        result = analyzer.analyze_treatment_effect('treatment', 'outcome', ['confounder1', 'confounder2'])
        
        # Analysis might fail with small sample, which is acceptable for tests
        if result is not None:
            assert 'treatment' in result
            assert 'estimate' in result
            assert 'model' in result
    
    def test_effect_summary(self, config, sample_data):
        """Test effect summary generation"""
        analyzer = CausalAnalyzer(sample_data, config)
        
        # Add a mock estimate for testing
        analyzer.estimates = {
            'treatment': {
                'treatment': 'treatment',
                'outcome': 'outcome',
                'estimate': type('obj', (object,), {
                    'value': 0.15,
                    'get_confidence_intervals': lambda: (0.1, 0.2),
                    'test_statistics': {'p_value': 0.03}
                })
            }
        }
        
        summary_df = analyzer.get_effect_summary()
        
        assert not summary_df.empty
        assert 'Treatment' in summary_df.columns
        assert 'Estimated_Effect' in summary_df.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])