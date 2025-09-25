import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional  # Added imports
import logging
import joblib

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """Handles SHAP explanations for model interpretability"""
    
    def __init__(self, model, preprocessor, feature_names: List[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X: pd.DataFrame, sample_size: int = 1000):
        """Create SHAP explainer"""
        # Sample data for faster computation
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            try:
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_sample)
            except:
                # Fallback to KernelExplainer for non-tree models
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
                self.shap_values = self.explainer.shap_values(X_sample)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, X_sample)
            self.shap_values = self.explainer.shap_values(X_sample)
        
        logger.info("SHAP explainer created successfully")
        return self.explainer, self.shap_values
    
    def summary_plot(self, X: pd.DataFrame, plot_type: str = "dot", 
                    max_display: int = 20, show: bool = True):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            self.create_explainer(X)
        
        plt.figure(figsize=(10, 8))
        
        # Handle binary classification
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_plot = self.shap_values[1]  # Use class 1 (positive class)
        else:
            shap_values_plot = self.shap_values
        
        shap.summary_plot(shap_values_plot, X, 
                         feature_names=self.feature_names,
                         plot_type=plot_type,
                         max_display=max_display,
                         show=show)
        
        plt.tight_layout()
        return plt.gcf()
    
    def force_plot(self, instance_idx: int, X: pd.DataFrame, 
                  expected_value: float = None, show: bool = True):
        """Create individual force plot"""
        if self.shap_values is None:
            self.create_explainer(X)
        
        if expected_value is None and self.explainer is not None:
            expected_value = self.explainer.expected_value
        
        # Handle binary classification
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_plot = self.shap_values[1]
        else:
            shap_values_plot = self.shap_values
        
        plt.figure(figsize=(12, 4))
        shap.force_plot(expected_value, 
                       shap_values_plot[instance_idx, :], 
                       X.iloc[instance_idx, :],
                       feature_names=self.feature_names,
                       matplotlib=True,
                       show=show)
        
        return plt.gcf()
    
    def dependence_plot(self, feature_name: str, X: pd.DataFrame, 
                       interaction_index: str = None, show: bool = True):
        """Create dependence plot for a specific feature"""
        if self.shap_values is None:
            self.create_explainer(X)
        
        # Handle binary classification
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_plot = self.shap_values[1]
        else:
            shap_values_plot = self.shap_values
        
        feature_idx = list(X.columns).index(feature_name) if feature_name in X.columns else None
        
        if feature_idx is not None:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature_idx, shap_values_plot, X, 
                               feature_names=self.feature_names,
                               interaction_index=interaction_index,
                               show=show)
            
            return plt.gcf()
        else:
            logger.warning(f"Feature {feature_name} not found in dataset")
            return None
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance from SHAP values"""
        if self.shap_values is None:
            self.create_explainer(X)
        
        # Handle binary classification
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_plot = self.shap_values[1]
        else:
            shap_values_plot = self.shap_values
        
        # Calculate mean absolute SHAP values
        shap_importance = np.abs(shap_values_plot).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df
    
    def create_waterfall_plot(self, instance_idx: int, X: pd.DataFrame, 
                            max_display: int = 10, show: bool = True):
        """Create waterfall plot for individual prediction"""
        if self.shap_values is None:
            self.create_explainer(X)
        
        # Handle binary classification
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_plot = self.shap_values[1]
            expected_value = self.explainer.expected_value[1]
        else:
            shap_values_plot = self.shap_values
            expected_value = self.explainer.expected_value
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(values=shap_values_plot[instance_idx],
                                           base_values=expected_value,
                                           data=X.iloc[instance_idx],
                                           feature_names=self.feature_names),
                          max_display=max_display,
                          show=show)
        
        return plt.gcf()