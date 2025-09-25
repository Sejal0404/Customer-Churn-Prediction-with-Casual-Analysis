import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class CausalAnalyzer:
    """Handles causal analysis with simplified approach"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.data = data
        self.config = config
        
    def prepare_data_for_analysis(self):
        """Prepare data for causal analysis"""
        df = self.data.copy()
        
        
        # Convert Churn to numeric
        if 'Churn' in df.columns:
            if df['Churn'].dtype == 'object':
                df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
            else:
                df['Churn_numeric'] = df['Churn']
        
        return df
    
    def analyze_treatment_effect_simple(self, treatment: str, outcome: str = 'Churn'):
        """Simple treatment effect analysis without complex dependencies"""
        df = self.prepare_data_for_analysis()
        
        if treatment not in df.columns:
            logger.warning(f"Treatment variable {treatment} not found in data")
            return None
        
        outcome_col = 'Churn_numeric' if outcome == 'Churn' else outcome
        
        try:
            if df[treatment].dtype == 'object':
                # Categorical treatment - calculate mean differences
                results = self._analyze_categorical_treatment(df, treatment, outcome_col)
            else:
                # Numerical treatment - calculate correlation and regression
                results = self._analyze_numerical_treatment(df, treatment, outcome_col)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {treatment}: {e}")
            return None
    
    def _analyze_categorical_treatment(self, df: pd.DataFrame, treatment: str, outcome_col: str) -> Dict[str, Any]:
        """Analyze categorical treatment variables"""
        # Calculate overall churn rate
        overall_churn = df[outcome_col].mean()
        
        # Calculate churn rate by treatment level
        churn_by_treatment = df.groupby(treatment)[outcome_col].agg(['mean', 'count']).round(4)
        churn_by_treatment['effect_size'] = churn_by_treatment['mean'] - overall_churn
        churn_by_treatment['relative_effect'] = churn_by_treatment['effect_size'] / overall_churn
        
        # Statistical test (chi-square)
        from scipy.stats import chi2_contingency
        contingency_table = pd.crosstab(df[treatment], df[outcome_col])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        return {
            'treatment': treatment,
            'type': 'categorical',
            'overall_churn_rate': overall_churn,
            'churn_by_category': churn_by_treatment.to_dict('index'),
            'statistical_test': {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        }
    
    def _analyze_numerical_treatment(self, df: pd.DataFrame, treatment: str, outcome_col: str) -> Dict[str, Any]:
        """Analyze numerical treatment variables"""
        # Calculate correlation
        correlation = df[treatment].corr(df[outcome_col])
        
        # Simple linear regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        X = df[[treatment]].values
        y = df[outcome_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return {
            'treatment': treatment,
            'type': 'numerical',
            'correlation_with_outcome': correlation,
            'regression_coefficient': model.coef_[0],
            'r_squared': r2
        }
    
    def analyze_all_treatments(self) -> Dict[str, Any]:
        """Analyze all specified treatment variables"""
        results = {}
        treatment_vars = self.config['causal_analysis']['treatment_variables']
        
        for treatment in treatment_vars:
            result = self.analyze_treatment_effect_simple(treatment)
            if result is not None:
                results[treatment] = result
        
        return results
    
    def get_effect_summary(self) -> pd.DataFrame:
        """Get summary of all treatment effects in a clean DataFrame"""
        results = self.analyze_all_treatments()
        
        summary_data = []
        
        for treatment, result in results.items():
            if result['type'] == 'categorical':
                # For categorical treatments, create a row for each category
                for category, stats in result['churn_by_category'].items():
                    summary_data.append({
                        'Treatment': treatment,
                        'Category': category,
                        'Churn_Rate': stats['mean'],
                        'Overall_Churn_Rate': result['overall_churn_rate'],
                        'Effect_Size': stats['effect_size'],
                        'Relative_Effect_Percent': stats['relative_effect'] * 100,
                        'Sample_Size': stats['count'],
                        'P_Value': result['statistical_test']['p_value'],
                        'Significant': result['statistical_test']['significant']
                    })
            else:
                # For numerical treatments
                summary_data.append({
                    'Treatment': treatment,
                    'Category': 'Numerical',
                    'Correlation': result['correlation_with_outcome'],
                    'Regression_Coefficient': result['regression_coefficient'],
                    'R_Squared': result['r_squared'],
                    'Effect_Size': result['correlation_with_outcome'],  # Use correlation as effect size
                    'Relative_Effect_Percent': abs(result['correlation_with_outcome']) * 100,
                    'Sample_Size': len(self.data),
                    'P_Value': None,  # Would need to calculate for correlation
                    'Significant': abs(result['correlation_with_outcome']) > 0.1  # Simple threshold
                })
        
        return pd.DataFrame(summary_data)
    
    def create_summary_plot(self, top_n: int = 10):
        """Create a summary plot of treatment effects"""
        summary_df = self.get_effect_summary()
        
        if summary_df.empty:
            logger.warning("No data available for plotting")
            return None
        
        # Get top effects by absolute size
        plot_data = summary_df.nlargest(top_n, 'Effect_Size', keep='all')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['red' if x < 0 else 'green' for x in plot_data['Effect_Size']]
        y_pos = range(len(plot_data))
        
        ax.barh(y_pos, plot_data['Effect_Size'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Treatment']} - {row['Category']}" for _, row in plot_data.iterrows()])
        ax.set_xlabel('Effect Size (Difference in Churn Rate)')
        ax.set_title('Top Treatment Effects on Customer Churn')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value annotations
        for i, v in enumerate(plot_data['Effect_Size']):
            ax.text(v, i, f'{v:.3f}', va='center', ha='left' if v < 0 else 'right')
        
        plt.tight_layout()
        return fig