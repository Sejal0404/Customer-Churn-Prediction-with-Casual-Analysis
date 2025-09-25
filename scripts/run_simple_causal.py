#!/usr/bin/env python3
"""
Simple causal analysis script that handles dependencies gracefully
"""

import sys
import os
import logging
import pandas as pd

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import setup_logging, load_config, load_dataframe

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting simple causal analysis...")
    
    try:
        # Load config
        config_path = os.path.join(project_root, "config", "params.yaml")
        config = load_config(config_path)
        
        # Load preprocessed data
        data_path = os.path.join(project_root, "data", "interim", "preprocessed_data.csv")
        df_clean = load_dataframe(data_path)
        
        if df_clean.empty:
            logger.error("No preprocessed data found.")
            return
        
        logger.info(f"Loaded data with shape: {df_clean.shape}")
        
        # Simple correlation analysis
        treatment_vars = config['causal_analysis']['treatment_variables']
        outcome = config['causal_analysis']['outcome']
        
        # Prepare data
        df_analysis = df_clean.copy()
        
        # Convert Churn to numeric
        if 'Churn' in df_analysis.columns:
            df_analysis['Churn_numeric'] = df_analysis['Churn'].map({'No': 0, 'Yes': 1})
            outcome_col = 'Churn_numeric'
        else:
            outcome_col = outcome
        
        results = []
        
        for treatment in treatment_vars:
            if treatment in df_analysis.columns:
                if df_analysis[treatment].dtype == 'object':
                    # Categorical treatment - calculate mean churn by category
                    churn_by_treatment = df_analysis.groupby(treatment)[outcome_col].mean()
                    base_rate = df_analysis[outcome_col].mean()
                    
                    for category, churn_rate in churn_by_treatment.items():
                        effect = churn_rate - base_rate
                        results.append({
                            'Treatment_Variable': treatment,
                            'Treatment_Level': category,
                            'Churn_Rate': churn_rate,
                            'Base_Churn_Rate': base_rate,
                            'Effect_Size': effect,
                            'Relative_Effect': effect / base_rate if base_rate > 0 else 0
                        })
                else:
                    # Numerical treatment - calculate correlation
                    try:
                        correlation = df_analysis[treatment].corr(df_analysis[outcome_col])
                        results.append({
                            'Treatment_Variable': treatment,
                            'Treatment_Level': 'Numerical',
                            'Correlation_with_Churn': correlation,
                            'Analysis_Type': 'Correlation'
                        })
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation for {treatment}: {e}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            print("\n" + "="*80)
            print("SIMPLE CAUSAL ANALYSIS RESULTS")
            print("="*80)
            print(results_df.to_string(index=False))
            
            # Save results
            results_path = os.path.join(project_root, "results", "reports", "simple_causal_effects.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"Results saved to {results_path}")
            
            # Print key insights
            print("\n" + "="*80)
            print("KEY INSIGHTS")
            print("="*80)
            
            # Find largest effects
            if 'Effect_Size' in results_df.columns:
                largest_effect = results_df.loc[results_df['Effect_Size'].abs().idxmax()]
                print(f"Largest effect: {largest_effect['Treatment_Variable']} = {largest_effect['Treatment_Level']}")
                print(f"Effect size: {largest_effect['Effect_Size']:.3f} ({(largest_effect['Relative_Effect']*100):.1f}% change)")
            
            if 'Correlation_with_Churn' in results_df.columns:
                strongest_corr = results_df.loc[results_df['Correlation_with_Churn'].abs().idxmax()]
                print(f"Strongest correlation: {strongest_corr['Treatment_Variable']}")
                print(f"Correlation: {strongest_corr['Correlation_with_Churn']:.3f}")
        
        else:
            logger.warning("No results generated")
            
    except Exception as e:
        logger.error(f"Error in simple causal analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()