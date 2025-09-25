#!/usr/bin/env python3
"""
Run causal analysis separately
"""

import sys
import os
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import setup_logging, load_config, load_dataframe
from src.casual_analysis import CausalAnalyzer
import pandas as pd

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting causal analysis...")
    
    try:
        # Load config
        config_path = os.path.join(project_root, "config", "params.yaml")
        config = load_config(config_path)
        
        # Load preprocessed data
        data_path = os.path.join(project_root, "data", "interim", "preprocessed_data.csv")
        df_clean = load_dataframe(data_path)
        
        if df_clean.empty:
            logger.error("No preprocessed data found. Please run the training pipeline first.")
            return
        
        logger.info(f"Loaded data with shape: {df_clean.shape}")
        
        # Check if DoWhy is available
        try:
            import dowhy
            logger.info("DoWhy is available - proceeding with causal analysis")
        except ImportError:
            logger.error("DoWhy not installed. Please run: pip install dowhy")
            return
        
        # Perform causal analysis
        causal_analyzer = CausalAnalyzer(df_clean, config)
        causal_results = causal_analyzer.analyze_all_treatments()
        
        if causal_results:
            causal_summary = causal_analyzer.get_effect_summary()
            print("\n" + "="*60)
            print("CAUSAL ANALYSIS RESULTS")
            print("="*60)
            print(causal_summary.to_string(index=False))
            
            # Save results
            results_path = os.path.join(project_root, "results", "reports", "causal_effects.csv")
            causal_summary.to_csv(results_path, index=False)
            logger.info(f"Causal analysis results saved to {results_path}")
        else:
            # Fallback to simple correlation analysis
            simple_results = causal_analyzer.get_simple_correlations()
            if not simple_results.empty:
                print("\n" + "="*60)
                print("SIMPLE CORRELATION ANALYSIS (DoWhy not available)")
                print("="*60)
                print(simple_results.to_string(index=False))
                
                results_path = os.path.join(project_root, "results", "reports", "causal_effects.csv")
                simple_results.to_csv(results_path, index=False)
                logger.info(f"Correlation analysis results saved to {results_path}")
            
    except Exception as e:
        logger.error(f"Error in causal analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()