#!/usr/bin/env python3
"""
Batch prediction script for customer churn
"""

import argparse
import pandas as pd
import numpy as np
from src.utils import setup_logging, load_model, save_dataframe
import logging
import sys
import os

sys.path.append(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description='Batch prediction for customer churn')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file path')
    parser.add_argument('--model', type=str, default='models/best_model.pkl', help='Model file path')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load model and preprocessor
        model = load_model(args.model)
        preprocessor = load_model('models/preprocessor.pkl')
        
        if model is None or preprocessor is None:
            logger.error("Model or preprocessor not found")
            return
        
        # Load data
        logger.info(f"Loading data from {args.input}")
        new_data = pd.read_csv(args.input)
        
        # Preprocess data
        X_new = preprocessor.transform(new_data)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)[:, 1]
        
        # Create results
        results = new_data.copy()
        results['Churn_Prediction'] = predictions
        results['Churn_Probability'] = probabilities
        results['Prediction_Confidence'] = np.abs(probabilities - 0.5) * 2
        
        # Save results
        save_dataframe(results, args.output)
        logger.info(f"Predictions saved to {args.output}")
        
        # Print summary
        churn_count = predictions.sum()
        churn_rate = churn_count / len(predictions) * 100
        
        print("\n" + "="*50)
        print("BATCH PREDICTION SUMMARY")
        print("="*50)
        print(f"Total customers: {len(predictions)}")
        print(f"Predicted churn: {churn_count} ({churn_rate:.1f}%)")
        print(f"Results saved to: {args.output}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise

if __name__ == "__main__":
    main()