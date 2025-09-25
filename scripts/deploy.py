#!/usr/bin/env python3
"""
Deployment script for the churn prediction model
"""

import sys
import os
import logging
import shutil
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import setup_logging, load_config

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting deployment process...")
    
    # Load config
    config_path = os.path.join(project_root, "config", "params.yaml")
    config = load_config(config_path)
    
    # Create deployment directory
    deploy_dir = os.path.join(project_root, "deployment")
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Files to include in deployment
    deployment_files = [
        "models/best_model.pkl",
        "models/preprocessor.pkl", 
        "config/params.yaml",
        "src/utils.py",
        "src/model_training.py"
    ]
    
    # Create deployment package
    logger.info("Creating deployment package...")
    
    for file_path in deployment_files:
        source_path = os.path.join(project_root, file_path)
        dest_path = os.path.join(deploy_dir, os.path.basename(file_path))
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied {file_path}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    # Create prediction API script
    create_prediction_api(deploy_dir, config)
    
    # Create requirements file for deployment
    create_requirements_file(deploy_dir)
    
    # Create deployment documentation
    create_deployment_docs(deploy_dir)
    
    logger.info(f"Deployment package created in: {deploy_dir}")
    logger.info("Deployment completed successfully!")

def create_prediction_api(deploy_dir, config):
    """Create a simple prediction API script"""
    api_script = """
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any

class ChurnPredictor:
    \"\"\"Simple churn prediction API\"\"\"
    
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Predict churn for a single customer\"\"\"
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Preprocess
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        probability = self.model.predict_proba(X_processed)[0][1]
        prediction = probability > 0.5
        
        return {
            'churn_probability': float(probability),
            'prediction': bool(prediction),
            'risk_level': 'high' if probability > 0.7 else 'medium' if probability > 0.3 else 'low'
        }
    
    def predict_batch(self, customers_data: list) -> list:
        \"\"\"Predict churn for multiple customers\"\"\"
        results = []
        for customer_data in customers_data:
            results.append(self.predict_single(customer_data))
        return results

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor('best_model.pkl', 'preprocessor.pkl')
    
    # Example customer data
    example_customer = {
        'tenure': 12,
        'MonthlyCharges': 70.5,
        'TotalCharges': 845.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
        'InternetService': 'Fiber optic'
    }
    
    result = predictor.predict_single(example_customer)
    print("Prediction result:", result)
"""
    
    with open(os.path.join(deploy_dir, "predictor.py"), "w") as f:
        f.write(api_script)

def create_requirements_file(deploy_dir):
    """Create a minimal requirements file for deployment"""
    requirements = """
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
joblib>=1.3.0
"""
    
    with open(os.path.join(deploy_dir, "requirements.txt"), "w") as f:
        f.write(requirements)

def create_deployment_docs(deploy_dir):
    """Create deployment documentation"""
    docs = f"""
# Customer Churn Prediction Model - Deployment Guide

## Deployment Package
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents
- `best_model.pkl`: Trained XGBoost model
- `preprocessor.pkl`: Data preprocessing pipeline  
- `params.yaml`: Configuration file
- `predictor.py`: Prediction API
- `requirements.txt`: Python dependencies

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt