#!/usr/bin/env python3
"""
Basic training pipeline without causal analysis (to test core functionality)
"""

import sys
import os
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import setup_logging, load_config, save_model, save_dataframe
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
import pandas as pd

def main():
    # Setup
    config_path = os.path.join(project_root, "config", "params.yaml")
    config = load_config(config_path)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting basic training pipeline...")
    
    try:
        # 1. Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        preprocessor = DataPreprocessor(config)
        df = preprocessor.load_data()
        logger.info(f"Original data shape: {df.shape}")
        
        df_clean = preprocessor.clean_data(df)
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        
        X_processed, y, fitted_preprocessor = preprocessor.prepare_data(df_clean)
        logger.info(f"Processed features shape: {X_processed.shape}")
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X_processed, y)
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "data", "interim"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "results", "reports"), exist_ok=True)
        
        # Save processed data
        save_dataframe(X_train, os.path.join(project_root, "data", "processed", "train_features.csv"))
        save_dataframe(pd.DataFrame(y_train), os.path.join(project_root, "data", "processed", "train_target.csv"))
        save_dataframe(X_test, os.path.join(project_root, "data", "processed", "test_features.csv"))
        save_dataframe(pd.DataFrame(y_test), os.path.join(project_root, "data", "processed", "test_target.csv"))
        save_model(fitted_preprocessor, os.path.join(project_root, "models", "preprocessor.pkl"))
        
        # Save cleaned data
        save_dataframe(df_clean, os.path.join(project_root, "data", "interim", "preprocessed_data.csv"))
        
        # 2. Model Training
        logger.info("Step 2: Model Training")
        model_trainer = ModelTrainer(config)
        results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save best model
        if model_trainer.best_model:
            save_model(model_trainer.best_model, os.path.join(project_root, "models", "best_model.pkl"))
            logger.info(f"Best model saved: {type(model_trainer.best_model).__name__}")
        
        # 3. Model Evaluation
        logger.info("Step 3: Model Evaluation")
        evaluation_df = model_trainer.evaluate_models(X_test, y_test)
        save_dataframe(evaluation_df, os.path.join(project_root, "results", "reports", "model_comparison.csv"))
        
        logger.info("Basic training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("BASIC TRAINING PIPELINE SUMMARY")
        print("="*50)
        print(f"Dataset: {len(df)} customers, {len(df.columns)-1} features")
        if model_trainer.best_model:
            print(f"Best Model: {type(model_trainer.best_model).__name__}")
            print(f"Best ROC-AUC: {model_trainer.best_score:.4f}")
        print(f"Models Trained: {len(results)}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in basic training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()