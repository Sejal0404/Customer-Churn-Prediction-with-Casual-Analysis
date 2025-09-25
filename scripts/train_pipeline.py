#!/usr/bin/env python3
"""
Full training pipeline for customer churn prediction with graceful fallbacks
"""

import sys
import os
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import setup_logging, load_config, save_model, save_dataframe
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
import pandas as pd

# Try to import optional modules with fallbacks
try:
    from src.explainability import SHAPExplainer
    SHAP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SHAP not available: {e}")
    SHAP_AVAILABLE = False

try:
    from src.causal_analysis import CausalAnalyzer
    CAUSAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Causal analysis not available: {e}")
    CAUSAL_AVAILABLE = False

def main():
    # Setup
    config_path = os.path.join(project_root, "config", "params.yaml")
    config = load_config(config_path)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training pipeline...")
    logger.info(f"SHAP available: {SHAP_AVAILABLE}")
    logger.info(f"Causal analysis available: {CAUSAL_AVAILABLE}")
    
    try:
        # 1. Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        preprocessor = DataPreprocessor(config)
        
        # Load data
        df = preprocessor.load_data()
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Clean data
        df_clean = preprocessor.clean_data(df)
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        
        # Check target distribution
        churn_distribution = df_clean['Churn'].value_counts()
        logger.info(f"Churn distribution: {churn_distribution.to_dict()}")
        
        # Prepare data for modeling
        X_processed, y, fitted_preprocessor = preprocessor.prepare_data(df_clean)
        logger.info(f"Processed features shape: {X_processed.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X_processed, y)
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "data", "interim"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "results", "reports"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "results", "figures"), exist_ok=True)
        
        # Save processed data
        save_dataframe(X_train, os.path.join(project_root, "data", "processed", "train_features.csv"))
        save_dataframe(pd.DataFrame(y_train, columns=['Churn']), os.path.join(project_root, "data", "processed", "train_target.csv"))
        save_dataframe(X_test, os.path.join(project_root, "data", "processed", "test_features.csv"))
        save_dataframe(pd.DataFrame(y_test, columns=['Churn']), os.path.join(project_root, "data", "processed", "test_target.csv"))
        save_model(fitted_preprocessor, os.path.join(project_root, "models", "preprocessor.pkl"))
        
        # Save cleaned data for potential causal analysis
        save_dataframe(df_clean, os.path.join(project_root, "data", "interim", "preprocessed_data.csv"))
        
        # 2. Feature Engineering
        logger.info("Step 2: Feature Engineering")
        feature_engineer = FeatureEngineer(config)
        feature_importance = feature_engineer.calculate_feature_importance(X_train, y_train)
        save_dataframe(feature_importance, os.path.join(project_root, "results", "reports", "feature_importance.csv"))
        
        logger.info("Top 10 features by importance:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # 3. Model Training
        logger.info("Step 3: Model Training")
        model_trainer = ModelTrainer(config)
        results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save best model
        if model_trainer.best_model:
            save_model(model_trainer.best_model, os.path.join(project_root, "models", "best_model.pkl"))
            logger.info(f"Best model saved: {type(model_trainer.best_model).__name__}")
        
        # 4. Model Evaluation
        logger.info("Step 4: Model Evaluation")
        evaluation_df = model_trainer.evaluate_models(X_test, y_test)
        save_dataframe(evaluation_df, os.path.join(project_root, "results", "reports", "model_comparison.csv"))
        
        # Print model results
        logger.info("Model comparison results:")
        for _, row in evaluation_df.iterrows():
            logger.info(f"  {row['Model']}: AUC={row['ROC-AUC']:.4f}, Accuracy={row['Accuracy']:.4f}")
        
        # 5. Explainability (Optional)
        if SHAP_AVAILABLE and model_trainer.best_model:
            logger.info("Step 5: Explainability Analysis")
            try:
                explainer = SHAPExplainer(
                    model_trainer.best_model, 
                    fitted_preprocessor,
                    X_train.columns.tolist()
                )
                # Use a small sample for SHAP to avoid memory issues
                X_sample = X_test.head(100)
                shap_importance = explainer.get_feature_importance(X_sample)
                save_dataframe(shap_importance, os.path.join(project_root, "results", "reports", "shap_importance.csv"))
                logger.info("SHAP analysis completed successfully")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
        else:
            logger.info("SHAP analysis skipped (not available or no best model)")
        
        # 6. Causal Analysis (Optional)
        if CAUSAL_AVAILABLE:
            logger.info("Step 6: Causal Analysis")
            try:
                causal_analyzer = CausalAnalyzer(df_clean, config)
                causal_results = causal_analyzer.analyze_all_treatments()
                causal_summary = causal_analyzer.get_effect_summary()
                save_dataframe(causal_summary, os.path.join(project_root, "results", "reports", "causal_effects.csv"))
                logger.info("Causal analysis completed successfully")
            except Exception as e:
                logger.warning(f"Causal analysis failed: {e}")
        else:
            logger.info("Causal analysis skipped (not available)")
        
        logger.info("Training pipeline completed successfully!")
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print("TRAINING PIPELINE SUMMARY")
        print("="*60)
        print(f"Dataset: {len(df)} customers, {len(df.columns)-1} features")
        print(f"Churn rate: {(df_clean['Churn'].mean()*100):.1f}%")
        print(f"Final features: {X_processed.shape[1]}")
        
        if model_trainer.best_model:
            best_model_name = type(model_trainer.best_model).__name__
            best_auc = model_trainer.best_score
            print(f"Best Model: {best_model_name} (AUC: {best_auc:.4f})")
        
        print(f"Models Trained: {len(results)}")
        print(f"SHAP Analysis: {'Completed' if SHAP_AVAILABLE else 'Skipped'}")
        print(f"Causal Analysis: {'Completed' if CAUSAL_AVAILABLE else 'Skipped'}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()