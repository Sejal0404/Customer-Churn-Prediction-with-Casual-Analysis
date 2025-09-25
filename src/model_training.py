import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
from typing import Dict, List, Tuple, Any  # Added imports
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and hyperparameter tuning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def get_model(self, model_name: str):
    """Get model instance by name"""
    models = {
        'RandomForest': RandomForestClassifier(random_state=self.config['data']['random_state']),
        'LogisticRegression': LogisticRegression(random_state=self.config['data']['random_state']),
        'SVM': SVC(probability=True, random_state=self.config['data']['random_state']),
        'XGBoost': XGBClassifier(random_state=self.config['data']['random_state'], eval_metric='logloss'),
        'LightGBM': LGBMClassifier(
            random_state=self.config['data']['random_state'],
            verbose=-1,  # Suppress warnings
            force_row_wise=True,  # Prevent splitting warnings
            min_child_samples=20,  # Prevent overfitting
            num_leaves=31,  # Conservative leaf count
            feature_fraction=0.8  # Prevent overfitting
        )
    }
    return models.get(model_name)
    
    def get_hyperparameters(self, model_name: str) -> Dict[str, List]:
        """Get hyperparameter grid for model"""
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        return param_grids.get(model_name, {})
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_test: pd.DataFrame = None, 
                          y_test: pd.Series = None) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning"""
        logger.info(f"Training {model_name}...")
        
        # Get model and parameters
        model = self.get_model(model_name)
        param_grid = self.get_hyperparameters(model_name)
        
        if not param_grid:
            # No hyperparameter tuning, just fit the model
            model.fit(X_train, y_train)
            best_model = model
            best_params = model.get_params()
        else:
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.config['models']['cross_validation'],
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        
        # Evaluate model
        train_score = best_model.score(X_train, y_train)
        y_pred = best_model.predict(X_test) if X_test is not None else None
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if X_test is not None else None
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'best_params': best_params,
            'train_score': train_score,
            'feature_importance': None
        }
        
        if X_test is not None and y_test is not None:
            test_score = best_model.score(X_test, y_test)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            metrics.update({
                'test_score': test_score,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
            
            # Feature importance for tree-based models
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                metrics['feature_importance'] = feature_importance
        
        self.models[model_name] = {
            'model': best_model,
            'metrics': metrics
        }
        
        logger.info(f"{model_name} trained. Train score: {train_score:.4f}")
        if 'test_score' in metrics:
            logger.info(f"{model_name} test score: {test_score:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train all specified models"""
        results = {}
        
        for model_name in self.config['models']['algorithms']:
            try:
                metrics = self.train_single_model(model_name, X_train, y_train, X_test, y_test)
                results[model_name] = metrics
                
                # Update best model
                if metrics.get('roc_auc', 0) > self.best_score:
                    self.best_score = metrics['roc_auc']
                    self.best_model = self.models[model_name]['model']
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all trained models"""
        evaluation_results = []
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            metrics = model_data['metrics']
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            
            evaluation = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
                'Best Parameters': str(metrics.get('best_params', {}))
            }
            
            evaluation_results.append(evaluation)
        
        return pd.DataFrame(evaluation_results)