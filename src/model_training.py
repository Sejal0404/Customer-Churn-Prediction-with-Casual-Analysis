import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
import joblib
from pathlib import Path

def train_models(X_train, y_train, X_test, y_test, model_type='tree'):
    """Train machine learning models"""
    
    if model_type == 'tree':
        models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, use_label_encoder=False)
        }
        param_grids = {
            'random_forest': {'n_estimators': [100, 200]},
            'xgboost': {'n_estimators': [100, 200]}
        }
    else:
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        param_grids = {
            'logistic_regression': {'C': [0.1, 1, 10]}
        }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model': best_model,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'report': classification_report(y_test, y_pred)
        }
        
        # Save model
        models_dir = Path('../models')
        models_dir.mkdir(exist_ok=True)
        joblib.dump(best_model, models_dir / f'{name}.pkl')
    
    return results

def select_best_model(results):
    """Select the best performing model"""
    best_score = -1
    best_name = None
    
    for name, result in results.items():
        if result['roc_auc'] > best_score:
            best_score = result['roc_auc']
            best_name = name
    
    return best_name, results[best_name]['model']