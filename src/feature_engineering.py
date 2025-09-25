import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering and selection"""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_selector = None
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df_eng = df.copy()
        
        # Create tenure groups
        df_eng['tenure_group'] = pd.cut(df_eng['tenure'], 
                                       bins=[0, 12, 24, 48, np.inf], 
                                       labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
        
        # Create charge ratio features
        if 'MonthlyCharges' in df_eng.columns and 'TotalCharges' in df_eng.columns:
            df_eng['charge_tenure_ratio'] = df_eng['TotalCharges'] / (df_eng['tenure'] + 1)
            df_eng['monthly_to_total_ratio'] = df_eng['MonthlyCharges'] / (df_eng['TotalCharges'] + 1)
        
        # Create service count features
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        service_cols_present = [col for col in service_columns if col in df_eng.columns]
        if service_cols_present:
            df_eng['total_services'] = df_eng[service_cols_present].apply(
                lambda x: (x == 'Yes').sum(), axis=1
            )
        
        logger.info("Interaction features created")
        return df_eng
    
    def select_features_anova(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple:
        """Select features using ANOVA F-test"""
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        logger.info(f"Selected {len(selected_features)} features using ANOVA")
        return X_selected, selected_features, feature_scores
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, 
                          n_features: int = 15) -> Tuple:
        """Select features using Recursive Feature Elimination"""
        estimator = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.config['data']['random_state']
        )
        
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.support_].tolist()
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_,
            'support': selector.support_
        }).sort_values('ranking')
        
        logger.info(f"Selected {len(selected_features)} features using RFE")
        return X_selected, selected_features, feature_ranking
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate feature importance using Random Forest"""
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.config['data']['random_state']
        )
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def create_feature_summary(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Create comprehensive feature summary"""
        # Basic statistics
        summary = {
            'total_features': X.shape[1],
            'numeric_features': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': X.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': X.isnull().sum().to_dict(),
            'feature_correlations': X.corrwith(y).to_dict()
        }
        
        # Feature importance
        if len(summary['numeric_features']) > 0:
            importance_df = self.calculate_feature_importance(X, y)
            summary['feature_importance'] = importance_df.to_dict('records')
        
        return summary