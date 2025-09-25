import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any  # Added imports
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_proba: np.ndarray = None, model_name: str = ""):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1_score': f1_score(self.y_true, self.y_pred),
        }
        
        if self.y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
        
        return metrics
    
    def plot_confusion_matrix(self, normalize: bool = False, 
                            figsize: tuple = (8, 6)) -> plt.Figure:
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.model_name}')
        
        return fig
    
    def plot_roc_curve(self, figsize: tuple = (8, 6)) -> plt.Figure:
        """Plot ROC curve"""
        if self.y_pred_proba is None:
            logger.warning("No probability scores available for ROC curve")
            return None
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        return fig
    
    def plot_precision_recall_curve(self, figsize: tuple = (8, 6)) -> plt.Figure:
        """Plot precision-recall curve"""
        if self.y_pred_proba is None:
            logger.warning("No probability scores available for PR curve")
            return None
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        avg_precision = np.mean(precision)
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 15, figsize: tuple = (10, 8)) -> plt.Figure:
        """Plot feature importance"""
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}')
        plt.tight_layout()
        
        return fig
    
    def generate_classification_report(self) -> str:
        """Generate detailed classification report"""
        report = classification_report(self.y_true, self.y_pred, 
                                      target_names=['No Churn', 'Churn'])
        return report
    
    def calculate_business_metrics(self, cost_matrix: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate business-oriented metrics"""
        if cost_matrix is None:
            cost_matrix = {
                'fp_cost': 100,  # Cost of false positive (unnecessary retention effort)
                'fn_cost': 500,  # Cost of false negative (lost customer)
                'tp_benefit': 300,  # Benefit of true positive (successful retention)
                'tn_benefit': 0    # Benefit of true negative (no action needed)
            }
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fp * cost_matrix['fp_cost'] + 
                     fn * cost_matrix['fn_cost'] - 
                     tp * cost_matrix['tp_benefit'] - 
                     tn * cost_matrix['tn_benefit'])
        
        metrics = {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_cost': total_cost,
            'cost_per_customer': total_cost / len(self.y_true)
        }
        
        return metrics
    
    def create_comprehensive_report(self) -> Dict[str, Any]:
        """Create comprehensive evaluation report"""
        report = {
            'model_name': self.model_name,
            'basic_metrics': self.calculate_metrics(),
            'classification_report': self.generate_classification_report(),
            'business_metrics': self.calculate_business_metrics(),
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred).tolist()
        }
        
        return report