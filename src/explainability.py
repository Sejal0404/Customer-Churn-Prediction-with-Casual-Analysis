import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

def explain_model(model, X_test, feature_names, model_type='tree'):
    """Generate SHAP explanations"""
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    try:
        X_test_array = X_test.values.astype(np.float32)
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_array)
        else:
            explainer = shap.LinearExplainer(model, X_test_array)
            shap_values = explainer.shap_values(X_test_array)

        # Create summary plot
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        shap.summary_plot(shap_values, X_test_array, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        return explainer, shap_values

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None, None

def plot_feature_importance(shap_values, feature_names):
    """Plot feature importance"""
    if shap_values is None:
        return None
    
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(15))
    plt.title('Feature Importance from SHAP')
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def plot_confusion_matrix(model, X_test, y_test):
    """Plot and save confusion matrix and classification report"""
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix.png', dpi=300)
    plt.close()

    # Save classification report as CSV
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(results_dir / 'classification_report.csv')

    return cr_df

def plot_feature_distributions(df, features):
    """Plot and save feature distributions for given features"""
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)

    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=feature, hue='Churn', multiple='stack', palette='Set2', kde=True)
        plt.title(f'Distribution of {feature} by Churn')
        plt.tight_layout()
        plt.savefig(results_dir / f'feature_distribution_{feature}.png', dpi=300)
        plt.close()
