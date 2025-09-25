import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a metric card for dashboard"""
    if delta is not None:
        return go.Indicator(
            mode="number+delta",
            value=value,
            title={"text": title},
            delta={'reference': delta, 'relative': True},
            domain={'row': 0, 'column': 0}
        )
    else:
        return go.Indicator(
            mode="number",
            value=value,
            title={"text": title},
            domain={'row': 0, 'column': 0}
        )

def create_confusion_matrix_plot(y_true, y_pred, labels=None):
    """Create a confusion matrix plot"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = ['Negative', 'Positive']
    
    fig = px.imshow(cm, 
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=labels, y=labels,
                   color_continuous_scale='Blues',
                   aspect="auto")
    
    # Add annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(x=j, y=i, text=str(cm[i, j]), 
                             showarrow=False, font=dict(color='white' if cm[i, j] > cm.max()/2 else 'black'))
    
    return fig

def create_roc_curve_plot(y_true, y_pred_proba):
    """Create ROC curve plot"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                           name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                           name='Random classifier', line=dict(dash='dash')))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=600, height=600
    )
    
    return fig

def create_feature_importance_plot(importance_df, top_n=15):
    """Create feature importance plot"""
    top_features = importance_df.head(top_n)
    
    fig = px.bar(top_features, x='importance', y='feature', 
                orientation='h', title=f'Top {top_n} Feature Importance',
                color='importance', color_continuous_scale='viridis')
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def create_causal_effect_plot(causal_results):
    """Create causal effect plot with confidence intervals"""
    fig = go.Figure()
    
    for _, row in causal_results.iterrows():
        color = 'red' if row['Estimated_Effect'] > 0 else 'green'
        
        fig.add_trace(go.Scatter(
            x=[row['Estimated_Effect']],
            y=[row['Treatment']],
            error_x=dict(
                type='data',
                array=[[row['Estimated_Effect'] - row['CI_Lower']]],
                arrayminus=[[row['CI_Upper'] - row['Estimated_Effect']]]
            ),
            mode='markers',
            marker=dict(size=10, color=color),
            name=row['Treatment']
        ))
    
    fig.update_layout(
        title="Causal Effects with Confidence Intervals",
        xaxis_title="Effect Size",
        yaxis_title="Treatment",
        showlegend=False
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig