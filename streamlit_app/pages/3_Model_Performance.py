import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
import logging

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

try:
    from src.utils import load_dataframe, load_model
    from src.evaluation import ModelEvaluator
except ImportError as e:
    st.error(f"Import error: {e}")

def render(config):
    st.title("🤖 Model Performance")
    
    try:
        # Load model results
        results_path = os.path.join(project_root, "results", "reports", "model_comparison.csv")
        model_comparison = load_dataframe(results_path)
        
        if model_comparison.empty:
            st.warning("Model performance data not available. Please run the training pipeline first.")
            if st.button("Run Training Pipeline"):
                os.system("python scripts/train_pipeline.py")
                st.rerun()
            return
        
        # Load test data for additional analysis
        X_test_path = os.path.join(project_root, "data", "processed", "test_features.csv")
        y_test_path = os.path.join(project_root, "data", "processed", "test_target.csv")
        X_test = load_dataframe(X_test_path)
        y_test = load_dataframe(y_test_path)
        
        # Main metrics overview
        st.subheader("Model Comparison Overview")
        
        # Find best model
        best_model_row = model_comparison.loc[model_comparison['ROC-AUC'].idxmax()]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best_model_row['Model'])
        with col2:
            st.metric("Best ROC-AUC", f"{best_model_row['ROC-AUC']:.4f}")
        with col3:
            st.metric("Best Accuracy", f"{best_model_row['Accuracy']:.4f}")
        with col4:
            st.metric("Best F1-Score", f"{best_model_row['F1-Score']:.4f}")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Comparison", "Detailed Metrics", "Performance Analysis", "Business Impact"
        ])
        
        with tab1:
            st.subheader("Model Performance Comparison")
            
            # Select metric for comparison
            metric_options = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            selected_metric = st.selectbox("Select metric for comparison", metric_options)
            
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            model_comparison.sort_values(selected_metric).plot.barh(
                x='Model', y=selected_metric, ax=ax, color='skyblue'
            )
            ax.set_title(f'Model Comparison - {selected_metric}')
            ax.set_xlabel(selected_metric)
            st.pyplot(fig)
            
            # Radar chart for multi-metric comparison
            st.subheader("Multi-Metric Comparison (Radar Chart)")
            
            # Normalize metrics for radar chart
            metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            normalized_data = model_comparison[metrics_for_radar].copy()
            normalized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())
            normalized_data['Model'] = model_comparison['Model']
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig_radar, ax_radar = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for idx, row in normalized_data.iterrows():
                values = row[metrics_for_radar].tolist()
                values += values[:1]  # Complete the circle
                ax_radar.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
                ax_radar.fill(angles, values, alpha=0.1)
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(metrics_for_radar)
            ax_radar.set_ylim(0, 1)
            ax_radar.legend(bbox_to_anchor=(1.1, 1.1))
            ax_radar.set_title('Model Performance Radar Chart')
            st.pyplot(fig_radar)
        
        with tab2:
            st.subheader("Detailed Model Metrics")
            
            # Select model for detailed analysis
            selected_model = st.selectbox(
                "Select model for detailed analysis", 
                model_comparison['Model'].tolist()
            )
            
            model_data = model_comparison[model_comparison['Model'] == selected_model].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Metrics gauges
                st.subheader("Performance Metrics")
                
                metrics = [
                    ('Accuracy', model_data['Accuracy'], 0, 1),
                    ('Precision', model_data['Precision'], 0, 1),
                    ('Recall', model_data['Recall'], 0, 1),
                    ('F1-Score', model_data['F1-Score'], 0, 1),
                    ('ROC-AUC', model_data['ROC-AUC'], 0, 1)
                ]
                
                for metric_name, value, min_val, max_val in metrics:
                    st.write(f"**{metric_name}**: {value:.4f}")
                    st.progress(float(value))
            
            with col2:
                st.subheader("Model Information")
                
                # Try to load model metadata
                try:
                    metadata_path = os.path.join(project_root, "models", "model_metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        st.json(metadata)
                    else:
                        st.info("Model metadata not available")
                except:
                    st.info("Could not load model metadata")
                
                # Feature importance if available
                try:
                    importance_path = os.path.join(project_root, "results", "reports", "feature_importance.csv")
                    feature_importance = load_dataframe(importance_path)
                    if not feature_importance.empty:
                        st.subheader("Top 5 Features")
                        top_features = feature_importance.head(5)
                        for _, row in top_features.iterrows():
                            st.write(f"- {row['feature']}: {row['importance']:.4f}")
                except:
                    pass
        
        with tab3:
            st.subheader("Performance Analysis")
            
            if not X_test.empty and not y_test.empty:
                # Load the best model
                try:
                    model_path = os.path.join(project_root, "models", "best_model.pkl")
                    model = load_model(model_path)
                    
                    if model:
                        # Make predictions
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Create evaluator
                        evaluator = ModelEvaluator(y_test.iloc[:, 0].values, y_pred, y_pred_proba, "Best Model")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Confusion matrix
                            st.subheader("Confusion Matrix")
                            fig_cm = evaluator.plot_confusion_matrix()
                            st.pyplot(fig_cm)
                        
                        with col2:
                            # ROC Curve
                            st.subheader("ROC Curve")
                            fig_roc = evaluator.plot_roc_curve()
                            if fig_roc:
                                st.pyplot(fig_roc)
                        
                        # Classification report
                        st.subheader("Classification Report")
                        report = evaluator.generate_classification_report()
                        st.text(report)
                        
                except Exception as e:
                    st.warning(f"Could not load model for detailed analysis: {e}")
        
        with tab4:
            st.subheader("Business Impact Analysis")
            
            st.info("""
            **Business Context**: 
            - Average customer value: $500
            - Cost of retention campaign: $50 per customer
            - Cost of lost customer: $500
            """)
            
            # Business metrics calculation
            if not X_test.empty and not y_test.empty:
                try:
                    model_path = os.path.join(project_root, "models", "best_model.pkl")
                    model = load_model(model_path)
                    
                    if model:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calculate business metrics at different thresholds
                        thresholds = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)
                        
                        # Simulate business impact
                        y_pred_business = (y_pred_proba > thresholds).astype(int)
                        
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_test, y_pred_business)
                        tn, fp, fn, tp = cm.ravel()
                        
                        # Business calculations
                        retention_cost = fp * 50  # False positives cost
                        lost_revenue = fn * 500   # False negatives cost
                        saved_revenue = tp * 500  # True positives value
                        net_benefit = saved_revenue - retention_cost - lost_revenue
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("True Positives", tp, "Customers saved")
                        with col2:
                            st.metric("False Positives", fp, "Unnecessary retention cost")
                        with col3:
                            st.metric("False Negatives", fn, "Lost customers")
                        with col4:
                            st.metric("Net Benefit", f"${net_benefit:,}", "Business value")
                        
                        # Threshold analysis
                        st.subheader("Threshold Optimization")
                        threshold_range = np.linspace(0.1, 0.9, 50)
                        net_benefits = []
                        
                        for thresh in threshold_range:
                            y_pred_temp = (y_pred_proba > thresh).astype(int)
                            cm_temp = confusion_matrix(y_test, y_pred_temp)
                            tn, fp, fn, tp = cm_temp.ravel()
                            net_benefit = (tp * 500) - (fp * 50) - (fn * 500)
                            net_benefits.append(net_benefit)
                        
                        fig_thresh, ax_thresh = plt.subplots(figsize=(10, 6))
                        ax_thresh.plot(threshold_range, net_benefits)
                        ax_thresh.axvline(x=thresholds, color='red', linestyle='--', label=f'Current: {thresholds}')
                        ax_thresh.set_xlabel('Classification Threshold')
                        ax_thresh.set_ylabel('Net Business Benefit ($)')
                        ax_thresh.set_title('Business Impact vs Classification Threshold')
                        ax_thresh.legend()
                        ax_thresh.grid(True, alpha=0.3)
                        st.pyplot(fig_thresh)
                        
                except Exception as e:
                    st.warning(f"Business impact analysis not available: {e}")
            
    except Exception as e:
        st.error(f"Error in model performance analysis: {e}")