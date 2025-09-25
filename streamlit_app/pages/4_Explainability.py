import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

try:
    from src.utils import load_dataframe, load_model
    from src.explainability import SHAPExplainer
except ImportError as e:
    st.error(f"Import error: {e}")

def render(config):
    st.title("💡 Model Explainability")
    
    try:
        # Load data and model
        X_test_path = os.path.join(project_root, "data", "processed", "test_features.csv")
        X_train_path = os.path.join(project_root, "data", "processed", "train_features.csv")
        y_test_path = os.path.join(project_root, "data", "processed", "test_target.csv")
        model_path = os.path.join(project_root, "models", "best_model.pkl")
        preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")
        
        X_test = load_dataframe(X_test_path)
        X_train = load_dataframe(X_train_path)
        y_test = load_dataframe(y_test_path)
        model = load_model(model_path)
        preprocessor = load_model(preprocessor_path)
        
        if X_test.empty or model is None:
            st.warning("Model or test data not available. Please run the training pipeline first.")
            if st.button("Run Training Pipeline"):
                os.system("python scripts/train_pipeline.py")
                st.rerun()
            return
        
        # Load SHAP importance if available
        shap_path = os.path.join(project_root, "results", "reports", "shap_importance.csv")
        shap_importance = load_dataframe(shap_path)
        
        # Initialize SHAP explainer
        try:
            explainer = SHAPExplainer(model, preprocessor, X_train.columns.tolist())
            explainer.create_explainer(X_train)
            shap_available = True
        except Exception as e:
            st.warning(f"SHAP analysis not available: {e}")
            shap_available = False
        
        # Tabs for different explainability views
        tab1, tab2, tab3 = st.tabs([
            "Global Explanations", "Local Explanations", "Feature Analysis"
        ])
        
        with tab1:
            st.subheader("Global Feature Importance")
            
            if shap_available and not shap_importance.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # SHAP summary plot
                    top_n_global = st.slider("Number of features to show", 5, 30, 15, key="global_slider")
                    
                    try:
                        fig_summary = explainer.summary_plot(X_test, max_display=top_n_global, show=False)
                        st.pyplot(fig_summary)
                    except Exception as e:
                        st.warning(f"SHAP summary plot not available: {e}")
                
                with col2:
                    st.write("**Top Features**")
                    top_features = shap_importance.head(5)
                    for idx, row in top_features.iterrows():
                        st.metric(f"Top {idx+1}", row['feature'], f"{row['shap_importance']:.4f}")
                    
                    st.write("**SHAP Statistics**")
                    total_shap = shap_importance['shap_importance'].sum()
                    st.metric("Total SHAP Impact", f"{total_shap:.4f}")
                    st.metric("Average Feature Impact", f"{shap_importance['shap_importance'].mean():.4f}")
            
            else:
                st.info("SHAP analysis not available. Using feature importance from model.")
                # Fallback to model feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    top_n = st.slider("Number of features to show", 5, 30, 15)
                    top_features = feature_importance.head(top_n)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
                    ax.set_title('Feature Importance (Model-based)')
                    st.pyplot(fig)
        
        with tab2:
            st.subheader("Individual Prediction Explanations")
            
            if shap_available:
                # Select instance to explain
                max_instance = min(100, len(X_test) - 1)
                instance_idx = st.slider("Select customer instance", 0, max_instance, 0)
                
                # Get prediction details
                instance_data = X_test.iloc[instance_idx:instance_idx+1]
                prediction = model.predict(instance_data)[0]
                prediction_proba = model.predict_proba(instance_data)[0][1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Churn", "Yes" if prediction == 1 else "No")
                with col2:
                    st.metric("Churn Probability", f"{prediction_proba:.4f}")
                with col3:
                    confidence = abs(prediction_proba - 0.5) * 2
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Force plot
                st.subheader("SHAP Force Plot")
                try:
                    fig_force = explainer.force_plot(instance_idx, X_test, show=False)
                    if fig_force:
                        st.pyplot(fig_force)
                    else:
                        st.info("Force plot not available for this model type.")
                except Exception as e:
                    st.warning(f"Force plot not available: {e}")
                
                # Feature contributions
                st.subheader("Feature Contributions")
                try:
                    # Get SHAP values for this instance
                    shap_values = explainer.shap_values
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_instance = shap_values[1][instance_idx]
                    else:
                        shap_instance = shap_values[instance_idx]
                    
                    contributions = pd.DataFrame({
                        'Feature': X_test.columns,
                        'SHAP Value': shap_instance,
                        'Feature Value': X_test.iloc[instance_idx].values,
                        'Absolute Impact': np.abs(shap_instance)
                    }).sort_values('Absolute Impact', ascending=False)
                    
                    # Display top contributors - Fix Arrow serialization
                    st.write("Top features influencing this prediction:")
                    top_contributors = contributions.head(10).copy()
                    top_contributors['Impact Direction'] = top_contributors['SHAP Value'].apply(
                        lambda x: 'Increases Churn Risk' if x > 0 else 'Decreases Churn Risk'
                    )
                    
                    # Format for display - Fix data types for Arrow compatibility
                    display_df = top_contributors[['Feature', 'Feature Value', 'SHAP Value', 'Impact Direction']].copy()
                    display_df['SHAP Value'] = display_df['SHAP Value'].round(4)
                    display_df['Feature Value'] = display_df['Feature Value'].round(4)
                    
                    st.dataframe(display_df, width='stretch')
                    
                except Exception as e:
                    st.warning(f"Feature contributions not available: {e}")
            
            else:
                st.info("Individual explanations require SHAP analysis.")
        
        with tab3:
            st.subheader("Feature Behavior Analysis")
            
            # Feature dependence analysis
            if shap_available and not X_test.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    feature_to_analyze = st.selectbox(
                        "Select feature for analysis",
                        X_test.columns.tolist()
                    )
                
                with col2:
                    interaction_feature = st.selectbox(
                        "Interaction feature (optional)",
                        [None] + X_test.columns.tolist()
                    )
                
                if feature_to_analyze:
                    st.subheader(f"Feature Behavior: {feature_to_analyze}")
                    
                    # Basic distribution
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Distribution plot
                    ax1.hist(X_test[feature_to_analyze], bins=30, alpha=0.7)
                    ax1.set_xlabel(feature_to_analyze)
                    ax1.set_ylabel('Frequency')
                    ax1.set_title(f'Distribution of {feature_to_analyze}')
                    
                    # Relationship with predictions
                    predictions = model.predict_proba(X_test)[:, 1]
                    ax2.scatter(X_test[feature_to_analyze], predictions, alpha=0.5)
                    ax2.set_xlabel(feature_to_analyze)
                    ax2.set_ylabel('Churn Probability')
                    ax2.set_title(f'Churn Probability vs {feature_to_analyze}')
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # SHAP dependence plot
                    st.subheader("SHAP Dependence Plot")
                    try:
                        fig_dependence = explainer.dependence_plot(
                            feature_to_analyze, 
                            X_test, 
                            interaction_index=interaction_feature,
                            show=False
                        )
                        st.pyplot(fig_dependence)
                    except Exception as e:
                        st.warning(f"Dependence plot not available: {e}")
                    
                    # Feature statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{X_test[feature_to_analyze].mean():.4f}")
                    with col2:
                        st.metric("Std Dev", f"{X_test[feature_to_analyze].std():.4f}")
                    with col3:
                        st.metric("Min", f"{X_test[feature_to_analyze].min():.4f}")
                    with col4:
                        st.metric("Max", f"{X_test[feature_to_analyze].max():.4f}")
            
            else:
                st.info("Feature analysis requires SHAP analysis and test data.")
                
    except Exception as e:
        st.error(f"Error loading explainability data: {e}")
        st.info("Please ensure the training pipeline has been run successfully.")