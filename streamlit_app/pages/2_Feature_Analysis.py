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
    from src.utils import load_dataframe
    from src.feature_engineering import FeatureEngineer
except ImportError as e:
    st.error(f"Import error: {e}")

def render(config):
    st.title("🔍 Feature Analysis")
    
    try:
        # Load data and feature importance
        data_path = os.path.join(project_root, "data", "processed", "train_features.csv")
        importance_path = os.path.join(project_root, "results", "reports", "feature_importance.csv")
        
        X_train = load_dataframe(data_path)
        feature_importance = load_dataframe(importance_path)
        
        if X_train.empty or feature_importance.empty:
            st.warning("Feature analysis data not available. Please run the training pipeline first.")
            if st.button("Run Training Pipeline"):
                os.system("python scripts/train_pipeline.py")
                st.rerun()
            return
        
        # Load target for correlation analysis
        target_path = os.path.join(project_root, "data", "processed", "train_target.csv")
        y_train = load_dataframe(target_path)
        
        # Display overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(X_train.columns))
        with col2:
            st.metric("Top Feature", feature_importance.iloc[0]['feature'])
        with col3:
            st.metric("Top Feature Importance", f"{feature_importance.iloc[0]['importance']:.4f}")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Feature Importance", "Correlation Analysis", "Feature Distributions", "Interactive Analysis"
        ])
        
        with tab1:
            st.subheader("Feature Importance Ranking")
            
            top_n = st.slider("Number of top features to show", 5, 50, 15)
            top_features = feature_importance.head(top_n)
            
            # Feature importance plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis', ax=ax)
            ax.set_title(f'Top {top_n} Most Important Features')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)
            
            # Feature importance table
            st.subheader("Feature Importance Table")
            display_df = top_features.copy()
            display_df['importance'] = display_df['importance'].round(4)
            display_df['cumulative_importance'] = display_df['importance'].cumsum()
            st.dataframe(display_df, use_container_width=True)
            
            # Cumulative importance
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(range(1, len(display_df) + 1), display_df['cumulative_importance'], marker='o')
            ax2.set_xlabel('Number of Features')
            ax2.set_ylabel('Cumulative Importance')
            ax2.set_title('Cumulative Feature Importance')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        
        with tab2:
            st.subheader("Correlation Analysis")
            
            # Calculate correlations with target
            if not y_train.empty:
                correlation_with_target = X_train.corrwith(y_train.iloc[:, 0]).sort_values(ascending=False)
                correlation_df = pd.DataFrame({
                    'feature': correlation_with_target.index,
                    'correlation': correlation_with_target.values
                }).sort_values('correlation', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Positive Correlations**")
                    top_positive = correlation_df.head(10)
                    fig_pos = plt.figure(figsize=(8, 6))
                    plt.barh(top_positive['feature'], top_positive['correlation'])
                    plt.xlabel('Correlation with Churn')
                    plt.title('Features Most Positively Correlated with Churn')
                    st.pyplot(fig_pos)
                
                with col2:
                    st.write("**Top Negative Correlations**")
                    top_negative = correlation_df.tail(10).iloc[::-1]
                    fig_neg = plt.figure(figsize=(8, 6))
                    plt.barh(top_negative['feature'], top_negative['correlation'])
                    plt.xlabel('Correlation with Churn')
                    plt.title('Features Most Negatively Correlated with Churn')
                    st.pyplot(fig_neg)
            
            # Correlation heatmap
            st.subheader("Feature Correlation Heatmap")
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                # Select features for heatmap
                selected_features = st.multiselect(
                    "Select features for correlation matrix",
                    options=numeric_cols.tolist(),
                    default=numeric_cols[:8].tolist()  # First 8 numeric features
                )
                
                if len(selected_features) > 1:
                    corr_matrix = X_train[selected_features].corr()
                    
                    fig_heat = plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                    plt.title('Feature Correlation Heatmap')
                    st.pyplot(fig_heat)
        
        with tab3:
            st.subheader("Feature Distributions by Churn Status")
            
            # Select feature to analyze
            feature_to_analyze = st.selectbox(
                "Select feature to analyze",
                options=X_train.columns.tolist()
            )
            
            if feature_to_analyze and not y_train.empty:
                # Create combined dataframe for plotting
                plot_data = pd.concat([X_train[feature_to_analyze], y_train], axis=1)
                plot_data.columns = ['feature', 'churn']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    fig_dist = plt.figure(figsize=(8, 6))
                    for churn_status in [0, 1]:
                        subset = plot_data[plot_data['churn'] == churn_status]
                        plt.hist(subset['feature'], alpha=0.7, 
                                label=f'Churn = {churn_status}', bins=30)
                    plt.xlabel(feature_to_analyze)
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.title(f'Distribution of {feature_to_analyze} by Churn Status')
                    st.pyplot(fig_dist)
                
                with col2:
                    # Box plot
                    fig_box = plt.figure(figsize=(8, 6))
                    sns.boxplot(data=plot_data, x='churn', y='feature', 
                               palette=['lightblue', 'lightcoral'])
                    plt.title(f'{feature_to_analyze} Distribution by Churn Status')
                    plt.xlabel('Churn Status')
                    plt.ylabel(feature_to_analyze)
                    st.pyplot(fig_box)
        
        with tab4:
            st.subheader("Interactive Feature Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature", X_train.columns.tolist(), index=0)
            with col2:
                y_feature = st.selectbox("Y-axis feature", X_train.columns.tolist(), 
                                       index=min(1, len(X_train.columns)-1))
            
            if not y_train.empty:
                # Scatter plot with churn coloring
                plot_data = pd.concat([X_train[[x_feature, y_feature]], y_train], axis=1)
                plot_data.columns = ['x', 'y', 'churn']
                
                fig_scatter = plt.figure(figsize=(10, 8))
                colors = ['lightblue' if c == 0 else 'lightcoral' for c in plot_data['churn']]
                plt.scatter(plot_data['x'], plot_data['y'], c=colors, alpha=0.6)
                plt.xlabel(x_feature)
                plt.ylabel(y_feature)
                plt.title(f'{x_feature} vs {y_feature} colored by Churn Status')
                
                # Create legend
                import matplotlib.patches as mpatches
                blue_patch = mpatches.Patch(color='lightblue', label='No Churn')
                red_patch = mpatches.Patch(color='lightcoral', label='Churn')
                plt.legend(handles=[blue_patch, red_patch])
                
                st.pyplot(fig_scatter)
            
    except Exception as e:
        st.error(f"Error in feature analysis: {e}")