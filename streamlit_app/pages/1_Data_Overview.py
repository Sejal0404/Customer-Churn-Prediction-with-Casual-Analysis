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
    from src.data_preprocessing import DataPreprocessor
except ImportError as e:
    st.error(f"Import error: {e}")

def render(config):
    st.title("📊 Data Overview")
    
    try:
        # Load data
        data_path = os.path.join(project_root, "data", "interim", "preprocessed_data.csv")
        df = load_dataframe(data_path)
        
        if df.empty:
            st.warning("No data available. Please run the training pipeline first.")
            if st.button("Run Training Pipeline"):
                os.system("python scripts/train_pipeline.py")
                st.rerun()
            return
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            if df['Churn'].dtype == 'object':
                churn_rate = (df['Churn'] == 'Yes').mean()
            else:
                churn_rate = df['Churn'].mean()
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        with col3:
            st.metric("Number of Features", len(df.columns) - 1)
        with col4:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Statistics", "Data Quality", "Target Analysis"])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            st.subheader("Numerical Features Statistics")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                st.dataframe(df[numerical_cols].describe(), use_container_width=True)
            else:
                st.info("No numerical features found")
            
            st.subheader("Categorical Features Summary")
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:8]:  # Show first 8
                    with st.expander(f"{col} ({df[col].nunique()} unique values)"):
                        value_counts = df[col].value_counts()
                        st.write(value_counts)
                        fig, ax = plt.subplots(figsize=(8, 4))
                        value_counts.head(10).plot(kind='bar', ax=ax)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
            else:
                st.info("No categorical features found")
        
        with tab3:
            st.subheader("Data Quality Assessment")
            
            # Missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.warning("Missing values detected!")
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': (missing_data / len(df)) * 100
                }).sort_values('Missing Count', ascending=False)
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.warning(f"⚠️ {duplicates} duplicate rows found")
            else:
                st.success("✅ No duplicate rows found")
            
            # Constant columns
            constant_cols = [col for col in df.columns if df[col].nunique() == 1]
            if constant_cols:
                st.warning(f"⚠️ Constant columns: {constant_cols}")
            else:
                st.success("✅ No constant columns found")
        
        with tab4:
            st.subheader("Target Variable Analysis")
            
            # Churn distribution
            if df['Churn'].dtype == 'object':
                churn_counts = df['Churn'].value_counts()
                churn_percentage = df['Churn'].value_counts(normalize=True) * 100
            else:
                churn_counts = df['Churn'].value_counts()
                churn_percentage = df['Churn'].value_counts(normalize=True) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Churn Cases", int(churn_counts.get(1, churn_counts.get('Yes', 0))))
                st.metric("Non-Churn Cases", int(churn_counts.get(0, churn_counts.get('No', 0))))
            
            with col2:
                st.metric("Churn Rate", f"{churn_percentage.get(1, churn_percentage.get('Yes', 0)):.1f}%")
                st.metric("Non-Churn Rate", f"{churn_percentage.get(0, churn_percentage.get('No', 0)):.1f}%")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Count plot
            if df['Churn'].dtype == 'object':
                churn_counts.plot(kind='bar', ax=ax1, color=['lightblue', 'lightcoral'])
            else:
                churn_counts.plot(kind='bar', ax=ax1, color=['lightblue', 'lightcoral'])
            ax1.set_title('Churn Distribution')
            ax1.set_ylabel('Count')
            
            # Pie chart
            if df['Churn'].dtype == 'object':
                labels = ['No Churn', 'Churn'] if 'No' in churn_counts.index else ['0', '1']
                ax2.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            else:
                ax2.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            ax2.set_title('Churn Percentage')
            
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please make sure the training pipeline has been run successfully.")