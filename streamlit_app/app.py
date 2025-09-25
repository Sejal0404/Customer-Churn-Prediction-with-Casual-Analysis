import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.utils import load_config, setup_logging
    logger.info("Successfully imported src modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error(f"Module import error: {e}")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS"""
    try:
        css_path = os.path.join(project_root, "streamlit_app", "assets", "style.css")
        if os.path.exists(css_path):
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            logger.info("CSS loaded successfully")
        else:
            logger.warning("CSS file not found")
    except Exception as e:
        logger.error(f"Error loading CSS: {e}")

def load_page_module(page_name):
    """Dynamically load page modules with error handling"""
    try:
        page_path = f"streamlit_app.pages.{page_name}"
        module = __import__(page_path, fromlist=['render'])
        return module
    except Exception as e:
        logger.error(f"Error loading page {page_name}: {e}")
        return None

def main():
    # Load configuration
    try:
        config_path = os.path.join(project_root, "config", "params.yaml")
        config = load_config(config_path)
        setup_logging()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        config = {}
        logger.error(f"Config error: {e}")
    
    # Load CSS
    load_css()
    
    # Sidebar
    st.sidebar.title("📊 Customer Churn Analysis")
    st.sidebar.markdown("---")
    
    # Display project info
    st.sidebar.markdown("""
    **Project Info:**
    - 🎯 **Goal**: Predict customer churn
    - 📈 **Best Model**: XGBoost (AUC: 0.845)
    - 👥 **Customers**: 7,043
    - 🔥 **Churn Rate**: 26.5%
    """)
    
    # Navigation
    page_options = {
        "1_Data_Overview": "📊 Data Overview",
        "2_Feature_Analysis": "🔍 Feature Analysis", 
        "3_Model_Performance": "🤖 Model Performance",
        "4_Explainability": "💡 Explainability",
        "5_Churn_Prediction": "🔮 Churn Prediction",
        "6_Causal_Analysis": "🧠 Causal Analysis"
    }
    
    selected_page = st.sidebar.radio(
        "Navigate to:",
        list(page_options.values())
    )
    
    # Get page key from selected value
    page_key = [k for k, v in page_options.items() if v == selected_page][0]
    
    # Page routing with robust error handling
    try:
        module = load_page_module(page_key)
        if module and hasattr(module, 'render'):
            module.render(config)
        else:
            st.error(f"Page {page_key} not loaded properly")
            show_fallback_interface(config)
    except Exception as e:
        st.error(f"Error rendering page {page_key}: {e}")
        logger.error(f"Page rendering error: {e}")
        show_fallback_interface(config)

def show_fallback_interface(config):
    """Show a fallback interface when pages fail to load"""
    st.title("🚧 Customer Churn Analysis Dashboard")
    st.warning("Some components failed to load. Showing basic interface.")
    
    st.subheader("Project Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", "7,043")
        st.metric("Churn Rate", "26.5%")
    
    with col2:
        st.metric("Best Model AUC", "0.845")
        st.metric("Accuracy", "80.6%")
    
    with col3:
        st.metric("Features", "46")
        st.metric("Models Trained", "3")
    
    st.subheader("Available Functions")
    
    if st.button("📁 View Data Summary"):
        show_data_summary()
    
    if st.button("📈 View Model Results"):
        show_model_results()
    
    if st.button("🔮 Make Prediction"):
        show_prediction_interface()

def show_data_summary():
    """Show basic data summary"""
    try:
        data_path = os.path.join(project_root, "data", "interim", "preprocessed_data.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.write("Data Summary:")
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", df.columns.tolist())
            st.write("Churn Distribution:", df['Churn'].value_counts().to_dict())
        else:
            st.error("Data file not found")
    except Exception as e:
        st.error(f"Error loading data: {e}")

def show_model_results():
    """Show basic model results"""
    try:
        results_path = os.path.join(project_root, "results", "reports", "model_comparison.csv")
        if os.path.exists(results_path):
            results = pd.read_csv(results_path)
            st.write("Model Performance:")
            st.dataframe(results)
        else:
            st.info("Run training pipeline to generate model results")
    except Exception as e:
        st.error(f"Error loading results: {e}")

def show_prediction_interface():
    """Simple prediction interface"""
    st.subheader("Simple Churn Prediction")
    st.info("Full prediction interface requires model loading")
    
    # Simple form for demonstration
    with st.form("prediction_form"):
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges", 0, 200, 70)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        if st.form_submit_button("Estimate Churn Risk"):
            # Simple heuristic based on known patterns
            risk_score = 0.3  # Base risk
            
            if contract == "Month-to-month":
                risk_score += 0.3
            elif contract == "One year":
                risk_score += 0.1
                
            if tenure < 12:
                risk_score += 0.2
                
            if monthly_charges > 70:
                risk_score += 0.1
                
            risk_score = min(risk_score, 0.9)  # Cap at 90%
            
            st.success(f"Estimated churn risk: {risk_score:.1%}")
            if risk_score > 0.5:
                st.error("High risk customer - recommend retention actions")
            elif risk_score > 0.3:
                st.warning("Medium risk - monitor closely")
            else:
                st.success("Low risk customer")

if __name__ == "__main__":
    main()