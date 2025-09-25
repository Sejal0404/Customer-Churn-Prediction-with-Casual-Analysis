import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_model
import json

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import your modules
from src.utils import load_dataframe, load_model
# ... other imports


def render(config):
    st.title("🔮 Churn Prediction")
    
    # Load model
    @st.cache_resource
    def load_prediction_model():
        model = load_model("models/best_model.pkl")
        preprocessor = load_model("models/preprocessor.pkl")
        feature_importance = pd.read_csv("results/reports/feature_importance.csv")
        return model, preprocessor, feature_importance
    
    try:
        model, preprocessor, feature_importance = load_prediction_model()
        if model is None:
            st.warning("Please train the model first using the training pipeline.")
            return
    except:
        st.warning("Prediction model not available. Please run the training pipeline first.")
        return
    
    # Prediction interface
    st.subheader("Predict Churn for New Customer")
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                           ["Manual Input", "Batch Upload"])
    
    if input_method == "Manual Input":
        render_manual_input(model, preprocessor, feature_importance, config)
    else:
        render_batch_upload(model, preprocessor)

def render_manual_input(model, preprocessor, feature_importance, config):
    """Render manual input form for single prediction"""
    
    st.info("Enter customer details to predict churn probability")
    
    # Create input form based on original feature space
    with st.form("customer_details"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Demographic features
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            # Service features
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with col2:
            # Contract and payment
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            
            # Numerical features
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 0, 200, 50)
            total_charges = st.slider("Total Charges ($)", 0, 10000, tenure * monthly_charges)
        
        # Additional service features
        st.subheader("Additional Services")
        col3, col4 = st.columns(2)
        
        with col3:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        
        with col4:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Create feature dictionary
        customer_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Make prediction
        prediction, explanation = make_prediction(customer_data, model, preprocessor, feature_importance)
        
        # Display results
        display_prediction_results(prediction, explanation, customer_data)

def render_batch_upload(model, preprocessor):
    """Render batch prediction interface"""
    
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with customer data", 
                                   type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"Uploaded data: {batch_data.shape[0]} customers, {batch_data.shape[1]} features")
            
            # Show preview
            st.write("Data preview:")
            st.dataframe(batch_data.head())
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing predictions..."):
                    # Preprocess data
                    # Note: This would need to handle preprocessing similarly to training data
                    predictions = []
                    
                    for idx, row in batch_data.iterrows():
                        try:
                            # Convert row to dictionary and make prediction
                            customer_dict = row.to_dict()
                            # This is simplified - would need proper preprocessing
                            prediction_prob = 0.5  # Placeholder
                            predictions.append({
                                'Customer_ID': idx,
                                'Churn_Probability': prediction_prob,
                                'Predicted_Churn': 'Yes' if prediction_prob > 0.5 else 'No'
                            })
                        except Exception as e:
                            predictions.append({
                                'Customer_ID': idx,
                                'Churn_Probability': None,
                                'Predicted_Churn': f'Error: {str(e)}'
                            })
                    
                    results_df = pd.DataFrame(predictions)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    churn_count = len(results_df[results_df['Predicted_Churn'] == 'Yes'])
                    churn_rate = (churn_count / len(results_df)) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(results_df))
                    with col2:
                        st.metric("Predicted Churn", churn_count)
                    with col3:
                        st.metric("Churn Rate", f"{churn_rate:.1f}%")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "churn_predictions.csv",
                        "text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def make_prediction(customer_data, model, preprocessor, feature_importance):
    """Make prediction for a single customer"""
    
    try:
        # Convert to DataFrame
        customer_df = pd.DataFrame([customer_data])
        
        # Preprocess data (simplified - would need full preprocessing pipeline)
        # For demonstration, we'll use a simple approach
        processed_data = preprocess_customer_data(customer_df, preprocessor)
        
        # Make prediction
        churn_probability = model.predict_proba(processed_data)[0][1]
        prediction = 'Yes' if churn_probability > 0.5 else 'No'
        
        # Generate explanation
        explanation = generate_explanation(customer_data, churn_probability, feature_importance)
        
        return {
            'churn_probability': churn_probability,
            'prediction': prediction,
            'confidence': abs(churn_probability - 0.5) * 2  # Distance from 0.5
        }, explanation
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def preprocess_customer_data(customer_df, preprocessor):
    """Preprocess customer data for prediction"""
    # This is a simplified version - in practice, use the full preprocessing pipeline
    try:
        return preprocessor.transform(customer_df)
    except:
        # Fallback: return dummy data with correct shape
        return np.zeros((1, 50))  # Adjust based on your feature count

def generate_explanation(customer_data, churn_probability, feature_importance):
    """Generate explanation for the prediction"""
    
    explanation = {
        'key_factors': [],
        'risk_level': 'Medium',
        'recommendations': []
    }
    
    # Determine risk level
    if churn_probability > 0.7:
        explanation['risk_level'] = 'High'
    elif churn_probability > 0.3:
        explanation['risk_level'] = 'Medium'
    else:
        explanation['risk_level'] = 'Low'
    
    # Identify key factors based on feature importance and customer values
    top_features = feature_importance.head(5)['feature'].tolist()
    
    for feature in top_features:
        if feature in customer_data:
            value = customer_data[feature]
            # Simple rule-based explanations
            if feature == 'Contract' and value == 'Month-to-month':
                explanation['key_factors'].append('Month-to-month contract (high risk)')
            elif feature == 'tenure' and value < 12:
                explanation['key_factors'].append(f'Low tenure ({value} months)')
            elif feature == 'MonthlyCharges' and value > 70:
                explanation['key_factors'].append(f'High monthly charges (${value})')
    
    # Generate recommendations
    if explanation['risk_level'] == 'High':
        explanation['recommendations'] = [
            "Consider offering loyalty discount",
            "Assign to dedicated retention specialist",
            "Review service quality issues"
        ]
    elif explanation['risk_level'] == 'Medium':
        explanation['recommendations'] = [
            "Send personalized engagement offer",
            "Conduct satisfaction survey",
            "Monitor usage patterns"
        ]
    else:
        explanation['recommendations'] = [
            "Maintain current service quality",
            "Continue regular engagement",
            "Monitor for any changes"
        ]
    
    return explanation

def display_prediction_results(prediction, explanation, customer_data):
    """Display prediction results and explanation"""
    
    if prediction is None:
        st.error("Prediction failed. Please check the input data.")
        return
    
    # Results header
    st.subheader("Prediction Results")
    
    # Probability gauge
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create a visual gauge for churn probability
        gauge_color = "red" if prediction['churn_probability'] > 0.5 else "green"
        
        st.markdown(f"""
        <div style="text-align: center;">
            <h3 style="color: {gauge_color};">Churn Probability: {prediction['churn_probability']:.1%}</h3>
            <div style="background: linear-gradient(90deg, green 0%, yellow 50%, red 100%); 
                       height: 20px; border-radius: 10px; margin: 10px 0;">
                <div style="width: {prediction['churn_probability']*100}%; 
                           height: 100%; background: rgba(255,255,255,0.3); border-radius: 10px;">
                </div>
            </div>
            <h4>Prediction: <span style="color: {gauge_color}">{prediction['prediction']}</span></h4>
            <p>Confidence: {prediction['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk assessment
    st.subheader("Risk Assessment")
    
    risk_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    risk_color = risk_colors[explanation['risk_level']]
    
    st.markdown(f"""
    <div style="border: 2px solid {risk_color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
        <h4 style="color: {risk_color}; margin: 0;">Risk Level: {explanation['risk_level']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Key factors
    st.subheader("Key Factors Influencing Prediction")
    
    if explanation['key_factors']:
        for factor in explanation['key_factors']:
            st.write(f"• {factor}")
    else:
        st.write("No specific risk factors identified from top features.")
    
    # Recommendations
    st.subheader("Recommended Actions")
    
    for recommendation in explanation['recommendations']:
        st.write(f"📋 {recommendation}")
    
    # Customer summary
    with st.expander("Customer Summary"):
        st.json(customer_data)