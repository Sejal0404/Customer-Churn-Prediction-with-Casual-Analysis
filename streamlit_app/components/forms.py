import streamlit as st
import pandas as pd

def create_customer_input_form(config):
    """Create a form for customer data input"""
    
    with st.form("customer_input_form"):
        st.subheader("Customer Information")
        
        # Personal information
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
        
        with col2:
            partner = st.radio("Partner", ["No", "Yes"])
            dependents = st.radio("Dependents", ["No", "Yes"])
        
        # Account information
        st.subheader("Account Information")
        
        col3, col4 = st.columns(2)
        
        with col3:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        with col4:
            paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        
        # Service information
        st.subheader("Service Information")
        
        col5, col6 = st.columns(2)
        
        with col5:
            phone_service = st.radio("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with col6:
            monthly_charges = st.slider("Monthly Charges ($)", 0, 200, 50)
            total_charges = st.number_input("Total Charges ($)", 0, 10000, tenure * monthly_charges)
        
        # Additional services
        st.subheader("Additional Services")
        
        col7, col8 = st.columns(2)
        
        with col7:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        
        with col8:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
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
            
            return customer_data
    
    return None

def create_batch_upload_form():
    """Create a form for batch file upload"""
    
    st.subheader("Batch Prediction Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=['csv'],
        help="File should contain columns matching the training data features"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(df)} records")
            
            # Show data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
            
            # Validate required columns
            required_columns = ['tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return None
            else:
                return df
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    return None

def create_policy_simulation_form():
    """Create a form for policy simulation"""
    
    with st.form("policy_simulation_form"):
        st.subheader("Policy Simulation Parameters")
        
        policy_type = st.selectbox(
            "Policy Type",
            ["Contract Incentives", "Payment Method Optimization", "Pricing Strategy", "Service Bundles"]
        )
        
        if policy_type == "Contract Incentives":
            current_mtm = st.slider("Current month-to-month percentage", 0, 100, 50)
            target_mtm = st.slider("Target month-to-month percentage", 0, 100, 40)
            incentive_amount = st.number_input("Incentive amount ($)", 0, 100, 20)
        
        elif policy_type == "Payment Method Optimization":
            current_auto = st.slider("Current automatic payment percentage", 0, 100, 30)
            target_auto = st.slider("Target automatic payment percentage", 0, 100, 50)
            discount_percentage = st.slider("Discount percentage", 0, 20, 5)
        
        elif policy_type == "Pricing Strategy":
            price_change = st.slider("Price change percentage", -20, 20, 0)
            affected_segment = st.selectbox("Affected segment", ["All", "New", "Existing", "High-risk"])
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_horizon = st.selectbox("Time horizon", ["3 months", "6 months", "1 year", "2 years"])
            confidence_level = st.slider("Confidence level", 80, 99, 95)
        
        with col2:
            customer_segment = st.multiselect(
                "Customer segments to include",
                ["New", "Existing", "High-value", "At-risk"],
                default=["New", "Existing"]
            )
        
        submitted = st.form_submit_button("Run Simulation")
        
        if submitted:
            simulation_params = {
                'policy_type': policy_type,
                'parameters': locals()  # Capture all local variables
            }
            return simulation_params
    
    return None