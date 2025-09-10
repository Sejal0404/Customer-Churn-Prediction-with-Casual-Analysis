#!/usr/bin/env python3
"""
Input Data Transformer for Customer Churn Prediction API

This utility transforms user-friendly input data into the exact one-hot encoded
format expected by the trained model.
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class InputTransformer:
    """Transforms input data to match model's expected feature format"""

    def __init__(self, models_dir="models"):
        """Initialize with model artifacts"""
        self.models_dir = Path(models_dir)
        self.feature_names = None
        self.scaler = None
        self.load_model_artifacts()

    def load_model_artifacts(self):
        """Load model artifacts needed for transformation"""
        try:
            # Load feature names
            self.feature_names = joblib.load(self.models_dir / "feature_names_linear.pkl")
            print(f"Loaded {len(self.feature_names)} expected features")

            # Load scaler
            self.scaler = joblib.load(self.models_dir / "scaler_linear.pkl")
            print("Loaded scaler")

        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            raise

    def transform_input(self, input_data):
        """
        Transform user input data to model's expected format

        Args:
            input_data (dict): User-friendly input data

        Returns:
            dict: Transformed data ready for model prediction
        """
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([input_data])

        # Handle numeric columns
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Handle SeniorCitizen
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)

        # One-hot encode categorical variables to match training data
        one_hot_mappings = {
            'gender': {'Female': [0], 'Male': [1]},  # Only gender_Male is in features
            'Partner': {'No': [0], 'Yes': [1]},  # Only Partner_Yes is in features
            'Dependents': {'No': [0], 'Yes': [1]},  # Only Dependents_Yes is in features
            'PhoneService': {'No': [0], 'Yes': [1]},  # Only PhoneService_Yes is in features
            'MultipleLines': {
                'No': [1, 0],  # MultipleLines_No phone service = 1, MultipleLines_Yes = 0
                'No phone service': [1, 0],  # MultipleLines_No phone service = 1, MultipleLines_Yes = 0
                'Yes': [0, 1]  # MultipleLines_No phone service = 0, MultipleLines_Yes = 1
            },
            'InternetService': {
                'DSL': [0, 0],  # InternetService_Fiber optic = 0, InternetService_No = 0
                'Fiber optic': [1, 0],  # InternetService_Fiber optic = 1, InternetService_No = 0
                'No': [0, 1]  # InternetService_Fiber optic = 0, InternetService_No = 1
            },
            'OnlineSecurity': {
                'No': [0, 0],  # OnlineSecurity_No internet service = 0, OnlineSecurity_Yes = 0
                'No internet service': [1, 0],  # OnlineSecurity_No internet service = 1, OnlineSecurity_Yes = 0
                'Yes': [0, 1]  # OnlineSecurity_No internet service = 0, OnlineSecurity_Yes = 1
            },
            'OnlineBackup': {
                'No': [0, 0],  # OnlineBackup_No internet service = 0, OnlineBackup_Yes = 0
                'No internet service': [1, 0],  # OnlineBackup_No internet service = 1, OnlineBackup_Yes = 0
                'Yes': [0, 1]  # OnlineBackup_No internet service = 0, OnlineBackup_Yes = 1
            },
            'DeviceProtection': {
                'No': [0, 0],  # DeviceProtection_No internet service = 0, DeviceProtection_Yes = 0
                'No internet service': [1, 0],  # DeviceProtection_No internet service = 1, DeviceProtection_Yes = 0
                'Yes': [0, 1]  # DeviceProtection_No internet service = 0, DeviceProtection_Yes = 1
            },
            'TechSupport': {
                'No': [0, 0],  # TechSupport_No internet service = 0, TechSupport_Yes = 0
                'No internet service': [1, 0],  # TechSupport_No internet service = 1, TechSupport_Yes = 0
                'Yes': [0, 1]  # TechSupport_No internet service = 0, TechSupport_Yes = 1
            },
            'StreamingTV': {
                'No': [0, 0],  # StreamingTV_No internet service = 0, StreamingTV_Yes = 0
                'No internet service': [1, 0],  # StreamingTV_No internet service = 1, StreamingTV_Yes = 0
                'Yes': [0, 1]  # StreamingTV_No internet service = 0, StreamingTV_Yes = 1
            },
            'StreamingMovies': {
                'No': [0, 0],  # StreamingMovies_No internet service = 0, StreamingMovies_Yes = 0
                'No internet service': [1, 0],  # StreamingMovies_No internet service = 1, StreamingMovies_Yes = 0
                'Yes': [0, 1]  # StreamingMovies_No internet service = 0, StreamingMovies_Yes = 1
            },
            'Contract': {
                'Month-to-month': [0, 0],  # Contract_One year = 0, Contract_Two year = 0
                'One year': [1, 0],  # Contract_One year = 1, Contract_Two year = 0
                'Two year': [0, 1]  # Contract_One year = 0, Contract_Two year = 1
            },
            'PaperlessBilling': {'No': [0], 'Yes': [1]},  # Only PaperlessBilling_Yes is in features
            'PaymentMethod': {
                'Bank transfer (automatic)': [0, 0, 0],  # None of the specific payment methods
                'Credit card (automatic)': [1, 0, 0],  # PaymentMethod_Credit card (automatic) = 1
                'Electronic check': [0, 1, 0],  # PaymentMethod_Electronic check = 1
                'Mailed check': [0, 0, 1]  # PaymentMethod_Mailed check = 1
            }
        }

        # Apply one-hot encoding
        for col, mapping in one_hot_mappings.items():
            if col in df.columns:
                value = df[col].iloc[0]
                if value in mapping:
                    encoded_values = mapping[value]
                else:
                    # Default to first category
                    encoded_values = list(mapping.values())[0]

                # Create the one-hot encoded columns
                if col == 'gender':
                    df['gender_Male'] = encoded_values[0]
                elif col == 'Partner':
                    df['Partner_Yes'] = encoded_values[0]
                elif col == 'Dependents':
                    df['Dependents_Yes'] = encoded_values[0]
                elif col == 'PhoneService':
                    df['PhoneService_Yes'] = encoded_values[0]
                elif col == 'MultipleLines':
                    df['MultipleLines_No phone service'] = encoded_values[0]
                    df['MultipleLines_Yes'] = encoded_values[1]
                elif col == 'InternetService':
                    df['InternetService_Fiber optic'] = encoded_values[0]
                    df['InternetService_No'] = encoded_values[1]
                elif col == 'OnlineSecurity':
                    df['OnlineSecurity_No internet service'] = encoded_values[0]
                    df['OnlineSecurity_Yes'] = encoded_values[1]
                elif col == 'OnlineBackup':
                    df['OnlineBackup_No internet service'] = encoded_values[0]
                    df['OnlineBackup_Yes'] = encoded_values[1]
                elif col == 'DeviceProtection':
                    df['DeviceProtection_No internet service'] = encoded_values[0]
                    df['DeviceProtection_Yes'] = encoded_values[1]
                elif col == 'TechSupport':
                    df['TechSupport_No internet service'] = encoded_values[0]
                    df['TechSupport_Yes'] = encoded_values[1]
                elif col == 'StreamingTV':
                    df['StreamingTV_No internet service'] = encoded_values[0]
                    df['StreamingTV_Yes'] = encoded_values[1]
                elif col == 'StreamingMovies':
                    df['StreamingMovies_No internet service'] = encoded_values[0]
                    df['StreamingMovies_Yes'] = encoded_values[1]
                elif col == 'Contract':
                    df['Contract_One year'] = encoded_values[0]
                    df['Contract_Two year'] = encoded_values[1]
                elif col == 'PaperlessBilling':
                    df['PaperlessBilling_Yes'] = encoded_values[0]
                elif col == 'PaymentMethod':
                    df['PaymentMethod_Credit card (automatic)'] = encoded_values[0]
                    df['PaymentMethod_Electronic check'] = encoded_values[1]
                    df['PaymentMethod_Mailed check'] = encoded_values[2]

                # Remove the original column
                df = df.drop(col, axis=1)

        # Add missing features with default values
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns to match expected feature order
        df = df[self.feature_names]

        # Apply scaling to numerical features BEFORE adding engineered features
        # The scaler was trained only on basic numerical features
        scaler_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        available_scaler_features = [f for f in scaler_features if f in df.columns]

        if self.scaler is not None and available_scaler_features:
            df[available_scaler_features] = self.scaler.transform(df[available_scaler_features])

        # Add feature engineering columns AFTER scaling
        tenure = df['tenure'].iloc[0]
        monthly_charges = df['MonthlyCharges'].iloc[0]
        total_charges = df['TotalCharges'].iloc[0]

        # Calculate charge_per_tenure
        if tenure > 0:
            df['charge_per_tenure'] = total_charges / tenure
        else:
            df['charge_per_tenure'] = monthly_charges

        # Calculate charge_ratio
        if df['charge_per_tenure'].iloc[0] > 0:
            df['charge_ratio'] = monthly_charges / df['charge_per_tenure'].iloc[0]
        else:
            df['charge_ratio'] = 1

        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns to match expected feature order
        df = df[self.feature_names]

        # Convert to dictionary format for API
        result = {}
        for col in df.columns:
            value = df[col].iloc[0]
            # Handle numpy types
            if isinstance(value, (np.integer, np.int64)):
                result[col] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                result[col] = float(value)
            else:
                result[col] = value

        return result

    def get_sample_input(self):
        """Get a sample input for testing"""
        return {
            'tenure': 12,
            'MonthlyCharges': 65.5,
            'TotalCharges': 786.0,
            'Contract': 'Month-to-month',
            'InternetService': 'Fiber optic',
            'PaymentMethod': 'Electronic check',
            'gender': 'Female',
            'SeniorCitizen': 'No',
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'PaperlessBilling': 'Yes'
        }

def main():
    """Command line interface for the transformer"""
    import argparse

    parser = argparse.ArgumentParser(description='Transform input data for Customer Churn Prediction API')
    parser.add_argument('--input', '-i', type=str, help='JSON string or file path containing input data')
    parser.add_argument('--output', '-o', type=str, help='Output file path (optional)')
    parser.add_argument('--sample', action='store_true', help='Use sample input data')

    args = parser.parse_args()

    # Initialize transformer
    transformer = InputTransformer()

    # Get input data
    if args.sample:
        input_data = transformer.get_sample_input()
        print("Using sample input data:")
        print(json.dumps(input_data, indent=2))
    elif args.input:
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(args.input)
    else:
        print("Please provide input data using --input or use --sample")
        return

    # Transform data
    transformed_data = transformer.transform_input(input_data)

    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(transformed_data, f, indent=2)
        print(f"Transformed data saved to {args.output}")
    else:
        print("Transformed data:")
        print(json.dumps(transformed_data, indent=2))

if __name__ == "__main__":
    main()
