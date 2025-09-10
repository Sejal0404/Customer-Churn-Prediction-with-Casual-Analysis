from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import config

app = Flask(__name__)

# Global variables
model = None
feature_names = None
scaler = None
label_encoders = None
model_loaded = False


def load_model():
    """Load model and preprocessing objects"""
    global model, feature_names, scaler, label_encoders, model_loaded
    
    try:
        # Load best model (logistic regression)
        model = joblib.load(config.MODELS_DIR / config.MODEL_FILES['best_model'])
        
        # Load feature names for TREE models (since best model is XGBoost)
        feature_names = joblib.load(config.MODELS_DIR / config.MODEL_FILES['feature_names_tree'])

        # Load scaler for TREE models (since best model is XGBoost)
        scaler = joblib.load(config.MODELS_DIR / config.MODEL_FILES['scaler_tree'])
        
        # Load label encoders
        label_encoders = joblib.load(config.MODELS_DIR / config.MODEL_FILES['label_encoders'])
        
        model_loaded = True
        print("Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Model expects {len(feature_names)} features")
        print(f"Model classes: {model.classes_}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
        return False
    
@app.before_request
def before_request():
    """Load model before the first request if not already loaded"""
    global model_loaded
    if not model_loaded:
        load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': config.API_MESSAGES['model_not_loaded']}), 500
    
    try:
        data = request.get_json()
        print(f"Received data: {data}") # DEBUG
        input_data = pd.DataFrame([data])
        processed_data = preprocess_input(input_data)
        
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # DEBUG: Print raw model output
        print(f"Raw model prediction: {prediction}")
        print(f"Raw model probabilities: {prediction_proba}")
        print(f"Model classes: {model.classes_}") # This should be [0, 1]

        # Check if the model's classes are [0, 1]
        if list(model.classes_) == [0, 1]:
            churn_probability = float(prediction_proba[0][1]) # Probability for class 1 (Churn)
        else:
            # Fallback: find the index of class 1
            churn_class_index = list(model.classes_).index(1)
            churn_probability = float(prediction_proba[0][churn_class_index])
        
        response = {
            'prediction': int(prediction[0]),
            'probability': churn_probability,
            'probability_class_0': float(prediction_proba[0][0]), # DEBUG
            'probability_class_1': float(prediction_proba[0][1]), # DEBUG
            'churn_risk': 'High' if prediction[0] == 1 else 'Low',
            'message': 'Customer is at high risk of churning' if prediction[0] == 1 else 'Customer is at low risk of churning'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def preprocess_input(input_data):
    """Preprocess input data"""
    processed = input_data.copy()

    # Convert string inputs to appropriate types
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_columns:
        if col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors='coerce')
            processed[col].fillna(0, inplace=True)

    # Handle SeniorCitizen (convert 'Yes'/'No' to 1/0)
    if 'SeniorCitizen' in processed.columns:
        processed['SeniorCitizen'] = processed['SeniorCitizen'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        processed['SeniorCitizen'].fillna(0, inplace=True)

    # 1. LABEL ENCODE CATEGORICAL VARIABLES TO MATCH TREE MODEL TRAINING
    # Use the label encoders loaded from training
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in processed.columns:
                try:
                    value = processed[col].iloc[0]  # Get the first (and only) value
                    print(f"Encoding {col}: {value}")  # DEBUG

                    # Handle unknown categories
                    if value in encoder.classes_:
                        encoded_value = encoder.transform([value])[0]
                    else:
                        # Use the first class as default for unknown values
                        encoded_value = 0
                        print(f"Unknown value '{value}' for {col}, using default: {encoded_value}")

                    processed[col] = encoded_value
                    print(f"Encoded {col} to: {encoded_value}")

                except Exception as e:
                    print(f"Error encoding column '{col}': {e}")
                    processed[col] = 0  # Default value

    # DEBUG: Print available columns before alignment
    print(f"Available columns after encoding: {processed.columns.tolist()}")
    print(f"Expected feature names: {feature_names}")

    # 2. ALIGN WITH FEATURE_NAMES *BEFORE* FEATURE ENGINEERING
    if feature_names is not None:
        # First, add any missing features from feature_names with value 0
        for feature in feature_names:
            if feature not in processed.columns:
                processed[feature] = 0
                print(f"Added missing feature: {feature}")

        # Now reorder to match EXACTLY the feature_names order
        missing_in_processed = [f for f in feature_names if f not in processed.columns]
        extra_in_processed = [f for f in processed.columns if f not in feature_names]

        if missing_in_processed:
            print(f"WARNING: Features still missing after adding: {missing_in_processed}")
        if extra_in_processed:
            print(f"WARNING: Extra features that will be dropped: {extra_in_processed}")

        # Select only the features in the exact order expected by the model
        processed = processed[feature_names]

    # 3. SKIP FEATURE ENGINEERING - Tree model was trained without engineered features
    print("Skipping feature engineering for tree-based model")

    # 4. SCALE NUMERICAL FEATURES IF SCALER IS AVAILABLE
    if scaler is not None:
        # Identify numerical features that exist in our processed data (tree model features only)
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        numerical_features = [col for col in numerical_features if col in processed.columns]

        if numerical_features:
            try:
                processed[numerical_features] = scaler.transform(processed[numerical_features])
            except Exception as e:
                print(f"Error scaling features: {e}")

    # DEBUG: Print final columns and first row values
    print(f"Final columns sent to model: {processed.columns.tolist()}")
    print(f"First row values: {processed.iloc[0].values}")

    return processed

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'features_loaded': feature_names is not None
    })

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'feature_count': len(feature_names) if feature_names else 0,
        'features_loaded': feature_names is not None,
        'scaler_loaded': scaler is not None,
        'model_classes': model.classes_.tolist() if hasattr(model, 'classes_') else []
    })

@app.route('/test')
def test_prediction():
    """Test endpoint with sample data"""
    sample_data = {
        'tenure': 12,
        'MonthlyCharges': 65.5,
        'TotalCharges': 786.0,
        'Contract': 'Month-to-month',
        'InternetService': 'Fiber optic',
        'PhoneService': 'Yes',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No'
    }
    
    try:
        response = predict_test(sample_data)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_test(data):
    """Test prediction function"""
    input_data = pd.DataFrame([data])
    processed_data = preprocess_input(input_data)
    
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)
    
    return {
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0][1]),
        'churn_risk': 'High' if prediction[0] == 1 else 'Low',
        'processed_features': processed_data.columns.tolist()[:10]  # First 10 features
    }

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({'error': config.API_MESSAGES['model_not_loaded']}), 500
    
    try:
        # Get file from request
        file = request.files['file']
        
        # Read the file
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        else:
            return jsonify({'error': config.API_MESSAGES['invalid_file_type']}), 400
        
        # Preprocess the data
        processed_data = preprocess_batch_input(data)
        
        # Make predictions
        predictions = model.predict(processed_data)
        predictions_proba = model.predict_proba(processed_data)
        
        # Add predictions to original data
        data['churn_prediction'] = predictions
        data['churn_probability'] = predictions_proba[:, 1]
        data['churn_risk'] = ['High' if p == 1 else 'Low' for p in predictions]
        
        # Return results as JSON
        return jsonify(data.to_dict(orient='records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def preprocess_batch_input(input_data):
    """Preprocess batch input data"""
    # Copy the input data
    processed = input_data.copy()
    
    # Handle TotalCharges (case-insensitive)
    total_charges_col = next((col for col in processed.columns if col.lower() == 'totalcharges'), None)
    if total_charges_col:
        processed[total_charges_col] = pd.to_numeric(processed[total_charges_col], errors='coerce')
        processed[total_charges_col].fillna(0, inplace=True)
    
    # 1. Encode categorical variables first
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in processed.columns:
                try:
                    processed[col] = processed[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
                    processed[col] = encoder.transform(processed[col])
                except Exception as e:
                    print(f"Error encoding {col}: {e}")
                    processed[col] = 0
    
    # 2. Align with feature_names
    if feature_names is not None:
        for feature in feature_names:
            if feature not in processed.columns:
                processed[feature] = 0
        processed = processed[feature_names]
    
    # 3. SKIP FEATURE ENGINEERING - Tree model was trained without engineered features
    print("Skipping feature engineering for tree-based model in batch processing")
    
    # 4. Scale numerical features
    if scaler is not None:
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        numerical_features = [col for col in numerical_features if col in processed.columns]

        if numerical_features:
            try:
                processed[numerical_features] = scaler.transform(processed[numerical_features])
            except Exception as e:
                print(f"Error scaling features: {e}")
    
    return processed

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Loading model...")
    if load_model():
        print("Model loaded successfully!")
    else:
        print("Model not loaded. Please run main.py first to train models.")
        print("You can still start the server, but predictions will fail until models are trained.")
    
    app.run(
        host=config.FLASK_HOST, 
        port=config.FLASK_PORT, 
        debug=config.FLASK_DEBUG
    )