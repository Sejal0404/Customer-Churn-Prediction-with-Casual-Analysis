# config.py
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data files
RAW_DATA_FILE = DATA_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

TREE_MODELS_FILE = PROCESSED_DATA_DIR / 'telco_churn_tree_models.csv'
LINEAR_MODELS_FILE = PROCESSED_DATA_DIR / 'telco_churn_linear_models.csv'

# Model files - UPDATED TO MATCH YOUR check_model_files() SCRIPT
MODEL_FILES = {
    'logistic_regression': 'logistic_regression.pkl',
    'random_forest': 'random_forest.pkl',
    'xgboost': 'xgboost.pkl',
    'feature_names_linear': 'feature_names_linear.pkl',
    'feature_names_tree': 'feature_names_tree.pkl',
    'scaler_linear': 'scaler_linear.pkl', 
    'scaler_tree': 'scaler_tree.pkl',
    'label_encoders': 'label_encoders.pkl',
    'best_model': 'logistic_regression.pkl'  # Default to logistic regression as best model
}

# Flask configuration
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = True

# API messages
API_MESSAGES = {
    'model_not_loaded': 'Model not loaded. Please try again later.',
    'invalid_file_type': 'Invalid file type. Please upload a CSV file.',
    'prediction_success': 'Prediction completed successfully.',
    'training_started': 'Model training started successfully.'
}

# Feature configuration
TARGET_COLUMN = 'Churn'
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Model training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# SMOTE configuration
SMOTE_RANDOM_STATE = 42

# Hyperparameter grids
LOGISTIC_REGRESSION_PARAMS = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2']
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

XGBOOST_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [1]  # Will be calculated dynamically
}

# Business simulation parameters
AVERAGE_MONTHLY_REVENUE = 64.76
RETENTION_COST_PER_CUSTOMER = 50  # Estimated cost to retain a customer
RETENTION_SUCCESS_RATE = 0.6  # 60% success rate for retention efforts

# Ensure the script runs correctly
if __name__ == '__main__':
    print("Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print("Available model files:")
    for key, filename in MODEL_FILES.items():
        print(f"  {key}: {filename}")