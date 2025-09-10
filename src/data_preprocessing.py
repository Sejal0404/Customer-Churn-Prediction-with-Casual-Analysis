import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

def load_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    return df

def prepare_datasets(input_path, output_dir):
    """Main preprocessing function"""
    df = load_data(input_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_path / 'telco_churn_cleaned.csv', index=False)
    
    return df

def preprocess_for_training(df, model_type='tree', test_size=0.2):
    """Prepare data for training"""
    X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test