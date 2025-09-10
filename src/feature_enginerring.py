import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

def create_engineered_features(df):
    """
    Create engineered features for the Telco Churn dataset
    """
    df_eng = df.copy()
    
    # 1. Convert TotalCharges to numeric and handle missing values
    df_eng['TotalCharges'] = pd.to_numeric(df_eng['TotalCharges'], errors='coerce')
    df_eng['TotalCharges'].fillna(0, inplace=True)
    
    # 2. Convert target variable
    if 'Churn' in df_eng.columns:
        if df_eng['Churn'].dtype == 'object':
            df_eng['Churn'] = df_eng['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 3. Basic tenure features
    df_eng['tenure_group'] = create_tenure_groups(df_eng['tenure'])
    df_eng['tenure_months'] = df_eng['tenure'] % 12
    df_eng['tenure_years'] = df_eng['tenure'] // 12
    
    # 4. Charge-related features
    df_eng = create_charge_features(df_eng)
    
    # 5. Service usage patterns
    df_eng = create_service_features(df_eng)
    
    # 6. Customer demographic features
    df_eng = create_demographic_features(df_eng)
    
    # 7. Interaction features
    df_eng = create_interaction_features(df_eng)
    
    # 8. Behavioral features
    df_eng = create_behavioral_features(df_eng)
    
    return df_eng

def create_tenure_groups(tenure_series):
    """Create tenure groups"""
    return pd.cut(tenure_series, 
                 bins=[0, 12, 24, 36, 48, 60, 72, np.inf],
                 labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr', '6+yr'])

def create_charge_features(df):
    """Create features related to charges and payments"""
    df_eng = df.copy()
    epsilon = 1e-10  # Avoid division by zero
    
    # Monthly charge per tenure
    df_eng['avg_monthly_revenue'] = np.where(
        df_eng['tenure'] > 0,
        df_eng['TotalCharges'] / df_eng['tenure'],
        0
    )

    # Ratio of monthly to total charges
    df_eng['charge_ratio'] = np.where(
        df_eng['TotalCharges'] > 0,
        df_eng['MonthlyCharges'] / (df_eng['TotalCharges'] + epsilon),
        0
    )
    
    return df_eng

def create_service_features(df):
    """Create features related to service usage"""
    df_eng = df.copy()
    
    # Count of services used
    service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    available_services = [col for col in service_columns if col in df_eng.columns]
    
    df_eng['total_services'] = df_eng[available_services].apply(
        lambda x: sum(1 for val in x if val in ['Yes', 'DSL', 'Fiber optic']), axis=1
    )
    
    return df_eng

def create_demographic_features(df):
    """Create demographic-related features"""
    df_eng = df.copy()
    
    # Family size
    df_eng['family_size'] = 1
    if 'Partner' in df_eng.columns:
        df_eng['family_size'] += (df_eng['Partner'] == 'Yes').astype(int)
    if 'Dependents' in df_eng.columns:
        df_eng['family_size'] += (df_eng['Dependents'] == 'Yes').astype(int)
    
    return df_eng

def create_interaction_features(df):
    """Create interaction features"""
    df_eng = df.copy()
    
    # Contract tenure interaction
    contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    if 'Contract' in df_eng.columns:
        df_eng['contract_tenure_ratio'] = df_eng['tenure'] / df_eng['Contract'].map(
            lambda x: contract_mapping.get(x, 1)
        )
    
    return df_eng

def create_behavioral_features(df):
    """Create behavioral features"""
    df_eng = df.copy()
    
    # Loyalty score
    contract_scores = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
    if 'Contract' in df_eng.columns:
        df_eng['loyalty_score'] = df_eng['tenure'] * df_eng['Contract'].map(
            lambda x: contract_scores.get(x, 1)
        )
    
    return df_eng

def prepare_linear_features(df_eng):
    """Prepare features for linear models"""
    categorical_cols = df_eng.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['customerID', 'Churn']]
    
    df_linear = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True)
    return df_linear

def prepare_tree_features(df_eng):
    """Prepare features for tree-based models"""
    df_tree = df_eng.copy()
    categorical_cols = df_tree.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['customerID', 'Churn']]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_tree[col] = le.fit_transform(df_tree[col].astype(str))
        label_encoders[col] = le
    
    return df_tree, label_encoders

def load_processed_data(model_type):
    """Load processed data for the specified model type"""
    if model_type == 'linear':
        df = pd.read_csv('data/processed/telco_churn_linear_models.csv')
        import joblib
        feature_names = joblib.load('data/processed/feature_names_linear.pkl')
    elif model_type == 'tree':
        df = pd.read_csv('data/processed/telco_churn_tree_models.csv')
        import joblib
        feature_names = joblib.load('data/processed/feature_names_tree.pkl')
    else:
        raise ValueError("model_type must be 'linear' or 'tree'")
    
    return df, feature_names

def prepare_training_data(df, feature_names, balance_method='smote', test_size=0.2, random_state=0):
    """Prepare training and test data with optional balancing"""
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    
    # Split features and target
    X = df[feature_names]
    y = df['Churn']
    print(f"Full y unique: {y.unique()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    numerical_features = [col for col in numerical_features if col in X.columns]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if numerical_features:
        X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Apply SMOTE for balancing
    if balance_method == 'smote':
        print(f"y_train unique: {y_train.unique()}")
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler

def prepare_linear_model_data(df_eng, output_dir):
    """Prepare features for linear models and save to file"""
    df_linear = prepare_linear_features(df_eng)
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df_linear.to_csv(output_path / 'telco_churn_linear_models.csv', index=False)
    # Save feature names
    feature_names = df_linear.drop(['Churn', 'customerID'], axis=1, errors='ignore').columns.tolist()
    import joblib
    joblib.dump(feature_names, output_path / 'feature_names_linear.pkl')
    return df_linear

def prepare_tree_model_data(df_eng, output_dir):
    """Prepare features for tree models and save to file"""
    df_tree, label_encoders = prepare_tree_features(df_eng)
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df_tree.to_csv(output_path / 'telco_churn_tree_models.csv', index=False)
    # Save feature names
    feature_names = df_tree.drop(['Churn', 'customerID'], axis=1, errors='ignore').columns.tolist()
    import joblib
    joblib.dump(feature_names, output_path / 'feature_names_tree.pkl')
    # Save label encoders
    joblib.dump(label_encoders, output_path / 'label_encoders.pkl')
    return df_tree, label_encoders

def load_data(file_path):
    """Load data from file"""
    return pd.read_csv(file_path)

# Example usage
if __name__ == "__main__":
    # Load data
    df = load_data('../data/raw/Customer_Churn_Prediction_Dataset.csv')

    # Create engineered features
    df_eng = create_engineered_features(df)

    # Prepare datasets
    df_linear = prepare_linear_features(df_eng)
    df_tree, label_encoders = prepare_tree_features(df_eng)

    # Save datasets
    output_dir = Path('../data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    df_linear.to_csv(output_dir / 'telco_churn_linear_models.csv', index=False)
    df_tree.to_csv(output_dir / 'telco_churn_tree_models.csv', index=False)

    print("Feature engineering completed!")
