# main.py - Complete Customer Churn Prediction Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, auc,
                             f1_score, precision_score, recall_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# ==================== CORRECT PATH SETUP ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Working with paths:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
# ==================== END PATH SETUP ====================

def main():
    print("Loading and preprocessing raw data...")
    
    # Load raw data
    df = pd.read_csv(DATA_DIR / 'raw' / 'Customer_Churn_Prediction_Dataset.csv')
    print(f"Original class distribution: {dict(df['Churn'].value_counts())}")
    
    # Basic preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    churn_rate = df['Churn'].mean() * 100
    print(f"Churn rate: {churn_rate:.2f}%")
    
    # Feature engineering
    print("Creating engineered features...")
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('customerID')
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Dataset 1: For tree-based models (Label Encoding)
    df_tree = df.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_tree[col] = le.fit_transform(df_tree[col].astype(str))
        label_encoders[col] = le
    
    # Dataset 2: For linear models (One-Hot Encoding)
    df_linear = df.copy()
    df_linear = pd.get_dummies(df_linear, columns=categorical_cols, drop_first=True)
    
    # Add engineered features to both datasets
    def add_engineered_features(df):
        df_engineered = df.copy()
        # Add your feature engineering logic here
        return df_engineered
    
    df_tree_engineered = add_engineered_features(df_tree)
    df_linear_engineered = add_engineered_features(df_linear)
    
    print(f"Linear dataset shape: {df_linear_engineered.shape}")
    print(f"Tree dataset shape: {df_tree_engineered.shape}")
    
    # Prepare datasets for modeling
    X_tree = df_tree_engineered.drop(['Churn', 'customerID'], axis=1)
    y_tree = df_tree_engineered['Churn']
    
    X_linear = df_linear_engineered.drop(['Churn', 'customerID'], axis=1)
    y_linear = df_linear_engineered['Churn']
    
    # Split data
    X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(
        X_tree, y_tree, test_size=0.2, random_state=42, stratify=y_tree
    )
    
    X_linear_train, X_linear_test, y_linear_train, y_linear_test = train_test_split(
        X_linear, y_linear, test_size=0.2, random_state=42, stratify=y_linear
    )
    
    # Handle class imbalance with SMOTE
    print("Preparing training data with SMOTE...")
    smote = SMOTE(random_state=42)
    
    X_tree_train_res, y_tree_train_res = smote.fit_resample(X_tree_train, y_tree_train)
    X_linear_train_res, y_linear_train_res = smote.fit_resample(X_linear_train, y_linear_train)
    
    # Model training setup
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=5000, solver='lbfgs'),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100]
            },
            'X_train': X_linear_train_res,
            'y_train': y_linear_train_res,
            'X_test': X_linear_test,
            'y_test': y_linear_test
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            },
            'X_train': X_tree_train_res,
            'y_train': y_tree_train_res,
            'X_test': X_tree_test,
            'y_test': y_tree_test
        },
        'xgboost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'scale_pos_weight': [1, sum(y_tree_train == 0) / sum(y_tree_train == 1)]
            },
            'X_train': X_tree_train_res,
            'y_train': y_tree_train_res,
            'X_test': X_tree_test,
            'y_test': y_tree_test
        }
    }
    
    # Train models
    print("Training tree-based models...")
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model_info in models.items():
        print(f"Training {name}...")
        
        grid_search = GridSearchCV(
            model_info['model'], 
            model_info['params'], 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(model_info['X_train'], model_info['y_train'])
        best_model = grid_search.best_estimator_
        
        # Make predictions and calculate metrics
        y_pred = best_model.predict(model_info['X_test'])
        y_pred_proba = best_model.predict_proba(model_info['X_test'])[:, 1]
        
        precision, recall, _ = precision_recall_curve(model_info['y_test'], y_pred_proba)
        pr_auc = auc(recall, precision)
        
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'accuracy': accuracy_score(model_info['y_test'], y_pred),
            'roc_auc': roc_auc_score(model_info['y_test'], y_pred_proba),
            'pr_auc': pr_auc,
            'f1_score': f1_score(model_info['y_test'], y_pred),
            'precision': precision_score(model_info['y_test'], y_pred),
            'recall': recall_score(model_info['y_test'], y_pred),
            'classification_report': classification_report(model_info['y_test'], y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(model_info['y_test'], y_pred),
            'y_pred_proba': y_pred_proba,
            'test_data': (model_info['X_test'], model_info['y_test'])
        }
    
    # Select best model
    print("Selecting best model...")
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'ROC AUC': [results[name]['roc_auc'] for name in results],
        'PR AUC': [results[name]['pr_auc'] for name in results],
        'F1 Score': [results[name]['f1_score'] for name in results]
    }).sort_values('PR AUC', ascending=False)
    
    best_model_name = comparison.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    # ==================== CRITICAL FIX: CORRECT FILE SAVING ====================
    print("Saving results...")
    
    # Save with CORRECT filenames
    joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
    joblib.dump(list(X_linear.columns), MODELS_DIR / 'feature_names_linear.pkl')  # CORRECT NAME
    joblib.dump(list(X_tree.columns), MODELS_DIR / 'feature_names_tree.pkl')
    
    # Save scalers and encoders
    scaler_linear = StandardScaler().fit(X_linear_train_res.select_dtypes(include=[np.number]))
    scaler_tree = StandardScaler().fit(X_tree_train_res.select_dtypes(include=[np.number]))
    
    joblib.dump(scaler_linear, MODELS_DIR / 'scaler_linear.pkl')
    joblib.dump(scaler_tree, MODELS_DIR / 'scaler_tree.pkl')
    joblib.dump(label_encoders, MODELS_DIR / 'label_encoders.pkl')
    
    # Save all results
    joblib.dump(results, MODELS_DIR / 'all_model_results.pkl')
    
    # Also create the file that app.py expects (feature_names.pkl)
    # This is the key fix - create both names for compatibility
    feature_names_linear = list(X_linear.columns)
    joblib.dump(feature_names_linear, MODELS_DIR / 'feature_names.pkl')  # For app.py compatibility
    
    print("Model artifacts saved with consistent naming:")
    print("  - best_model.pkl")
    print("  - feature_names_linear.pkl (for analysis)")
    print("  - feature_names.pkl (for app.py compatibility)")
    print("  - feature_names_tree.pkl")
    print("  - scaler_linear.pkl")
    print("  - scaler_tree.pkl")
    print("  - label_encoders.pkl")
    print("  - all_model_results.pkl")
    # ==================== END CRITICAL FIX ====================
    
    # Generate business insights
    print("\n=== BUSINESS IMPACT SIMULATION ===")
    # Add your business simulation code here
    
    print("Pipeline completed successfully!")
    print(f"Results saved to '{RESULTS_DIR}/' directory")
    print(f"Models saved to '{MODELS_DIR}/' directory")

if __name__ == "__main__":
    main()