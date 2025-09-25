import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Tuple, Dict, Any, List  # Added List import

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing and cleaning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        try:
            df = pd.read_csv(self.config['data']['raw_file'])
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        df_clean = df.copy()
        
        # Handle TotalCharges - convert to numeric, coerce errors to NaN
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Drop customerID as it's not useful for modeling
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
        
        # Handle missing values
        missing_values = df_clean.isnull().sum()
        if missing_values.any():
            logger.info(f"Missing values found: {missing_values[missing_values > 0]}")
            # For TotalCharges, fill with 0 for new customers (tenure=0)
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
        
        # Convert SeniorCitizen to categorical
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(str)
        
        # Convert target variable to binary
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def create_preprocessor(self, numeric_features: List[str], 
                          categorical_features: List[str]) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, 
                                                     ColumnTransformer]:
        """Prepare data for modeling"""
        # Separate features and target
        X = df.drop(self.config['data']['target_column'], axis=1)
        y = df[self.config['data']['target_column']]
        
        # Get feature lists
        numeric_features = self.config['data']['numeric_features']
        categorical_features = self.config['data']['categorical_features']
        
        # Create and fit preprocessor
        self.preprocessor = self.create_preprocessor(numeric_features, categorical_features)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names = []
        if 'num' in self.preprocessor.named_transformers_:
            feature_names.extend(numeric_features)
        
        if 'cat' in self.preprocessor.named_transformers_:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = cat_encoder.get_feature_names_out(categorical_features)
            feature_names.extend(cat_features)
        
        self.feature_names = feature_names
        
        # Convert back to DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        logger.info(f"Data preparation completed. Features: {len(feature_names)}")
        return X_processed_df, y, self.preprocessor
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test