import yaml
import logging
import logging.config
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import joblib
import os

def setup_logging(logging_config_path: str = None, 
                  default_level: int = logging.INFO) -> None:
    """Setup logging configuration"""
    if logging_config_path is None:
        logging_config_path = os.path.join(os.path.dirname(__file__), "..", "config", "logging.yaml")
    
    if os.path.exists(logging_config_path):
        try:
            with open(logging_config_path, 'r') as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
        except Exception as e:
            logging.basicConfig(level=default_level)
            logging.warning(f"Error loading logging config: {e}. Using basic config.")
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config file not found at {logging_config_path}. Using basic config.")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise

def save_model(model, filepath: str) -> None:
    """Save model to file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model to {filepath}: {e}")
        raise

def load_model(filepath: str):
    """Load model from file"""
    try:
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            logging.info(f"Model loaded from {filepath}")
            return model
        else:
            logging.error(f"Model file not found at {filepath}")
            return None
    except Exception as e:
        logging.error(f"Error loading model from {filepath}: {e}")
        return None

def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """Save DataFrame to CSV"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=index)
        logging.info(f"DataFrame saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {filepath}: {e}")
        raise

def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load DataFrame from CSV"""
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            logging.info(f"DataFrame loaded from {filepath}")
            return df
        else:
            logging.error(f"Data file not found at {filepath}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading DataFrame from {filepath}: {e}")
        return pd.DataFrame()

def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))