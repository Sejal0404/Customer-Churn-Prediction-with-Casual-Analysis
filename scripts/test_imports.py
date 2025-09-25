#!/usr/bin/env python3
"""
Test script to check imports
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("Testing imports...")

try:
    from src.utils import setup_logging, load_config
    print("✓ utils imported successfully")
    
    from src.data_preprocessing import DataPreprocessor
    print("✓ data_preprocessing imported successfully")
    
    from src.model_training import ModelTrainer
    print("✓ model_training imported successfully")
    
    print("All core imports successful!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check the module files for syntax errors")