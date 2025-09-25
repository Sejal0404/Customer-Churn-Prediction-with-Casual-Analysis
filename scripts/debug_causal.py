#!/usr/bin/env python3
"""
Debug script to understand data issues
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_dataframe

def main():
    # Load data
    data_path = os.path.join(project_root, "data", "interim", "preprocessed_data.csv")
    df = load_dataframe(data_path)
    
    print("DATA DEBUG INFORMATION")
    print("="*60)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check Churn column
    if 'Churn' in df.columns:
        print(f"\nChurn column info:")
        print(f"  Data type: {df['Churn'].dtype}")
        print(f"  Unique values: {df['Churn'].unique()}")
        print(f"  Value counts:")
        print(df['Churn'].value_counts())
    
    # Check treatment variables
    treatment_vars = ['Contract', 'PaymentMethod', 'InternetService']
    
    for treatment in treatment_vars:
        if treatment in df.columns:
            print(f"\n{treatment} column info:")
            print(f"  Data type: {df[treatment].dtype}")
            print(f"  Unique values: {df[treatment].unique()}")
            print(f"  Non-null count: {df[treatment].count()}/{len(df)}")
            print(f"  Sample values: {df[treatment].head(5).tolist()}")
            
            # Check if we can create a valid contingency table
            if 'Churn' in df.columns:
                contingency = pd.crosstab(df[treatment], df['Churn'])
                print(f"  Contingency table shape: {contingency.shape}")
                print(f"  Contingency table:")
                print(contingency)
    
    # Test basic analysis
    print("\nBASIC ANALYSIS TEST")
    print("="*60)
    
    if 'Churn' in df.columns:
        # Convert Churn to numeric
        df['Churn_numeric'] = df['Churn'].map({'No': 0, 'Yes': 1}) if df['Churn'].dtype == 'object' else df['Churn']
        
        for treatment in treatment_vars:
            if treatment in df.columns:
                try:
                    # Simple groupby analysis
                    result = df.groupby(treatment)['Churn_numeric'].mean()
                    print(f"\n{treatment} churn rates:")
                    for category, rate in result.items():
                        print(f"  {category}: {rate:.3f}")
                except Exception as e:
                    print(f"Error analyzing {treatment}: {e}")

if __name__ == "__main__":
    main()