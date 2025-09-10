import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def propensity_score_matching(df, treatment_col, outcome_col, confounders):
    """
    Complete implementation of propensity score matching
    """
    # Prepare data
    X = df[confounders]
    y = df[treatment_col]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Estimate propensity scores
    ps_model = LogisticRegression(random_state=42, max_iter=5000, solver='lbfgs')
    ps_model.fit(X_scaled, y)
    propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
    
    df['propensity_score'] = propensity_scores
    
    # Perform matching
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    # Calculate distances
    distances = pairwise_distances(
        treated[['propensity_score']].values,
        control[['propensity_score']].values
    )
    
    # Find matches
    matches = {}
    for i in range(len(treated)):
        match_idx = np.argmin(distances[i])
        matches[i] = match_idx
    
    # Create matched dataset
    matched_treated = treated.iloc[list(matches.keys())]
    matched_control = control.iloc[list(matches.values())]
    matched_df = pd.concat([matched_treated, matched_control])
    
    # Calculate treatment effect
    att = matched_treated[outcome_col].mean() - matched_control[outcome_col].mean()
    
    return att, matched_df, propensity_scores

def estimate_intervention_effect(df, intervention_col='discount_offered', outcome_col='Churn'):
    """Estimate intervention effect"""
    confounders = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    # Check if intervention column exists
    if intervention_col not in df.columns:
        print(f"Warning: {intervention_col} not found in dataframe")
        return None
    
    att, matched_df, propensity_scores = propensity_score_matching(
        df, intervention_col, outcome_col, confounders
    )
    
    return {
        'treatment_effect': att,
        'matched_data': matched_df,
        'propensity_scores': propensity_scores
    }