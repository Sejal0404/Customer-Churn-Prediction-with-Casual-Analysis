#!/usr/bin/env python3
"""
Simple analysis that will definitely work - FIXED VERSION
"""

import pandas as pd
import os

# Load data
data_path = "data/interim/preprocessed_data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("SIMPLE CAUSAL ANALYSIS RESULTS")
    print("="*70)
    
    # Basic info
    print(f"Dataset: {len(df):,} customers")
    
    # Check Churn column type and calculate overall churn rate
    if 'Churn' in df.columns:
        if df['Churn'].dtype == 'object':
            overall_churn = (df['Churn'] == 'Yes').mean()
            churn_numeric = (df['Churn'] == 'Yes').astype(int)
        else:
            overall_churn = df['Churn'].mean()
            churn_numeric = df['Churn']
        
        print(f"Overall churn rate: {overall_churn:.1%}")
    else:
        print("Churn column not found!")
        exit()
    
    # Analyze key factors
    factors = ['Contract', 'PaymentMethod', 'InternetService', 'tenure']
    
    for factor in factors:
        if factor in df.columns:
            print(f"\n{factor.upper()} ANALYSIS:")
            print("-" * 40)
            
            if df[factor].dtype == 'object':
                # Categorical factor - group by the factor and calculate churn rates
                grouped = df.groupby(factor)
                
                for category, group_data in grouped:
                    churn_rate = churn_numeric[group_data.index].mean()
                    count = len(group_data)
                    effect = churn_rate - overall_churn
                    
                    print(f"  {category}:")
                    print(f"    Churn rate: {churn_rate:.1%}")
                    print(f"    Effect vs overall: {effect:+.1%}")
                    print(f"    Customers: {count:,}")
                    
                    # Add interpretation
                    if effect > 0.05:
                        print("    ⚠️  HIGH RISK - Significantly above average")
                    elif effect < -0.05:
                        print("    ✅ LOW RISK - Significantly below average")
                    print()
                    
            else:
                # Numerical factor
                correlation = df[factor].corr(churn_numeric)
                print(f"  Correlation with churn: {correlation:.3f}")
                print(f"  Average {factor}: {df[factor].mean():.1f}")
                print(f"  Range: {df[factor].min():.1f} to {df[factor].max():.1f}")
                
                # Interpretation
                if abs(correlation) > 0.3:
                    strength = "strong"
                elif abs(correlation) > 0.1:
                    strength = "moderate"
                else:
                    strength = "weak"
                    
                direction = "increases" if correlation > 0 else "decreases"
                print(f"  📊 {strength} {direction} churn likelihood")
    
    print("\n" + "="*70)
    print("KEY BUSINESS INSIGHTS:")
    print("="*70)
    
    # Generate automatic insights
    insights = []
    
    # Contract insights
    if 'Contract' in df.columns and df['Contract'].dtype == 'object':
        contract_churn = df.groupby('Contract')[churn_numeric.name].mean()
        worst_contract = contract_churn.idxmax()
        best_contract = contract_churn.idxmin()
        insights.append(f"• {worst_contract} contracts have the highest churn")
        insights.append(f"• {best_contract} contracts have the lowest churn")
    
    # Payment method insights
    if 'PaymentMethod' in df.columns and df['PaymentMethod'].dtype == 'object':
        payment_churn = df.groupby('PaymentMethod')[churn_numeric.name].mean()
        worst_payment = payment_churn.idxmax()
        insights.append(f"• {worst_payment} users are most likely to churn")
    
    # Tenure insights
    if 'tenure' in df.columns:
        # Create tenure groups for insight
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 60, 100], 
                                   labels=['0-1yr', '1-2yr', '2-5yr', '5+yr'])
        tenure_churn = df.groupby('tenure_group')[churn_numeric.name].mean()
        highest_tenure_group = tenure_churn.idxmax()
        insights.append(f"• {highest_tenure_group} customers have highest churn risk")
    
    # Print insights
    for insight in insights:
        print(insight)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. Focus retention efforts on high-risk customer segments")
    print("2. Promote contracts with lower churn rates")
    print("3. Review payment methods associated with high churn")
    print("4. Implement early intervention for new customers")
    print("5. Use the trained model to predict at-risk customers")
    
else:
    print("Data file not found at:", data_path)
    print("Please run the training pipeline first: python scripts/train_pipeline.py")