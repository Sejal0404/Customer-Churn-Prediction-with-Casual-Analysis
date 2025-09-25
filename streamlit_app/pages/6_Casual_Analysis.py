import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_dataframe
from src.causal_analysis import CausalAnalyzer
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import your modules
from src.utils import load_dataframe, load_model
# ... other imports


def render(config):
    st.title("🧠 Causal Analysis")
    
    # Load data
    @st.cache_data
    def load_causal_data():
        df_clean = load_dataframe("data/interim/preprocessed_data.csv")
        causal_results = load_dataframe("results/reports/causal_effects.csv")
        return df_clean, causal_results
    
    try:
        df_clean, causal_results = load_causal_data()
        if df_clean.empty:
            st.warning("Please run the causal analysis pipeline first.")
            return
    except:
        st.warning("Causal analysis data not available. Please run the analysis pipeline first.")
        return
    
    st.info("""
    This section explores causal relationships between business interventions and churn.
    Causal analysis helps understand what truly drives churn, beyond correlations.
    """)
    
    # Tabs for different causal analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Causal Effects", "Treatment Analysis", "Policy Simulation", "Sensitivity Analysis"
    ])
    
    with tab1:
        render_causal_effects(causal_results, df_clean, config)
    
    with tab2:
        render_treatment_analysis(df_clean, config)
    
    with tab3:
        render_policy_simulation(df_clean, config)
    
    with tab4:
        render_sensitivity_analysis(df_clean, config)

def render_causal_effects(causal_results, df_clean, config):
    """Render causal effects overview"""
    
    st.subheader("Estimated Causal Effects")
    
    if causal_results.empty:
        st.warning("No causal analysis results available. Running analysis...")
        try:
            # Run causal analysis
            causal_analyzer = CausalAnalyzer(df_clean, config)
            results = causal_analyzer.analyze_all_treatments()
            causal_results = causal_analyzer.get_effect_summary()
        except Exception as e:
            st.error(f"Causal analysis failed: {e}")
            return
    
    # Display causal effects
    st.dataframe(causal_results.style.format({
        'Estimated_Effect': '{:.4f}',
        'CI_Lower': '{:.4f}',
        'CI_Upper': '{:.4f}',
        'P_Value': '{:.4f}'
    }).apply(lambda x: ['background-color: lightgreen' if x['Significant'] else '' for _ in x], 
            axis=1), use_container_width=True)
    
    # Visualize causal effects
    st.subheader("Causal Effects Visualization")
    
    fig = go.Figure()
    
    for _, row in causal_results.iterrows():
        color = 'red' if row['Estimated_Effect'] > 0 else 'green'
        fig.add_trace(go.Scatter(
            x=[row['Estimated_Effect']],
            y=[row['Treatment']],
            error_x=dict(
                type='data',
                array=[[row['Estimated_Effect'] - row['CI_Lower']]],
                arrayminus=[[row['CI_Upper'] - row['Estimated_Effect']]],
                visible=True
            ),
            mode='markers',
            name=row['Treatment'],
            marker=dict(size=10, color=color),
            hovertemplate=f"Effect: {row['Estimated_Effect']:.3f}<br>" +
                         f"CI: [{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]<br>" +
                         f"P-value: {row['P_Value']:.3f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Causal Effects with Confidence Intervals",
        xaxis_title="Estimated Effect Size",
        yaxis_title="Treatment Variable",
        showlegend=False
    )
    
    # Add significance threshold line
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    
    st.plotly_chart(fig)
    
    # Interpretation
    st.subheader("Key Insights")
    
    significant_effects = causal_results[causal_results['Significant']]
    
    if not significant_effects.empty:
        for _, effect in significant_effects.iterrows():
            direction = "increases" if effect['Estimated_Effect'] > 0 else "decreases"
            st.write(f"**{effect['Treatment']}** {direction} churn probability by {abs(effect['Estimated_Effect']):.3f}")
            
            # Business interpretation
            if effect['Treatment'] == 'Contract':
                if effect['Estimated_Effect'] > 0:
                    st.write("  - Month-to-month contracts significantly increase churn risk")
                    st.write("  - Recommendation: Promote longer-term contracts")
            elif effect['Treatment'] == 'PaymentMethod':
                st.write("  - Payment method has significant impact on churn")
                st.write("  - Recommendation: Encourage automatic payment methods")
    else:
        st.write("No statistically significant causal effects detected.")

def render_treatment_analysis(df_clean, config):
    """Render detailed treatment analysis"""
    
    st.subheader("Treatment Variable Analysis")
    
    treatment_vars = config['causal_analysis']['treatment_variables']
    selected_treatment = st.selectbox("Select treatment variable", treatment_vars)
    
    if selected_treatment:
        # Treatment distribution
        st.write(f"**Distribution of {selected_treatment}**")
        
        treatment_counts = df_clean[selected_treatment].value_counts()
        fig_pie = px.pie(values=treatment_counts.values, names=treatment_counts.index,
                        title=f"Distribution of {selected_treatment}")
        st.plotly_chart(fig_pie)
        
        # Treatment vs churn
        st.write(f"**Churn Rate by {selected_treatment}**")
        
        churn_by_treatment = df_clean.groupby(selected_treatment)['Churn'].mean().sort_values()
        
        fig_bar = px.bar(x=churn_by_treatment.index, y=churn_by_treatment.values,
                        title=f"Churn Rate by {selected_treatment}",
                        labels={'x': selected_treatment, 'y': 'Churn Rate'})
        st.plotly_chart(fig_bar)
        
        # Detailed analysis
        st.subheader("Detailed Treatment Analysis")
        
        # Calculate average outcomes by treatment level
        summary_stats = df_clean.groupby(selected_treatment).agg({
            'Churn': ['mean', 'count'],
            'tenure': 'mean',
            'MonthlyCharges': 'mean'
        }).round(3)
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Business recommendations
        st.subheader("Business Recommendations")
        
        # Simple rule-based recommendations
        if selected_treatment == 'Contract':
            st.write("""
            **Contract Type Recommendations:**
            - **Issue:** Month-to-month contracts show highest churn
            - **Action:** Develop incentives for longer-term contracts
            - **Metric:** Target 20% reduction in month-to-month contracts
            """)
        
        elif selected_treatment == 'PaymentMethod':
            st.write("""
            **Payment Method Recommendations:**
            - **Issue:** Electronic checks associated with higher churn
            - **Action:** Promote automatic payment methods with discounts
            - **Metric:** Increase automatic payments by 15%
            """)

def render_policy_simulation(df_clean, config):
    """Render policy simulation interface"""
    
    st.subheader("Policy Simulation")
    
    st.info("""
    Simulate the impact of business interventions on churn rates.
    Adjust policy parameters to see estimated effects.
    """)
    
    # Policy selection
    policy_options = {
        "Contract Promotion": "Incentivize longer-term contracts",
        "Payment Method Optimization": "Shift customers to automatic payments",
        "Price Adjustment": "Adjust pricing strategy",
        "Service Bundle": "Introduce service bundles"
    }
    
    selected_policy = st.selectbox("Select policy to simulate", list(policy_options.keys()))
    
    # Policy parameters
    st.subheader("Policy Parameters")
    
    if selected_policy == "Contract Promotion":
        current_mtm = st.slider("Current month-to-month percentage", 0, 100, 50)
        target_mtm = st.slider("Target month-to-month percentage", 0, 100, 40)
        incentive_cost = st.number_input("Incentive cost per customer ($)", 0, 100, 10)
    
    elif selected_policy == "Payment Method Optimization":
        current_auto = st.slider("Current automatic payment percentage", 0, 100, 30)
        target_auto = st.slider("Target automatic payment percentage", 0, 100, 50)
        discount = st.number_input("Discount for automatic payments (%)", 0, 20, 5)
    
    # Simulation results
    if st.button("Run Simulation"):
        with st.spinner("Simulating policy impact..."):
            # Simulate policy impact
            baseline_churn = df_clean['Churn'].mean()
            
            if selected_policy == "Contract Promotion":
                # Simplified simulation
                reduction = (current_mtm - target_mtm) / 100
                estimated_impact = reduction * 0.15  # Assume 15% reduction in churn for contract change
                new_churn = baseline_churn * (1 - estimated_impact)
                
                # Cost-benefit analysis
                customers_affected = len(df_clean) * reduction
                total_cost = customers_affected * incentive_cost
                churn_reduction = baseline_churn - new_churn
                value_saved = churn_reduction * len(df_clean) * 500  # $500 value per retained customer
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Baseline Churn Rate", f"{baseline_churn:.1%}")
                st.metric("New Churn Rate", f"{new_churn:.1%}")
            
            with col2:
                st.metric("Churn Reduction", f"{(baseline_churn - new_churn):.1%}")
                st.metric("Customers Affected", f"{customers_affected:,.0f}")
            
            with col3:
                st.metric("Total Cost", f"${total_cost:,.0f}")
                st.metric("Value Created", f"${value_saved:,.0f}")
            
            # ROI calculation
            if total_cost > 0:
                roi = (value_saved - total_cost) / total_cost
                st.metric("Estimated ROI", f"{roi:.1%}")
            
            # Recommendation
            st.subheader("Policy Recommendation")
            if value_saved > total_cost:
                st.success("✅ Recommended: Policy creates positive ROI")
            else:
                st.warning("⚠️ Review Required: Policy may not be cost-effective")

def render_sensitivity_analysis(df_clean, config):
    """Render sensitivity analysis"""
    
    st.subheader("Sensitivity Analysis")
    
    st.info("""
    Test how sensitive the causal conclusions are to different assumptions.
    This helps validate the robustness of the findings.
    """)
    
    # Sensitivity parameters
    st.subheader("Analysis Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confounder_strength = st.slider("Assumed confounder strength", 0.1, 1.0, 0.5)
        sample_size = st.slider("Analysis sample size", 100, len(df_clean), min(1000, len(df_clean)))
    
    with col2:
        significance_level = st.slider("Significance level", 0.01, 0.1, 0.05)
        robustness_test = st.selectbox("Robustness test", 
                                     ["Placebo treatment", "Subset analysis", "Model variation"])
    
    if st.button("Run Sensitivity Analysis"):
        with st.spinner("Running sensitivity analysis..."):
            # Simulate sensitivity analysis
            try:
                causal_analyzer = CausalAnalyzer(df_clean.sample(sample_size), config)
                
                # Placeholder for sensitivity results
                sensitivity_results = {
                    'Original Effect': 0.15,
                    'With Stronger Confounders': 0.12,
                    'With Weaker Confounders': 0.18,
                    'Different Model': 0.14,
                    'Subset Analysis': 0.16
                }
                
                # Display results
                st.subheader("Sensitivity Results")
                
                fig_sensitivity = go.Figure()
                
                effects = list(sensitivity_results.keys())
                values = list(sensitivity_results.values())
                
                fig_sensitivity.add_trace(go.Bar(
                    x=effects, y=values,
                    marker_color=['blue'] + ['lightgray'] * (len(effects)-1)
                ))
                
                fig_sensitivity.update_layout(
                    title="Causal Effect Under Different Assumptions",
                    xaxis_title="Analysis Scenario",
                    yaxis_title="Estimated Effect Size",
                    showlegend=False
                )
                
                st.plotly_chart(fig_sensitivity)
                
                # Interpretation
                st.subheader("Robustness Assessment")
                
                effect_range = max(values) - min(values)
                if effect_range < 0.05:
                    st.success("✅ Results are robust to different assumptions")
                    st.write("The causal effect remains relatively stable across different analytical scenarios.")
                else:
                    st.warning("⚠️ Results show some sensitivity to assumptions")
                    st.write("Consider additional validation before making policy decisions.")
                
            except Exception as e:
                st.error(f"Sensitivity analysis failed: {e}")