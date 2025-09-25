import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_metric_card(title, value, delta=None, delta_color="normal", help_text=None):
    """Display a metric card with optional delta"""
    
    if delta is not None:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help=help_text
        )
    else:
        st.metric(
            label=title,
            value=value,
            help=help_text
        )

def display_gauges(metrics_dict, title="Performance Metrics"):
    """Display multiple metrics as gauges"""
    
    num_metrics = len(metrics_dict)
    cols = st.columns(num_metrics)
    
    for idx, (metric_name, metric_value) in enumerate(metrics_dict.items()):
        with cols[idx]:
            display_gauge(metric_name, metric_value)

def display_gauge(metric_name, value, min_val=0, max_val=1):
    """Display a single metric as a gauge"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': metric_name},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, 0.5], 'color': "lightgray"},
                {'range': [0.5, 0.7], 'color': "yellow"},
                {'range': [0.7, max_val], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

def display_kpi_grid(kpis, columns=3):
    """Display a grid of KPI cards"""
    
    num_kpis = len(kpis)
    num_rows = (num_kpis + columns - 1) // columns
    
    for row in range(num_rows):
        cols = st.columns(columns)
        for col in range(columns):
            idx = row * columns + col
            if idx < num_kpis:
                kpi = kpis[idx]
                with cols[col]:
                    display_metric_card(
                        kpi['title'],
                        kpi['value'],
                        kpi.get('delta'),
                        kpi.get('delta_color', 'normal'),
                        kpi.get('help_text')
                    )

def create_performance_dashboard(metrics_dict):
    """Create a comprehensive performance dashboard"""
    
    st.subheader("Model Performance Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card("Accuracy", f"{metrics_dict.get('accuracy', 0):.2%}")
    with col2:
        display_metric_card("Precision", f"{metrics_dict.get('precision', 0):.2%}")
    with col3:
        display_metric_card("Recall", f"{metrics_dict.get('recall', 0):.2%}")
    with col4:
        display_metric_card("F1-Score", f"{metrics_dict.get('f1_score', 0):.2%}")
    
    # Business metrics row
    if 'business_metrics' in metrics_dict:
        business_metrics = metrics_dict['business_metrics']
        
        st.subheader("Business Impact")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            display_metric_card(
                "Cost Savings", 
                f"${business_metrics.get('cost_savings', 0):,.0f}",
                help_text="Estimated annual savings from churn reduction"
            )
        with col6:
            display_metric_card(
                "Customers Retained",
                business_metrics.get('customers_retained', 0),
                help_text="Number of customers potentially retained"
            )
        with col7:
            display_metric_card(
                "ROI",
                f"{business_metrics.get('roi', 0):.0%}",
                help_text="Return on investment from retention efforts"
            )
        with col8:
            display_metric_card(
                "Risk Reduction",
                f"{business_metrics.get('risk_reduction', 0):.0%}",
                help_text="Reduction in churn risk"
            )

def display_causal_effect_metrics(causal_results):
    """Display metrics for causal analysis results"""
    
    if causal_results.empty:
        st.warning("No causal results available")
        return
    
    significant_effects = causal_results[causal_results['Significant']]
    
    st.subheader("Causal Impact Summary")
    
    if not significant_effects.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Significant Effects",
                len(significant_effects),
                help_text="Number of statistically significant causal relationships"
            )
        
        with col2:
            avg_effect = significant_effects['Estimated_Effect'].mean()
            st.metric(
                "Average Effect Size",
                f"{avg_effect:.3f}",
                help_text="Average impact of significant treatments"
            )
        
        with col3:
            strongest_effect = significant_effects['Estimated_Effect'].abs().max()
            st.metric(
                "Strongest Effect",
                f"{strongest_effect:.3f}",
                help_text="Largest absolute effect size"
            )
        
        # Display individual significant effects
        st.subheader("Significant Causal Effects")
        
        for _, effect in significant_effects.iterrows():
            effect_color = "red" if effect['Estimated_Effect'] > 0 else "green"
            effect_direction = "increases" if effect['Estimated_Effect'] > 0 else "decreases"
            
            st.info(
                f"**{effect['Treatment']}** {effect_direction} churn probability by "
                f"**{abs(effect['Estimated_Effect']):.3f}** "
                f"(p-value: {effect['P_Value']:.3f})"
            )
    else:
        st.warning("No statistically significant causal effects detected")