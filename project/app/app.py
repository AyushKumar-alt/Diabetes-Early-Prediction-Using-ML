"""
Streamlit app for Diabetes Risk Prediction
Clean pipeline - NO SMOTE
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import predict_risk, validate_all_no_input

# Page config
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-low { background-color: #d4edda; border: 2px solid #28a745; }
    .risk-moderate { background-color: #fff3cd; border: 2px solid #ffc107; }
    .risk-high { background-color: #f8d7da; border: 2px solid #dc3545; }
    .risk-very-high { background-color: #f5c6cb; border: 2px solid #721c24; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar inputs
st.sidebar.header("📋 Patient Health Information")

# Demographics
st.sidebar.subheader("📊 Demographics")
age = st.sidebar.slider("Age Group (1-13)", 1, 13, 5, help="1=18-24, 2=25-29, ..., 13=80+")
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
education = st.sidebar.slider("Education Level", 1, 6, 4, help="1=Never, 2=Elementary, ..., 6=College")
income = st.sidebar.slider("Income Level", 1, 8, 5, help="1=<$10k, ..., 8=$75k+")

# Health Conditions
st.sidebar.subheader("💊 Health Conditions")
high_bp = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
high_chol = st.sidebar.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
chol_check = st.sidebar.selectbox("Cholesterol Check (Past 5 Years)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
stroke = st.sidebar.selectbox("History of Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.sidebar.selectbox("Heart Disease or Attack", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Physical Health
st.sidebar.subheader("📏 Physical Health")
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=100.0, value=22.0, step=0.1)
gen_hlth = st.sidebar.slider("General Health", 1, 5, 2, help="1=Excellent, 5=Poor")
phys_hlth = st.sidebar.number_input("Physical Health (Days in Past 30)", min_value=0, max_value=30, value=0)
ment_hlth = st.sidebar.number_input("Mental Health (Days in Past 30)", min_value=0, max_value=30, value=0)
diff_walk = st.sidebar.selectbox("Difficulty Walking/Climbing Stairs", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Lifestyle
st.sidebar.subheader("🏃 Lifestyle Factors")
phys_activity = st.sidebar.selectbox("Physical Activity (Past 30 Days)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
smoker = st.sidebar.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Smoker")
fruits = st.sidebar.selectbox("Fruits (1+ Times/Day)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
veggies = st.sidebar.selectbox("Vegetables (1+ Times/Day)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
hvy_alcohol = st.sidebar.selectbox("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Healthcare
st.sidebar.subheader("🏥 Healthcare Access")
any_healthcare = st.sidebar.selectbox("Has Healthcare Coverage", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
no_doc_cost = st.sidebar.selectbox("Could Not See Doctor (Cost)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Create patient data
patient_data = {
    'HighBP': high_bp,
    'HighChol': high_chol,
    'CholCheck': chol_check,
    'BMI': bmi,
    'Smoker': smoker,
    'Stroke': stroke,
    'HeartDiseaseorAttack': heart_disease,
    'PhysActivity': phys_activity,
    'Fruits': fruits,
    'Veggies': veggies,
    'HvyAlcoholConsump': hvy_alcohol,
    'AnyHealthcare': any_healthcare,
    'NoDocbcCost': no_doc_cost,
    'GenHlth': gen_hlth,
    'MentHlth': ment_hlth,
    'PhysHlth': phys_hlth,
    'DiffWalk': diff_walk,
    'Sex': sex,
    'Age': age,
    'Education': education,
    'Income': income
}

# Prediction button
if st.sidebar.button("🔍 Predict Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing patient data..."):
        try:
            result = predict_risk(patient_data, return_explanation=True)
            st.session_state['prediction_result'] = result
            st.session_state['patient_data'] = patient_data
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state['prediction_result'] = None

# Display results
if 'prediction_result' in st.session_state and st.session_state['prediction_result'] is not None:
    result = st.session_state['prediction_result']
    prob_percent = result['probability'] * 100
    
    # Main results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Risk Assessment Results")
        
        # Risk level display
        risk_level = result['risk_level']
        risk_class = risk_level.lower().replace(" ", "-")
        
        st.markdown(f"""
        <div class="risk-box risk-{risk_class}">
            <h2>{result['risk_icon']} {risk_level}</h2>
            <p style="font-size: 1.1rem;"><strong>Prediction:</strong> {result['prediction']}</p>
            <p style="font-size: 1.1rem;"><strong>Probability:</strong> {prob_percent:.2f}%</p>
            <p style="font-size: 1.1rem;"><strong>Confidence:</strong> {result['confidence']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability gauge (using plotly if available)
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_percent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "At-Risk Probability (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': result['risk_color']},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.progress(prob_percent / 100)
            st.caption(f"At-Risk Probability: {prob_percent:.2f}%")
    
    with col2:
        st.subheader("📈 Key Metrics")
        st.metric("Probability", f"{prob_percent:.2f}%")
        st.metric("Threshold", f"{result['threshold_used']*100:.0f}%")
        
        confidence_emoji = {
            "Low": "🟢",
            "Moderate": "🟡",
            "High": "🟠",
            "Very High": "🔴"
        }.get(result['confidence'], "⚪")
        st.metric("Confidence", f"{confidence_emoji} {result['confidence']}")
        
        st.subheader("💡 Recommendation")
        st.info(result['recommendation'])
    
    # SHAP Explanation
    if 'explanation' in result and result['explanation'] is not None:
        st.markdown("---")
        st.subheader("🔍 Explanation of Your Risk Score")
        
        explanations = result['explanation']['top_features']
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Patient Profile", "Detailed Data"])
        
        with tab1:
            # SHAP Bar Chart
            try:
                import plotly.express as px
                
                # Prepare data for plot
                shap_data = pd.DataFrame(explanations)
                shap_data['Color'] = shap_data['shap_value'].apply(lambda x: 'Risk Increasing' if x > 0 else 'Risk Reducing')
                
                fig_shap = px.bar(
                    shap_data, 
                    x='shap_value', 
                    y='feature', 
                    orientation='h',
                    color='Color',
                    color_discrete_map={'Risk Increasing': 'red', 'Risk Reducing': 'green'},
                    title="Top Risk Factors (SHAP Values)",
                    labels={'shap_value': 'Impact on Risk Score', 'feature': 'Feature'}
                )
                fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_shap, use_container_width=True)
                
                st.caption("Positive values increase risk, negative values reduce risk.")
                
            except ImportError:
                st.error("Plotly is required for visualizations.")
                st.write("**Top factors affecting your risk:**")
                for exp in explanations[:5]:
                    icon = "🔺" if exp['shap_value'] > 0 else "🔻"
                    st.write(f"{icon} **{exp['feature']}** {exp['direction']} your risk ({exp['shap_value']:.4f})")

        with tab2:
            # Radar Chart for Patient Profile
            try:
                import plotly.graph_objects as go
                
                # Normalize key metrics for radar chart (0-1 scale)
                # Define max values for normalization (approximate)
                max_values = {
                    'BMI': 50,
                    'Age': 13,
                    'GenHlth': 5,
                    'PhysHlth': 30,
                    'MentHlth': 30,
                    'Education': 6,
                    'Income': 8
                }
                
                radar_metrics = ['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth', 'Education', 'Income']
                patient_vals = []
                
                # Get values from session state patient_data
                p_data = st.session_state['patient_data']
                
                for m in radar_metrics:
                    val = p_data.get(m, 0)
                    # Normalize
                    norm_val = min(val / max_values[m], 1.0)
                    patient_vals.append(norm_val)
                
                # Close the loop
                radar_metrics.append(radar_metrics[0])
                patient_vals.append(patient_vals[0])
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=patient_vals,
                    theta=radar_metrics,
                    fill='toself',
                    name='Patient Profile'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False,
                    title="Patient Health Profile (Normalized)"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                st.caption("Values are normalized relative to typical maximums.")
                
            except Exception as e:
                st.error(f"Could not generate radar chart: {e}")

        with tab3:
            # Detailed Input Data
            st.write("### Patient Input Data")
            st.json(st.session_state['patient_data'])
            
            st.write("### Detailed SHAP Values")
            exp_df = pd.DataFrame(explanations)
            exp_df.columns = ['Feature', 'SHAP Value', 'Effect']
            st.dataframe(exp_df, use_container_width=True, hide_index=True)
    
    # Export report
    st.markdown("---")
    if st.button("📥 Download Report"):
        report = f"""
# Diabetes Risk Assessment Report

## Patient Information
- **Risk Level**: {result['risk_level']}
- **Prediction**: {result['prediction']}
- **Probability**: {prob_percent:.2f}%
- **Confidence**: {result['confidence']}

## Recommendation
{result['recommendation']}

## Assessment Details
- Threshold Used: {result['threshold_used']*100:.0f}%
- Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*This is a screening tool and not a diagnostic tool. Please consult with a healthcare provider for medical advice.*
        """
        st.download_button(
            label="Download TXT Report",
            data=report,
            file_name=f"diabetes_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

else:
    st.info("👈 Please fill in the patient information in the sidebar and click 'Predict Risk' to begin.")
    
    # Validation test
    with st.expander("🔍 Model Validation Test", expanded=False):
        if st.button("Test: All NO Inputs → LOW Risk"):
            try:
                validation = validate_all_no_input()
                if validation['passed']:
                    st.success("✅ Model validation PASSED: All NO inputs correctly predict LOW risk")
                else:
                    st.error("❌ Model validation FAILED: Check model calibration")
            except Exception as e:
                st.error(f"Validation error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🩺 Diabetes Risk Prediction System | Clean ML Pipeline (NO SMOTE)</p>
    <p><small>For screening purposes only. Not a substitute for professional medical advice.</small></p>
</div>
""", unsafe_allow_html=True)

