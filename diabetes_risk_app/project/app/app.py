"""
Professional Streamlit app for Diabetes Risk Prediction
Includes: SHAP explainability, Gauge Meter, PDF Report Generation
"""

import streamlit as st

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="Assist Diabetes AI - Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import sys
import os
from io import BytesIO
import tempfile

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This is diabetes_risk_app/project/
sys.path.insert(0, parent_dir)

from predict import predict_risk, validate_all_no_input

# Try to import PDF generator (optional)
PDF_AVAILABLE = False
_pdf_error = None
_pdf_traceback = None

try:
    # First check if reportlab is available
    import reportlab
    
    # generate_pdf.py is in the same directory as predict.py (parent_dir)
    # So we can import it directly since parent_dir is already in sys.path
    from generate_pdf import generate_pdf_report
    PDF_AVAILABLE = True
except Exception as e:
    PDF_AVAILABLE = False
    # Store error for debugging
    import traceback
    _pdf_error = str(e)
    _pdf_traceback = traceback.format_exc()
    # Debug: print to console (visible in Streamlit logs)
    print(f"❌ PDF import failed: {_pdf_error}")
    print(f"Traceback: {_pdf_traceback}")
else:
    print("✅ PDF import successful! PDF_AVAILABLE = True")

# Custom CSS - Enhanced UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
        border: 3px solid #28a745; 
    }
    .risk-moderate { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%); 
        border: 3px solid #ffc107; 
    }
    .risk-high { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c2c7 100%); 
        border: 3px solid #dc3545; 
    }
    .risk-very-high { 
        background: linear-gradient(135deg, #f5c6cb 0%, #f1aeb5 100%); 
        border: 3px solid #721c24; 
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        margin-bottom: 1rem !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #666 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #1E88E5 !important;
        font-weight: bold !important;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin-top: 1rem;
    }
    h2, h3 {
        color: #1E88E5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🩺 Assist Diabetes AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional Diabetes Risk Assessment System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Patient Information
st.sidebar.header("📋 Patient Information")

# Patient Demographics (for PDF)
patient_name = st.sidebar.text_input("Patient Name", value="")
patient_age = st.sidebar.text_input("Age", value="")
patient_gender = st.sidebar.selectbox("Gender", ["", "Male", "Female", "Other"])
patient_phone = st.sidebar.text_input("Phone", value="")
patient_id = st.sidebar.text_input("Patient ID", value=f"PAT-{pd.Timestamp.now().strftime('%Y%m%d')}-001")

st.sidebar.markdown("---")
st.sidebar.header("📊 Health Indicators")

# Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age Group (1-13)", 1, 13, 5, help="1=18-24, 2=25-29, ..., 13=80+")
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
education = st.sidebar.slider("Education Level", 1, 6, 4, help="1=Never, 2=Elementary, ..., 6=College")
income = st.sidebar.slider("Income Level", 1, 8, 5, help="1=<$10k, ..., 8=$75k+")

# Health Conditions
st.sidebar.subheader("Health Conditions")
high_bp = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
high_chol = st.sidebar.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
chol_check = st.sidebar.selectbox("Cholesterol Check (Past 5 Years)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
stroke = st.sidebar.selectbox("History of Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.sidebar.selectbox("Heart Disease or Attack", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Physical Health
st.sidebar.subheader("Physical Health")
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=100.0, value=22.0, step=0.1)
gen_hlth = st.sidebar.slider("General Health", 1, 5, 2, help="1=Excellent, 5=Poor")
phys_hlth = st.sidebar.number_input("Physical Health (Days in Past 30)", min_value=0, max_value=30, value=0)
ment_hlth = st.sidebar.number_input("Mental Health (Days in Past 30)", min_value=0, max_value=30, value=0)
diff_walk = st.sidebar.selectbox("Difficulty Walking/Climbing Stairs", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Lifestyle
st.sidebar.subheader("Lifestyle Factors")
phys_activity = st.sidebar.selectbox("Physical Activity (Past 30 Days)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
smoker = st.sidebar.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Smoker")
fruits = st.sidebar.selectbox("Fruits (1+ Times/Day)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
veggies = st.sidebar.selectbox("Vegetables (1+ Times/Day)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
hvy_alcohol = st.sidebar.selectbox("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Healthcare
st.sidebar.subheader("Healthcare Access")
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
            st.session_state['patient_info'] = {
                'name': patient_name or 'N/A',
                'age': patient_age or 'N/A',
                'gender': patient_gender or 'N/A',
                'phone': patient_phone or 'N/A',
                'patient_id': patient_id or 'N/A'
            }
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state['prediction_result'] = None

# Display results
if 'prediction_result' in st.session_state and st.session_state['prediction_result'] is not None:
    result = st.session_state['prediction_result']
    prob_percent = result['probability'] * 100
    
    # Main results section
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
        
        # Gauge Meter (Speedometer-style)
        st.subheader("🎯 Risk Gauge Meter")
        try:
            import plotly.graph_objects as go
            
            # Determine gauge color based on risk
            if prob_percent < 20:
                gauge_color = "#4CAF50"  # Green
            elif prob_percent < 40:
                gauge_color = "#FFC107"  # Yellow
            elif prob_percent < 70:
                gauge_color = "#FF9800"  # Orange
            else:
                gauge_color = "#F44336"  # Red
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_percent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "At-Risk Probability (%)", 'font': {'size': 20}},
                delta={'reference': 30, 'position': "top"},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': "#E8F5E9"},
                        {'range': [20, 40], 'color': "#FFF9C4"},
                        {'range': [40, 70], 'color': "#FFE0B2"},
                        {'range': [70, 100], 'color': "#FFCDD2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            fig.update_layout(
                height=400,
                paper_bgcolor="white",
                font={'color': "darkblue", 'family': "Arial"}
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.progress(prob_percent / 100)
            st.caption(f"At-Risk Probability: {prob_percent:.2f}%")
    
    with col2:
        st.markdown("### 📈 Key Metrics")
        
        # Probability metric
        with st.container():
            st.metric("Probability", f"{prob_percent:.2f}%")
        
        # Threshold metric
        with st.container():
            st.metric("Threshold", f"{result['threshold_used']*100:.0f}%")
        
        # Confidence metric
        confidence_emoji = {
            "Low": "🟢",
            "Moderate": "🟡",
            "High": "🟠",
            "Very High": "🔴"
        }.get(result['confidence'], "⚪")
        
        with st.container():
            st.metric("Confidence", f"{confidence_emoji} {result['confidence']}")
        
        st.markdown("---")
        st.markdown("### 💡 Recommendation")
        st.markdown(f"""
        <div class="recommendation-box">
            <p style="margin: 0; font-size: 1rem; line-height: 1.6;">{result['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # SHAP Explainability Section
    if 'explanation' in result and result['explanation'] is not None:
        st.markdown("---")
        st.subheader("🔍 Explainable AI (XAI) - Personalized Risk Factors")
        
        with st.expander("ℹ️ About Explainable AI", expanded=False):
            st.markdown("""
            **Explainable AI (XAI) Integration**
            
            This system uses SHAP (SHapley Additive exPlanations) to provide transparent, interpretable predictions.
            Each feature is assigned a contribution score showing how much it increased or decreased your risk.
            
            **Understanding the Results:**
            - **Positive values (↑)**: Factors that increase your diabetes risk
            - **Negative values (↓)**: Factors that decrease your diabetes risk
            - **Larger absolute values**: More significant impact on the prediction
            
            **Benefits:**
            - Transparent, trustworthy predictions you can understand
            - Personalized insights into your health factors
            - Actionable information for risk reduction
            - Clinical interpretability for healthcare providers
            """)
        
        explanations = result['explanation']['top_features']
        
        # Feature contributions bar chart
        st.subheader("📊 Feature Contribution Bar Chart")
        try:
            import matplotlib.pyplot as plt
            
            features = [exp['feature'] for exp in explanations[:10]]
            values = [exp['shap_value'] for exp in explanations[:10]]
            colors_list = ['#d32f2f' if v > 0 else '#388e3c' for v in values]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(features, values, color=colors_list, alpha=0.7)
            ax.set_xlabel('SHAP Value (Contribution to Risk)', fontsize=12, fontweight='bold')
            ax.set_title('Top 10 Feature Contributions to This Prediction', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.4f}',
                       va='center', ha='left' if val > 0 else 'right', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate bar chart: {str(e)}")
        
        # Textual explanations
        st.subheader("📋 Personalized Risk Factor Analysis")
        st.write("**Top factors affecting your risk:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔺 Factors Increasing Risk:**")
            for exp in explanations[:5]:
                if exp['shap_value'] > 0:
                    st.write(f"• **{exp['feature']}**: {exp['direction']} risk ({exp['shap_value']:.4f})")
        
        with col2:
            st.write("**🔻 Factors Decreasing Risk:**")
            for exp in explanations[:5]:
                if exp['shap_value'] < 0:
                    st.write(f"• **{exp['feature']}**: {exp['direction']} risk ({exp['shap_value']:.4f})")
        
        # Detailed table
        st.subheader("📋 Detailed Feature Contributions")
        contrib_data = []
        for exp in explanations:
            contrib_data.append({
                'Feature': exp['feature'],
                'SHAP Value': f"{exp['shap_value']:.4f}",
                'Effect': exp['direction'].title() + ' risk',
                'Impact': '↑ Increases' if exp['shap_value'] > 0 else '↓ Decreases'
            })
        
        contrib_df = pd.DataFrame(contrib_data)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        
        # Store feature contributions for PDF
        st.session_state['feature_contrib'] = [
            (exp['feature'], f"{exp['shap_value']:+.4f}") for exp in explanations[:10]
        ]
    
    # PDF Report Generation
    st.markdown("---")
    st.subheader("📄 Generate Medical Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if PDF_AVAILABLE:
            if st.button("📥 Generate & Download PDF Report", type="primary", use_container_width=True):
                try:
                    # Prepare data for PDF
                    patient_info = st.session_state.get('patient_info', {
                        'name': patient_name or 'N/A',
                        'age': patient_age or 'N/A',
                        'gender': patient_gender or 'N/A',
                        'phone': patient_phone or 'N/A',
                        'patient_id': patient_id or 'N/A'
                    })
                    
                    prediction_info = {
                        'probability': result['probability'],
                        'risk_level': result['risk_level'],
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'recommendation': result['recommendation']
                    }
                    
                    feature_contrib = st.session_state.get('feature_contrib', [])
                    
                    # Generate PDF in temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        pdf_path = tmp_file.name
                        generate_pdf_report(pdf_path, patient_info, prediction_info, feature_contrib)
                        
                        # Read PDF bytes
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        # Clean up
                        os.unlink(pdf_path)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"Diabetes_Risk_Report_{patient_info['patient_id']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime='application/pdf',
                        use_container_width=True
                    )
                    
                    st.success("✅ PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ PDF generation is currently unavailable.")
            with st.expander("🔧 Troubleshooting & Debug Info", expanded=False):
                if _pdf_error:
                    st.write("**Error details:**")
                    st.code(_pdf_error, language='text')
                    if _pdf_traceback:
                        st.write("**Full traceback:**")
                        st.code(_pdf_traceback, language='text')
                else:
                    st.info("No error details available. PDF import check failed silently.")
                
                st.write("**To fix:**")
                st.code("""
# 1. Install reportlab (if not installed):
pip install reportlab

# 2. Verify installation:
python -c "import reportlab; print('reportlab OK')"

# 3. Test PDF module import:
cd diabetes_risk_app\\project
python -c "from generate_pdf import generate_pdf_report; print('PDF module OK')"

# 4. Restart Streamlit app:
streamlit run app/app.py

# If issues persist, clear browser cache or restart Streamlit
                """)

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
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>🩺 Assist Diabetes AI</strong></p>
    <p>Email: support@diascan.ai | For Any Query Dial: +91 9931147182</p>
    <p><small>This is an AI-assisted screening tool, not a diagnostic test. Please consult a healthcare professional for medical decisions.</small></p>
</div>
""", unsafe_allow_html=True)

