"""
Assist Diabetes AI - Full production-ready Streamlit app (Single-file)
Features:
 - Loads calibrated XGBoost model (joblib)
 - Correct BRFSS mapping of inputs
 - Per-patient SHAP explanation (TreeExplainer)
 - Hybrid medical-rule layer (hierarchical override + additive boosts)
 - Hospital-style PDF generation (ReportLab) with SHAP image and top contributors
 - Debug output of model input (mapped)
 - Uses uploaded sample PDF as visual reference: /mnt/data/Project_ML-1.pdf
"""

import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
import plotly.express as px
import smtplib
import ssl
from email.message import EmailMessage
import base64
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------- Paths & config ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # diabetes_risk_app/app -> parent
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# sample uploaded lab pdf path - visual reference
SAMPLE_PDF_PATH = "/mnt/data/Project_ML-1.pdf"

MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model_calibrated.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLS_PATH = os.path.join(MODELS_DIR, "model_columns.pkl")

# ---------- Safe loaders ----------
def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

MODEL = safe_load_joblib(MODEL_PATH)
SCALER = safe_load_joblib(SCALER_PATH)
MODEL_COLS = safe_load_joblib(COLS_PATH)

# ---------- BRFSS labels ----------
AGE_LABELS = {
    1: "18-24",2: "25-29",3: "30-34",4: "35-39",5: "40-44",
    6: "45-49",7: "50-54",8: "55-59",9: "60-64",10: "65-69",
    11: "70-74",12: "75-79",13: "80+"
}
EDU_LABELS = {
    1: "Never attended school",2: "Elementary",3: "Some high school",
    4: "High school graduate",5: "Some college",6: "College graduate"
}
INCOME_LABELS = {
    1: "< $10k",2:"$10k-15k",3:"$15k-20k",4:"$20k-25k",5:"$25k-35k",
    6:"$35k-50k",7:"$50k-75k",8:"> $75k"
}

# ---------- Hybrid rule utils (file: utils/hybrid_rules.py equivalent) ----------
def compute_medical_boost(patient: dict) -> (float, float):
    """
    Return (boost, override_min)
    boost: additive adjustment to calibrated ML probability (can be negative)
    override_min: None or minimum probability (e.g. 0.40) enforced via hierarchical strategy
    """
    boost = 0.0
    override = None

    # Rule 1: Major cardio comorbidity override
    if (patient.get("Stroke",0) == 1 or patient.get("HeartDiseaseorAttack",0) == 1) and \
       (patient.get("HighBP",0) == 1 or patient.get("HighChol",0) == 1):
        override = 0.40  # at least High Risk

    # Rule 2: metabolic cluster booster
    bmi = float(patient.get("BMI", 0.0))
    if bmi >= 30:
        boost += 0.15
    elif bmi >= 25:
        boost += 0.04

    if patient.get("HighBP",0) == 1 and patient.get("HighChol",0) == 1:
        boost += 0.15

    # Rule 3: functional impairment
    if patient.get("DiffWalk",0) == 1 or int(patient.get("PhysHlth",0)) >= 15:
        boost += 0.08

    # Rule 4: age boost (60+)
    age = int(patient.get("Age", 1))
    if age >= 9:
        boost += 0.07

    # Rule 5: healthcare access caution
    if patient.get("AnyHealthcare",1) == 0 and patient.get("CholCheck",1) == 0:
        boost += 0.05

    # Rule 6: protective offset
    if patient.get("PhysActivity",0) == 1 and patient.get("Smoker",0) == 0 and patient.get("HvyAlcoholConsump",0) == 0:
        boost -= 0.05

    # Bound boost to avoid dominating the ML
    boost = float(np.clip(boost, -0.2, 0.45))
    return boost, override

def combine_ml_and_medical(ml_prob: float, patient: dict, method: str = "hierarchical"):
    boost, override = compute_medical_boost(patient)
    combined = float(np.clip(ml_prob + boost, 0.0, 0.9999))
    if method == "hierarchical" and override is not None:
        combined = max(combined, override)
    # risk strata
    if combined < 0.20:
        risk_level = "Low Risk"
    elif combined < 0.40:
        risk_level = "Moderate Risk"
    elif combined < 0.70:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"
    return {
        "ml_prob": float(ml_prob),
        "medical_boost": float(boost),
        "override_min": override,
        "combined_prob": float(combined),
        "risk_level": risk_level
    }

# ---------- SHAP helper ----------
SHAP_AVAILABLE = True
try:
    import shap
    # matplotlib already imported
except Exception:
    SHAP_AVAILABLE = False

def compute_shap_instance(model, X_df, top_n=10, save_path=None):
    """
    Returns list of dicts [{'feature':..., 'shap_value':...}], and writes PNG if save_path provided.
    Uses TreeExplainer on the tree model (handles wrappers).
    """
    if not SHAP_AVAILABLE:
        return [], None
    try:
        base = getattr(model, "base_estimator", model)  # if calibrated wrapper
        expl = shap.TreeExplainer(base)
        shap_vals = expl.shap_values(X_df)
        # shap_vals may be list of arrays for multi-class; for binary shap_vals[1] is class1
        arr = None
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            # shap_vals[1] shape (n_samples, n_features)
            arr = shap_vals[1][0]
        else:
            arr = np.array(shap_vals).flatten()
            if arr.shape[0] != X_df.shape[1]:
                arr = arr[:X_df.shape[1]]
        feat_names = X_df.columns.tolist()
        pairs = [{"feature": f, "shap_value": float(v)} for f, v in zip(feat_names, arr)]
        pairs = sorted(pairs, key=lambda x: abs(x['shap_value']), reverse=True)[:top_n]

        # save horizontal bar chart
        if save_path:
            feats = [p['feature'] for p in pairs][::-1]
            vals = [p['shap_value'] for p in pairs][::-1]
            plt.figure(figsize=(6, max(2, len(feats)*0.45)))
            plt.barh(feats, vals)
            plt.xlabel("SHAP value (impact on probability)")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            return pairs, save_path
        return pairs, None
    except Exception as e:
        return [], None

# Medical Recommendation Engine
def generate_recommendation(prob):
    if prob < 0.20:
        return {
            "risk_label": "Low Risk",
            "advice": (
                "Your risk of diabetes is low. Maintain healthy lifestyle habits and "
                "continue annual preventive checkups."
            ),
            "precautions": [
                "Maintain a balanced diet.",
                "Exercise 150 minutes per week.",
                "Avoid sugary drinks."
            ]
        }
    elif prob < 0.40:
        return {
            "risk_label": "Moderate Risk",
            "advice": (
                "Your diabetes risk is moderate. Lifestyle improvements are recommended. "
                "Consider a preventive consultation within 3–6 months."
            ),
            "precautions": [
                "Walk 30 minutes daily.",
                "Reduce sugar and fatty foods.",
                "Check fasting glucose yearly."
            ]
        }
    elif prob < 0.60:
        return {
            "risk_label": "High Risk",
            "advice": (
                "Your diabetes risk is high. A medical evaluation is recommended within "
                "1–2 months."
            ),
            "precautions": [
                "Eliminate sugary drinks.",
                "Follow a balanced plate diet.",
                "Exercise 200 minutes weekly."
            ]
        }
    elif prob < 0.80:
        return {
            "risk_label": "Very High Risk",
            "advice": (
                "Your diabetes risk is very high. A clinical evaluation is needed soon."
            ),
            "precautions": [
                "Avoid high-glycemic foods.",
                "Follow a doctor-guided diet.",
                "Practice stress-reducing exercises."
            ]
        }
    else:
        return {
            "risk_label": "Critical Risk",
            "advice": (
                "Your diabetes risk is extremely high. Seek medical evaluation "
                "immediately."
            ),
            "precautions": [
                "Strict no-sugar diet.",
                "Avoid alcohol and smoking.",
                "Monitor for severe symptoms."
            ]
        }

# ---------- PDF Report Generation (ReportLab) ----------
def generate_pdf_report(patient_info, prediction_info, recommendation_data=None):
    """
    Generate a professional diabetes risk assessment PDF report using ReportLab.
    Returns BytesIO object containing PDF bytes.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab not available. Install with: pip install reportlab")
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#0b6fa4'),
        spaceAfter=6,
        alignment=1  # CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#0b6fa4'),
        spaceAfter=8,
        spaceBefore=8,
        borderColor=colors.HexColor('#0b6fa4'),
        borderWidth=0.5,
        borderPadding=5,
        borderRadius=2
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=6
    )
    warning_style = ParagraphStyle(
        'Warning',
        parent=styles['BodyText'],
        fontSize=9,
        textColor=colors.red,
        spaceAfter=6,
        borderColor=colors.red,
        borderWidth=1,
        borderPadding=8,
        leftIndent=10
    )
    
    # Title
    story.append(Paragraph("ASSIST DIABETES AI", title_style))
    story.append(Paragraph("Diabetes Risk Assessment Report", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Organization info
    story.append(Paragraph(
        "CSE Incubator, Vidyashilp University Bangalore<br/>Email: support_assist_diabetes@gmail.com<br/>Contact: +91 9911471182",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information Section
    story.append(Paragraph("PATIENT INFORMATION", heading_style))
    p = patient_info or {}
    patient_data = [
        ["Name:", p.get('name', 'N/A')],
        ["Patient ID:", p.get('patient_id', 'N/A')],
        ["Age Group:", p.get('age_display', 'N/A')],
        ["Gender:", p.get('gender', 'N/A')],
        ["Phone:", p.get('phone', 'N/A')],
        ["Referred By:", p.get('referred_by', 'Self')],
        ["Report Date:", datetime.now().strftime('%d-%b-%Y %H:%M:%S')],
    ]
    patient_table = Table(patient_data, colWidths=[1.5*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Risk Prediction Results
    story.append(Paragraph("RISK PREDICTION RESULTS", heading_style))
    pr = prediction_info or {}
    ml_prob = pr.get('ml_prob', 0.0) * 100
    medical_boost = pr.get('medical_boost', 0.0) * 100
    combined_prob = pr.get('combined_prob', 0.0) * 100
    risk_level = pr.get('risk_level', 'UNKNOWN')
    
    results_data = [
        ["Parameter", "Value"],
        ["ML Probability", f"{ml_prob:.2f}%"],
        ["Medical Boost", f"{medical_boost:+.1f}%"],
        ["Combined Probability", f"{combined_prob:.2f}%"],
        ["Risk Level", risk_level.upper()],
    ]
    results_table = Table(results_data, colWidths=[2*inch, 4*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0b6fa4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f8ff'), colors.white])
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendation Section
    if recommendation_data:
        story.append(Paragraph("MEDICAL RECOMMENDATION", heading_style))
        rec_data = [
            ["Category:", recommendation_data.get('risk_level', 'N/A')],
            ["Recommendation:", recommendation_data.get('recommendation', 'Consult healthcare professional')],
            ["Next Steps:", recommendation_data.get('next_steps', 'Schedule regular check-ups')],
        ]
        for item in recommendation_data.get('precautions', []):
            rec_data.append(["Precaution:", f"• {item}"])
        
        rec_table = Table(rec_data, colWidths=[1.5*inch, 4.5*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff4e6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    story.append(Paragraph("IMPORTANT DISCLAIMER", heading_style))
    disclaimer_text = (
        "<b>SCREENING TOOL ONLY:</b> This AI-assisted diabetes risk assessment is a screening tool "
        "and <b>NOT a medical diagnosis</b>. Results should not be used for self-diagnosis or treatment decisions. "
        "Please consult a certified healthcare professional for proper examination, diagnosis, and medical advice. "
        "Any actions taken based on this report are at the patient's own risk."
    )
    story.append(Paragraph(disclaimer_text, warning_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Footer
    footer_text = f"Generated on {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} | Assist Diabetes AI v1.0"
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=1, textColor=colors.grey)))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------- UI (Streamlit) ----------
st.set_page_config(page_title="Assist Diabetes AI", layout="wide")
st.markdown("<h1 style='text-align:center; color:#0b6fa4;'>🩺 Assist Diabetes AI — Hybrid Screening</h1>", unsafe_allow_html=True)
st.markdown("---")

# Horizontal patient info
c1,c2,c3,c4 = st.columns(4)
with c1:
    patient_name = st.text_input("Patient Name")
with c2:
    patient_id = st.text_input("Patient ID", value=f"PAT-{datetime.now().strftime('%Y%m%d%H%M')}")
with c3:
    phone = st.text_input("Phone")
with c4:
    referred_by = st.text_input("Referred By", value="Self")

# Patient email for sending report
patient_email = st.text_input("Patient Email (optional, used to send report)")

st.markdown("### Demographics")
d1,d2,d3,d4 = st.columns(4)
with d1:
    Age = st.selectbox("Age Group (BRFSS code)", list(AGE_LABELS.keys()), format_func=lambda x: f"{AGE_LABELS[x]} ({x})", index=6)
with d2:
    Sex = st.selectbox("Sex", ["Female","Male"])
with d3:
    Education = st.selectbox("Education", list(EDU_LABELS.keys()), format_func=lambda x: EDU_LABELS[x], index=5)
with d4:
    Income = st.selectbox("Income", list(INCOME_LABELS.keys()), format_func=lambda x: INCOME_LABELS[x], index=3)

# Health rows
st.markdown("### Health indicators")
h1,h2,h3,h4 = st.columns(4)
with h1:
    HighBP = st.selectbox("High BP", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    HighChol = st.selectbox("High Cholesterol", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    CholCheck = st.selectbox("Cholesterol Check (past5yrs)", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
with h2:
    Stroke = st.selectbox("History of Stroke", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    HeartDiseaseorAttack = st.selectbox("Heart Disease / Attack", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    BMI = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
with h3:
    PhysActivity = st.selectbox("Physical Activity (1=yes)", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    Smoker = st.selectbox("Smoker", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    Fruits = st.selectbox("Fruits (1+/day)", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
with h4:
    Veggies = st.selectbox("Veggies (1+/day)", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    HvyAlcoholConsump = st.selectbox("Heavy Alcohol", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    AnyHealthcare = st.selectbox("Has Healthcare Coverage", [0,1], format_func=lambda x:"No" if x==0 else "Yes")

h5,h6,h7,h8 = st.columns(4)
with h5:
    NoDocbcCost = st.selectbox("Could not see doctor due to cost", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    DiffWalk = st.selectbox("Difficulty walking", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
with h6:
    GenHlth = st.selectbox("General Health (1=Excellent,5=Poor)", [1,2,3,4,5], index=2)
    PhysHlth = st.number_input("Physical health bad days (0-30)", min_value=0, max_value=30, value=0)
with h7:
    MentHlth = st.number_input("Mental health bad days (0-30)", min_value=0, max_value=30, value=0)
with h8:
    # placeholder column
    st.write("")

st.markdown("---")

# Predict button area
if st.button("Predict Risk", use_container_width=True):
    # Map UI to BRFSS-coded input dictionary
    ui_vals = {
        "HighBP": int(HighBP),
        "HighChol": int(HighChol),
        "CholCheck": int(CholCheck),
        "Smoker": int(Smoker),
        "Stroke": int(Stroke),
        "HeartDiseaseorAttack": int(HeartDiseaseorAttack),
        "PhysActivity": int(PhysActivity),
        "Fruits": int(Fruits),
        "Veggies": int(Veggies),
        "HvyAlcoholConsump": int(HvyAlcoholConsump),
        "AnyHealthcare": int(AnyHealthcare),
        "NoDocbcCost": int(NoDocbcCost),
        "DiffWalk": int(DiffWalk),
        "BMI": float(BMI),
        "MentHlth": int(MentHlth),
        "PhysHlth": int(PhysHlth),
        "GenHlth": int(GenHlth),
        "Sex": 1 if Sex == "Male" else 0,
        "Age": int(Age),
        "Education": int(Education),
        "Income": int(Income)
    }
    st.subheader("🔍 Model input (after mapping)")
    # Pretty-print JSON for clarity (sorted keys, proper commas)
    try:
        pretty = json.dumps(ui_vals, indent=2, sort_keys=True)
        st.code(pretty, language='json')
    except Exception:
        st.json(ui_vals)
    # Build DataFrame aligned to model cols
    if MODEL_COLS:
        df = pd.DataFrame([ui_vals])
        for c in MODEL_COLS:
            if c not in df.columns:
                df[c] = 0
        df = df[MODEL_COLS]
    else:
        df = pd.DataFrame([ui_vals])

    # Scale if scaler available
    X_for_model = df.copy()
    if SCALER:
        try:
            X_scaled = SCALER.transform(X_for_model)
        except Exception:
            X_scaled = X_for_model.values
    else:
        X_scaled = X_for_model.values

    # Check model
    if MODEL is None:
        st.error(f"Model not found. Place calibrated model at: {MODEL_PATH}")
    else:
        try:
            # predict probability
            ml_prob = float(MODEL.predict_proba(X_scaled)[0][1])
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            raise

        # Prepare SHAP explanation (unscaled df for TreeExplainer)
        shap_png = None
        shap_pairs = []
        if SHAP_AVAILABLE:
            try:
                shap_png = os.path.join(REPORTS_DIR, f"shap_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                shap_pairs, shpimg = compute_shap_instance(MODEL, df, top_n=10, save_path=shap_png)
                if shpimg is None:
                    shap_png = None
            except Exception:
                shap_pairs = []
                shap_png = None

        # Combine ML + medical rules (hierarchical)
        hybrid = combine_ml_and_medical(ml_prob, ui_vals, method="hierarchical")

        # store for pdf & session
        st.session_state['last_result'] = {
            "ml_prob": hybrid['ml_prob'],
            "medical_boost": hybrid['medical_boost'],
            "combined_prob": hybrid['combined_prob'],
            "risk_level": hybrid['risk_level'],
            "override_min": hybrid['override_min'],
            "feature_contrib": [(p['feature'], f"{p['shap_value']:+.4f}") for p in shap_pairs] if shap_pairs else [],
            "shap_png": shap_png
        }
        # Generate and store textual recommendation data for UI and PDF
        try:
            recommendation_data = generate_recommendation(hybrid['combined_prob'])
        except Exception:
            recommendation_data = generate_recommendation(float(hybrid.get('combined_prob', 0.0)))
        st.session_state['recommendation_data'] = recommendation_data
        # UI display
        st.metric("ML Probability", f"{hybrid['ml_prob']*100:.2f}%")
        st.metric("Medical Boost", f"{hybrid['medical_boost']*100:+.1f}%")
        st.metric("Combined Probability", f"{hybrid['combined_prob']*100:.2f}%")
        st.write("Hybrid Risk Level:", f"**{hybrid['risk_level']}**")

        # --- Visualizations: Gauge and Bar chart ---
        fig_dir = REPORTS_DIR
        figs_dir = os.path.join(fig_dir, 'figures')
        os.makedirs(figs_dir, exist_ok=True)

        # Prepare values
        ml_pct = hybrid['ml_prob'] * 100
        boost_pct = hybrid['medical_boost'] * 100
        combined_pct = hybrid['combined_prob'] * 100

        # Gauge for combined probability
        try:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=combined_pct,
                number={'suffix': '%', 'valueformat': '.2f'},
                delta={'reference': ml_pct, 'suffix': '%', 'increasing': {'color': 'red'}, 'decreasing': {'color': 'green'}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#0b6fa4'},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 60], 'color': 'gold'},
                        {'range': [60, 80], 'color': 'orange'},
                        {'range': [80, 100], 'color': 'red'}
                    ]
                }
            ))
            gauge.update_layout(height=300, margin={'t':20,'b':20,'l':20,'r':20}, title={'text':'Combined Risk Probability'})
            st.plotly_chart(gauge, use_container_width=True)
            gauge_path = os.path.join(figs_dir, f"gauge_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            # save static image via write_image if kaleido available, else skip saving
            try:
                gauge.write_image(gauge_path, scale=2)
                st.write(f"Saved gauge: {gauge_path}")
            except Exception:
                pass
        except Exception:
            pass

        # Bar chart for ML prob, Boost, Combined
        try:
            bar_df = pd.DataFrame({
                'Metric': ['ML Probability','Medical Boost','Combined Probability'],
                'Value': [ml_pct, boost_pct, combined_pct]
            })
            bar_fig = px.bar(bar_df, x='Metric', y='Value', color='Metric', text='Value', color_discrete_sequence=['#2a9d8f','#74c69d','#0b6fa4'])
            bar_fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            bar_fig.update_layout(yaxis_range=[0,100], height=320, showlegend=False, title='Probability Breakdown (%)')
            st.plotly_chart(bar_fig, use_container_width=True)
            bar_path = os.path.join(figs_dir, f"bar_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            try:
                bar_fig.write_image(bar_path, scale=2)
                st.write(f"Saved bar chart: {bar_path}")
            except Exception:
                pass
        except Exception:
            pass

        # Explanation text
        notes = []
        if hybrid['override_min'] is not None:
            notes.append("Clinical override applied: major cardiovascular comorbidity with hypertension/dyslipidemia.")
        if hybrid['medical_boost'] > 0:
            notes.append(f"Medical rules increased risk by {hybrid['medical_boost']*100:.1f}%.")
        if hybrid['medical_boost'] < 0:
            notes.append(f"Medical rules decreased risk by {abs(hybrid['medical_boost'])*100:.1f}% due to protective factors.")
        if not notes:
            notes.append("No medical adjustment applied.")
        st.info("\n".join(notes))

        # show shap table and image
        if st.session_state['last_result']['feature_contrib']:
            st.subheader("Top Feature Contributions (SHAP)")
            fc_df = pd.DataFrame(st.session_state['last_result']['feature_contrib'], columns=['Feature','Impact'])
            st.table(fc_df)
            if st.session_state['last_result']['shap_png']:
                st.image(st.session_state['last_result']['shap_png'], caption="SHAP contributions (top features)")

        # Recommendation text (use generated recommendation_data)
        recommendation_data = st.session_state.get('recommendation_data') or generate_recommendation(hybrid['combined_prob'])
        # ensure session copy
        st.session_state['recommendation_data'] = recommendation_data

        st.subheader("💡 Recommendation")
        st.markdown(f"### **Risk Level:** {recommendation_data['risk_label']}")
        st.write(recommendation_data['advice'])

        st.markdown("### **Precautions:**")
        for p in recommendation_data['precautions']:
            st.write(f"- {p}")

# ---------- PDF generation UI ----------
st.markdown("---")
st.header("📥 Download Professional PDF Report")
if st.button("Generate PDF Report", use_container_width=True):
    if 'last_result' not in st.session_state:
        st.error("Run a prediction first.")
    else:
        patient_info = {
            "name": patient_name or "N/A",
            "patient_id": patient_id or f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "age_display": AGE_LABELS.get(Age, "N/A"),
            "gender": Sex,
            "phone": phone or "N/A",
            "referred_by": referred_by or "Self"
        }
        pred = st.session_state['last_result']
        recommendation_data = st.session_state.get('recommendation_data')
        prediction_info = {
            "ml_prob": pred.get('ml_prob', 0.0),
            "medical_boost": pred.get('medical_boost', 0.0),
            "combined_prob": pred.get('combined_prob', 0.0),
            "risk_level": pred.get('risk_level', 'N/A'),
            "recommendation": f"Risk Level: {pred.get('risk_level', 'N/A')}",
            "recommendation_data": recommendation_data
        }
        try:
            if REPORTLAB_AVAILABLE:
                pdf_buffer = generate_pdf_report(patient_info, prediction_info, recommendation_data=recommendation_data)
                filename = f"{patient_info.get('patient_id','PAT')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📄 Download Report (PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name=filename,
                    mime="application/pdf"
                )
                st.success("✅ PDF report generated successfully")
            else:
                st.error("ReportLab not available for PDF generation")
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")

        # Email sending block (requires SMTP credentials in environment variables)
        smtp_host = os.environ.get('SMTP_HOST')
        smtp_port = os.environ.get('SMTP_PORT')
        smtp_user = os.environ.get('SMTP_USER')
        smtp_pass = os.environ.get('SMTP_PASSWORD')
        smtp_from = os.environ.get('SMTP_FROM', smtp_user)
        smtp_use_ssl = os.environ.get('SMTP_USE_SSL', '1')

        def send_pdf_via_email(pdf_filepath, to_email, subject=None, body=None):
            """Send PDF via SendGrid (if SENDGRID_API_KEY set) else fallback to SMTP.
            Raises on failure.
            """
            if not to_email:
                raise ValueError('Recipient email is empty')

            subject = subject or f"Your Diabetes Risk Report - {patient_info.get('patient_id','PAT')}"
            body = body or "Please find attached your diabetes risk assessment report (PDF). This is an AI-assisted screening result."

            # Try SendGrid if API key present
            sendgrid_key = os.environ.get('SENDGRID_API_KEY')
            if sendgrid_key:
                try:
                    with open(pdf_filepath, 'rb') as f:
                        pdf_bytes = f.read()
                    payload = {
                        "personalizations": [{"to": [{"email": to_email}]}],
                        "from": {"email": smtp_from or os.environ.get('SENDGRID_FROM', smtp_user)},
                        "subject": subject,
                        "content": [{"type": "text/plain", "value": body}],
                        "attachments": [{
                            "content": base64.b64encode(pdf_bytes).decode('utf-8'),
                            "filename": os.path.basename(pdf_filepath),
                            "type": "application/pdf",
                            "disposition": "attachment"
                        }]
                    }
                    headers = {
                        'Authorization': f'Bearer {sendgrid_key}',
                        'Content-Type': 'application/json'
                    }
                    resp = requests.post('https://api.sendgrid.com/v3/mail/send', json=payload, headers=headers, timeout=30)
                    if resp.status_code not in (200, 202):
                        raise RuntimeError(f"SendGrid send failed: {resp.status_code} {resp.text}")
                    return True
                except Exception as se:
                    # fall through to SMTP fallback if available
                    st.warning(f"SendGrid send failed, will try SMTP fallback: {se}")

            # SMTP fallback
            if not smtp_host or not smtp_port or not smtp_user or not smtp_pass:
                raise RuntimeError('No email delivery method available: set SENDGRID_API_KEY or SMTP_* environment variables.')

            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = smtp_from or smtp_user
            msg['To'] = to_email
            msg.set_content(body)
            with open(pdf_filepath, 'rb') as f:
                pdf_data = f.read()
            msg.add_attachment(pdf_data, maintype='application', subtype='pdf', filename=os.path.basename(pdf_filepath))

            port = int(smtp_port)
            use_ssl = str(smtp_use_ssl).lower() in ('1', 'true', 'yes')
            if use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(smtp_host, port, context=context) as server:
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(smtp_host, port) as server:
                    server.ehlo()
                    server.starttls(context=ssl.create_default_context())
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)

        # Note: Email report sending can be added in the test email section below

    # Small test email UI so users can validate SMTP/SendGrid settings without generating a PDF
st.markdown("---")
st.header("✉️ Test Email Settings")
st.write("Send a short test email to validate your SMTP or SendGrid configuration. No PDF required.")
test_recipient = st.text_input("Test recipient email (for configuration test)", value=patient_email or "")
if st.button("Send test email"):
    try:
        # reuse send logic: try SendGrid then SMTP
        send_ok = False
        sendgrid_key = os.environ.get('SENDGRID_API_KEY')
        test_subject = "Assist Diabetes AI - Test Email"
        test_body = "This is a test email from Assist Diabetes AI to validate email settings."
        if sendgrid_key:
            # send via SendGrid
            payload = {
                "personalizations": [{"to": [{"email": test_recipient}]}],
                "from": {"email": os.environ.get('SENDGRID_FROM', smtp_from or smtp_user)},
                "subject": test_subject,
                "content": [{"type": "text/plain", "value": test_body}]
            }
            headers = {'Authorization': f'Bearer {sendgrid_key}', 'Content-Type': 'application/json'}
            resp = requests.post('https://api.sendgrid.com/v3/mail/send', json=payload, headers=headers, timeout=30)
            if resp.status_code in (200,202):
                st.success(f"Test email sent via SendGrid to {test_recipient}")
                send_ok = True
            else:
                st.warning(f"SendGrid test failed: {resp.status_code} {resp.text}")

        if not send_ok:
            # try SMTP fallback
            try:
                msg = EmailMessage()
                msg['Subject'] = test_subject
                msg['From'] = smtp_from or smtp_user
                msg['To'] = test_recipient
                msg.set_content(test_body)
                port = int(os.environ.get('SMTP_PORT', smtp_port or 0))
                use_ssl = str(os.environ.get('SMTP_USE_SSL', smtp_use_ssl)).lower() in ('1', 'true', 'yes')
                if use_ssl:
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(smtp_host, port, context=context) as server:
                        server.login(smtp_user, smtp_pass)
                        server.send_message(msg)
                else:
                    with smtplib.SMTP(smtp_host, port) as server:
                        server.ehlo()
                        server.starttls(context=ssl.create_default_context())
                        server.login(smtp_user, smtp_pass)
                        server.send_message(msg)
                st.success(f"Test email sent via SMTP to {test_recipient}")
            except Exception as e:
                st.error("Test email failed: " + str(e))

    except Exception as outer_e:
        st.error("Test email failed: " + str(outer_e))

st.markdown("---")
st.caption("This is a screening tool — not a diagnostic test. For diagnosis, perform laboratory tests (HbA1c/FPG).")
st.caption(f"Sample visual reference (uploaded): {SAMPLE_PDF_PATH}")
