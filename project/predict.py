"""
Clean prediction module - NO SMOTE
Loads calibrated model and makes predictions with proper thresholding.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEFAULT_THRESHOLD = 0.30
MODEL_DIR = 'models'

# Global model and scaler (loaded once)
_model = None
_scaler = None
_feature_columns = None
_metadata = None

def load_model_artifacts(model_dir=MODEL_DIR):
    """
    Load model and all artifacts.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model artifacts
    
    Returns:
    --------
    dict : Loaded artifacts
    """
    global _model, _scaler, _feature_columns, _metadata
    
    if _model is not None:
        return {
            'model': _model,
            'scaler': _scaler,
            'feature_columns': _feature_columns,
            'metadata': _metadata
        }
    
    # Load model
    model_path = os.path.join(model_dir, 'diabetes_model_calibrated.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    _model = joblib.load(model_path)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    _scaler = joblib.load(scaler_path)
    
    # Load feature columns
    columns_path = os.path.join(model_dir, 'model_columns.pkl')
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"Feature columns not found: {columns_path}")
    _feature_columns = joblib.load(columns_path)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    if os.path.exists(metadata_path):
        _metadata = joblib.load(metadata_path)
    else:
        _metadata = {'threshold': DEFAULT_THRESHOLD}
    
    return {
        'model': _model,
        'scaler': _scaler,
        'feature_columns': _feature_columns,
        'metadata': _metadata
    }

def preprocess_input(patient_data, scaler, feature_columns):
    """
    Preprocess patient data for prediction.
    
    Parameters:
    -----------
    patient_data : dict or pandas.DataFrame
        Patient health indicators
    scaler : StandardScaler
        Fitted scaler
    feature_columns : list
        Expected feature column names
    
    Returns:
    --------
    pandas.DataFrame : Preprocessed data ready for prediction
    """
    # Convert to DataFrame
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    else:
        raise TypeError("patient_data must be dict or pandas.DataFrame")
    
    # Check for missing features
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Reorder columns to match training
    df = df[feature_columns]
    
    # Scale continuous features
    continuous_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
    df_scaled = df.copy()
    df_scaled[continuous_features] = scaler.transform(df[continuous_features])
    
    return df_scaled

def stratify_risk(probability):
    """
    Convert probability to risk level and recommendation.
    
    Parameters:
    -----------
    probability : float
        Probability of being at-risk (0.0 to 1.0)
    
    Returns:
    --------
    tuple : (risk_level, recommendation, color, icon)
    """
    if probability < 0.20:
        return (
            "Low Risk",
            "Maintain a healthy lifestyle. No immediate medical intervention required. Continue regular check-ups.",
            "green",
            "✅"
        )
    elif probability < 0.40:
        return (
            "Moderate Risk",
            "Consider lifestyle modifications (diet, exercise) and schedule follow-up glucose testing within 6 months. Monitor your health indicators regularly.",
            "yellow",
            "⚠️"
        )
    elif probability < 0.70:
        return (
            "High Risk",
            "You may be developing insulin resistance. Please take a diabetes screening test (HbA1c or Fasting Plasma Glucose test). Consult with your healthcare provider for personalized advice.",
            "orange",
            "🔶"
        )
    else:
        return (
            "Very High Risk",
            "High likelihood of diabetes. Medical testing is strongly advised immediately. Please schedule an appointment with your healthcare provider for comprehensive diabetes screening and evaluation.",
            "red",
            "🔴"
        )

def predict_risk(patient_data, model_dir=MODEL_DIR, threshold=None, return_explanation=False):
    """
    Predict diabetes risk for a patient.
    
    Parameters:
    -----------
    patient_data : dict or pandas.DataFrame
        Patient health indicators
    model_dir : str
        Directory containing model artifacts
    threshold : float, optional
        Decision threshold (defaults to model's threshold)
    return_explanation : bool
        If True, also return SHAP explanation
    
    Returns:
    --------
    dict : Prediction results
    """
    # Load artifacts
    artifacts = load_model_artifacts(model_dir)
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']
    metadata = artifacts['metadata']
    
    # Use threshold from metadata or default
    if threshold is None:
        threshold = metadata.get('threshold', DEFAULT_THRESHOLD)
    
    # Preprocess input
    X_processed = preprocess_input(patient_data, scaler, feature_columns)
    
    # Predict probability (calibrated)
    proba = model.predict_proba(X_processed)[0, 1]
    
    # Ensure valid range
    proba = max(0.0, min(1.0, proba))
    
    # Binary prediction
    predicted_class = 1 if proba >= threshold else 0
    
    # Risk stratification
    risk_level, recommendation, risk_color, risk_icon = stratify_risk(proba)
    
    # Confidence level
    if proba < 0.30:
        confidence = "Low"
    elif proba < 0.50:
        confidence = "Moderate"
    elif proba < 0.70:
        confidence = "High"
    else:
        confidence = "Very High"
    
    # Build result
    result = {
        'probability': float(proba),
        'prediction': 'At-Risk' if predicted_class == 1 else 'Healthy',
        'risk_level': risk_level,
        'recommendation': recommendation,
        'confidence': confidence,
        'risk_color': risk_color,
        'risk_icon': risk_icon,
        'threshold_used': threshold
    }
    
    # Add SHAP explanation if requested
    if return_explanation:
        try:
            import shap
            # Get base estimator from CalibratedClassifierCV
            base_model = model.base_estimator if hasattr(model, 'base_estimator') else model
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_processed)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary, get positive class
            
            # Handle array shape
            if len(shap_values.shape) > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # Get top contributing features
            feature_contributions = list(zip(feature_columns, shap_vals))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            result['explanation'] = {
                'top_features': [
                    {
                        'feature': feat,
                        'shap_value': float(val),
                        'direction': 'increases' if val > 0 else 'reduces'
                    }
                    for feat, val in feature_contributions[:10]
                ]
            }
        except Exception as e:
            result['explanation'] = None
            result['explanation_error'] = str(e)
    
    return result

def validate_all_no_input(model_dir=MODEL_DIR):
    """
    Validate that all NO inputs give LOW risk.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model artifacts
    
    Returns:
    --------
    dict : Validation results
    """
    # Create all NO patient
    all_no_patient = {
        'HighBP': 0,
        'HighChol': 0,
        'CholCheck': 0,
        'BMI': 22.0,
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'Veggies': 1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 2,
        'MentHlth': 0,
        'PhysHlth': 0,
        'DiffWalk': 0,
        'Sex': 0,
        'Age': 5,
        'Education': 5,
        'Income': 6
    }
    
    result = predict_risk(all_no_patient, model_dir)
    
    print("=" * 80)
    print("VALIDATION: All NO Inputs Test")
    print("=" * 80)
    print(f"Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Expected: < 0.05 (< 5%), Low Risk")
    
    if result['probability'] < 0.05 and result['risk_level'] == "Low Risk":
        print("✅ PASS: Model correctly predicts LOW risk for healthy patient")
        return {'passed': True, 'result': result}
    else:
        print("❌ FAIL: Model predicts incorrect risk level")
        return {'passed': False, 'result': result}

if __name__ == "__main__":
    # Test prediction
    test_patient = {
        'HighBP': 1,
        'HighChol': 1,
        'CholCheck': 1,
        'BMI': 30.5,
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'Veggies': 1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 3,
        'MentHlth': 0,
        'PhysHlth': 0,
        'DiffWalk': 0,
        'Sex': 1,
        'Age': 7,
        'Education': 4,
        'Income': 6
    }
    
    result = predict_risk(test_patient)
    print("\nTest Prediction:")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Probability: {result['probability']:.2%}")
    print(f"  Recommendation: {result['recommendation']}")
    
    # Validate all NO input
    validate_all_no_input()

