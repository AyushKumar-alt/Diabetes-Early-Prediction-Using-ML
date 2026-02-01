"""
Core prediction module for diabetes risk assessment.
Combines preprocessing, model loading, and risk stratification.
"""

import os
import xgboost as xgb
import numpy as np
from utils.preprocess import preprocess_input, validate_input
from utils.risk_stratification import stratify_risk, get_risk_color, get_risk_icon

# Optimal threshold determined from model optimization
OPTIMAL_THRESHOLD = 0.30

# Global model instance (loaded once)
_model = None

def load_model(model_path=None):
    """
    Load the trained XGBoost model.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to model file. Defaults to models/bst_binary.json
    """
    global _model
    
    if _model is not None:
        return _model
    
    if model_path is None:
        # Get the directory of this file (core/predict.py)
        current_file = os.path.abspath(__file__)
        # Get core/ directory
        core_dir = os.path.dirname(current_file)
        # Go up one level to get project root (diabetes_risk_app/)
        project_root = os.path.dirname(core_dir)
        model_path = os.path.join(project_root, 'models', 'bst_binary.json')
        
        # If model not found, try alternative paths (for different working directories)
        if not os.path.exists(model_path):
            # Try 1: Relative to current working directory
            alt_paths = [
                os.path.join('diabetes_risk_app', 'models', 'bst_binary.json'),
                os.path.join('models', 'bst_binary.json'),
                os.path.join(os.getcwd(), 'diabetes_risk_app', 'models', 'bst_binary.json'),
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
    
    if not os.path.exists(model_path):
        # Provide helpful error message with all attempted paths
        attempted_paths = [os.path.join(project_root, 'models', 'bst_binary.json')]
        attempted_paths.extend([
            os.path.join('diabetes_risk_app', 'models', 'bst_binary.json'),
            os.path.join('models', 'bst_binary.json'),
            os.path.join(os.getcwd(), 'diabetes_risk_app', 'models', 'bst_binary.json'),
        ])
        error_msg = f"Model file not found: {model_path}\n"
        error_msg += f"Attempted paths:\n" + "\n".join(f"  - {p}" for p in attempted_paths)
        error_msg += f"\nCurrent working directory: {os.getcwd()}"
        error_msg += f"\nPlease ensure the model file exists at one of these locations."
        raise FileNotFoundError(error_msg)
    
    _model = xgb.Booster()
    _model.load_model(model_path)
    
    return _model

def predict_risk(patient_data, model_path=None, return_explanation=False):
    """
    Predict diabetes risk for a patient.
    
    Parameters:
    -----------
    patient_data : dict, pandas.DataFrame, or list
        Patient health indicators
    model_path : str, optional
        Path to model file (if not using default)
    return_explanation : bool
        If True, also return feature importance explanation
    
    Returns:
    --------
    dict : Prediction results
        - probability: float (0.0 to 1.0)
        - prediction: str ("Healthy" or "At-Risk")
        - risk_level: str
        - recommendation: str
        - confidence: str ("High" or "Moderate")
        - explanation: dict (optional, if return_explanation=True)
    """
    # Validate input
    validation = validate_input(patient_data)
    if not validation['valid']:
        raise ValueError(f"Invalid input: {validation['errors']}")
    
    # Load model if not already loaded
    model = load_model(model_path)
    
    # Preprocess input
    dmatrix = preprocess_input(patient_data)
    
    # Predict probability of "At-Risk"
    proba = model.predict(dmatrix)
    
    # Handle single prediction
    if isinstance(proba, np.ndarray):
        prob = float(proba[0])
    else:
        prob = float(proba)
    
    # Ensure probability is in valid range
    prob = max(0.0, min(1.0, prob))
    
    # Binary prediction using optimized threshold
    predicted_class = 1 if prob >= OPTIMAL_THRESHOLD else 0
    
    # Risk level + recommendation
    risk_level, recommendation = stratify_risk(prob)
    
    # Determine confidence based on probability range
    if prob < 0.30:
        confidence = "Low"
    elif prob < 0.50:
        confidence = "Moderate"
    elif prob < 0.70:
        confidence = "High"
    else:
        confidence = "Very High"
    
    # Build result dictionary
    result = {
        "probability": prob,
        "prediction": "At-Risk" if predicted_class == 1 else "Healthy",
        "risk_level": risk_level,
        "recommendation": recommendation,
        "confidence": confidence,
        "risk_color": get_risk_color(risk_level),
        "risk_icon": get_risk_icon(risk_level),
        "threshold_used": OPTIMAL_THRESHOLD
    }
    
    # Add explanation if requested
    if return_explanation:
        try:
            from utils.explainability import explain_prediction
            explanation = explain_prediction(model, dmatrix)
            result["explanation"] = explanation
        except Exception as e:
            result["explanation"] = None
            result["explanation_error"] = str(e)
    
    # Add warnings if any
    if validation['warnings']:
        result["warnings"] = validation['warnings']
    
    return result

def predict_batch(patient_data_list, model_path=None):
    """
    Predict risk for multiple patients.
    
    Parameters:
    -----------
    patient_data_list : list
        List of patient data (dict, DataFrame, or list)
    model_path : str, optional
        Path to model file
    
    Returns:
    --------
    list : List of prediction results
    """
    results = []
    for patient_data in patient_data_list:
        try:
            result = predict_risk(patient_data, model_path)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "patient_data": patient_data
            })
    
    return results

