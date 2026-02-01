"""
Preprocessing utilities for diabetes risk prediction.
Handles feature scaling and DMatrix creation for XGBoost.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Feature order expected by the model
FEATURE_ORDER = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# Continuous features that need scaling
CONTINUOUS_FEATURES = ['BMI', 'MentHlth', 'PhysHlth', 'Age']

# Scaler instance (will be loaded or created)
_scaler = None

def load_scaler(scaler_path=None):
    """Load a saved scaler if available, otherwise create a default one."""
    global _scaler
    
    if scaler_path and os.path.exists(scaler_path):
        _scaler = joblib.load(scaler_path)
        return _scaler
    
    # If no scaler file exists, create one with training data statistics
    # These are approximate values from the dataset (can be refined)
    # For production, you should save the scaler from training
    _scaler = StandardScaler()
    
    # Approximate means and stds from the dataset (update with actual training stats)
    # These are placeholder values - in production, load from saved scaler
    _scaler.mean_ = np.array([28.38, 3.1, 4.0, 7.0])  # BMI, MentHlth, PhysHlth, Age
    _scaler.scale_ = np.array([6.61, 7.2, 8.5, 1.7])  # Approximate stds
    
    return _scaler

def preprocess_input(patient_data, scaler_path=None):
    """
    Preprocess patient data for prediction.
    
    Parameters:
    -----------
    patient_data : dict, pandas.DataFrame, or list
        Patient health indicators. Can be:
        - dict with feature names as keys
        - DataFrame with feature columns
        - list of values in FEATURE_ORDER
    
    scaler_path : str, optional
        Path to saved scaler file
    
    Returns:
    --------
    xgb.DMatrix
        Preprocessed data ready for XGBoost prediction
    """
    global _scaler
    
    # Load scaler if not already loaded
    if _scaler is None:
        load_scaler(scaler_path)
    
    # Convert input to DataFrame
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, list):
        # Assume list is in FEATURE_ORDER
        if len(patient_data) == len(FEATURE_ORDER):
            df = pd.DataFrame([patient_data], columns=FEATURE_ORDER)
        else:
            raise ValueError(f"Expected {len(FEATURE_ORDER)} features, got {len(patient_data)}")
    elif isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    else:
        raise TypeError("patient_data must be dict, list, or pandas.DataFrame")
    
    # Ensure all required features are present
    missing_features = set(FEATURE_ORDER) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Reorder columns to match FEATURE_ORDER
    df = df[FEATURE_ORDER]
    
    # Scale continuous features
    df_scaled = df.copy()
    if _scaler is not None:
        df_scaled[CONTINUOUS_FEATURES] = _scaler.transform(df[CONTINUOUS_FEATURES])
    
    # Convert to DMatrix
    dmatrix = xgb.DMatrix(df_scaled)
    
    return dmatrix

def validate_input(patient_data):
    """
    Validate patient data before preprocessing.
    
    Returns:
    --------
    dict with validation results
    """
    errors = []
    warnings = []
    
    if isinstance(patient_data, dict):
        # Check for missing features
        missing = set(FEATURE_ORDER) - set(patient_data.keys())
        if missing:
            errors.append(f"Missing features: {missing}")
        
        # Check for invalid values
        if 'BMI' in patient_data:
            if patient_data['BMI'] < 10 or patient_data['BMI'] > 100:
                warnings.append("BMI value seems unusual (expected 10-100)")
        
        if 'Age' in patient_data:
            if patient_data['Age'] < 1 or patient_data['Age'] > 13:
                warnings.append("Age value seems unusual (expected 1-13, BRFSS encoding)")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

