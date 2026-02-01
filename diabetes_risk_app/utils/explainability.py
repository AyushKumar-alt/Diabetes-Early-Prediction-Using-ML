"""
Explainability utilities using SHAP values.
Provides feature importance and individual prediction explanations.
"""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Global explainer (will be initialized on first use)
_explainer = None
_model = None

def initialize_explainer(model, background_data=None, max_background=100):
    """
    Initialize SHAP explainer for the model.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    background_data : xgb.DMatrix or array-like, optional
        Background dataset for SHAP. If None, uses a sample from training data.
    max_background : int
        Maximum number of samples to use for background
    """
    global _explainer, _model
    _model = model
    
    # Create TreeExplainer (fast for XGBoost)
    _explainer = shap.TreeExplainer(model)
    return _explainer

def get_feature_importance(model, importance_type='gain', top_n=10):
    """
    Get feature importance from XGBoost model.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained model
    importance_type : str
        Type of importance ('gain', 'weight', 'cover')
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    dict : {feature_name: importance_score}
    """
    importance_dict = model.get_score(importance_type=importance_type)
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N
    return dict(sorted_features[:top_n])

def explain_prediction(model, patient_data, explainer=None, top_n=10):
    """
    Explain a single prediction using SHAP values.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained model
    patient_data : xgb.DMatrix or array-like
        Patient data for prediction
    explainer : shap.TreeExplainer, optional
        Pre-initialized explainer. If None, creates new one.
    top_n : int
        Number of top contributing features to return
    
    Returns:
    --------
    dict : Explanation results
        - shap_values: SHAP values for each feature
        - base_value: Base prediction value
        - top_features: Top N features contributing to prediction
    """
    global _explainer
    
    # Initialize explainer if needed
    if explainer is None:
        if _explainer is None:
            _explainer = initialize_explainer(model)
        explainer = _explainer
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(patient_data)
    
    # Get base value (expected value)
    base_value = explainer.expected_value
    
    # Convert to array if needed
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Get feature names
    feature_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    # Create feature importance dataframe
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values
    })
    
    # Sort by absolute SHAP value
    feature_contributions['abs_shap'] = feature_contributions['shap_value'].abs()
    feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)
    
    # Get top N features
    top_features = feature_contributions.head(top_n).to_dict('records')
    
    return {
        'shap_values': shap_values,
        'base_value': base_value,
        'top_features': top_features,
        'all_features': feature_contributions.to_dict('records')
    }

def get_feature_descriptions():
    """
    Get human-readable descriptions of features.
    
    Returns:
    --------
    dict : {feature_name: description}
    """
    return {
        'HighBP': 'High Blood Pressure',
        'HighChol': 'High Cholesterol',
        'CholCheck': 'Cholesterol Check in Past 5 Years',
        'BMI': 'Body Mass Index',
        'Smoker': 'Smoking Status',
        'Stroke': 'History of Stroke',
        'HeartDiseaseorAttack': 'History of Heart Disease or Attack',
        'PhysActivity': 'Physical Activity in Past 30 Days',
        'Fruits': 'Consumes Fruit 1+ Times Per Day',
        'Veggies': 'Consumes Vegetables 1+ Times Per Day',
        'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
        'AnyHealthcare': 'Has Any Healthcare Coverage',
        'NoDocbcCost': 'Could Not See Doctor Due to Cost',
        'GenHlth': 'General Health (1=Excellent, 5=Poor)',
        'MentHlth': 'Mental Health (Days in Past 30)',
        'PhysHlth': 'Physical Health (Days in Past 30)',
        'DiffWalk': 'Difficulty Walking or Climbing Stairs',
        'Sex': 'Gender (0=Female, 1=Male)',
        'Age': 'Age Group (1-13, BRFSS encoding)',
        'Education': 'Education Level (1-6)',
        'Income': 'Income Level (1-8)'
    }

def shap_bar_chart(model, patient_data, top_n=10, save_path=None):
    """
    Generate SHAP bar chart for feature contributions.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained model
    patient_data : xgb.DMatrix or pandas.DataFrame
        Patient data
    top_n : int
        Number of top features to show
    save_path : str, optional
        Path to save the chart image
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    
    # Initialize explainer
    explainer = initialize_explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(patient_data)
    
    # Convert to array if needed
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Get feature names
    if isinstance(patient_data, pd.DataFrame):
        feature_names = patient_data.columns.tolist()
        shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    else:
        feature_names = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Create sorted list of (feature, shap_value) tuples
    shap_dict = sorted(
        list(zip(feature_names, shap_vals)),
        key=lambda x: abs(x[1]), reverse=True
    )
    
    # Get top N
    features, values = zip(*shap_dict[:top_n])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ['#d32f2f' if v > 0 else '#388e3c' for v in values]  # Red for positive, green for negative
    ax.barh(features, values, color=bar_colors, alpha=0.7)
    ax.set_xlabel('SHAP Value (Contribution to Risk)', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Contributions to This Prediction', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def get_patient_shap_explanation(model, patient_data, top_n=10):
    """
    Get SHAP explanation for a single patient in text format.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained model
    patient_data : xgb.DMatrix or pandas.DataFrame
        Patient data
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    list : List of dicts with feature, shap_value, direction, effect
    """
    explainer = initialize_explainer(model)
    shap_values = explainer.shap_values(patient_data)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    if isinstance(patient_data, pd.DataFrame):
        feature_names = patient_data.columns.tolist()
        shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    else:
        feature_names = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Create list of explanations
    explanations = []
    for feat, val in zip(feature_names, shap_vals):
        explanations.append({
            'feature': feat,
            'shap_value': val,
            'direction': 'increases' if val > 0 else 'reduces',
            'effect': 'risk' if val > 0 else 'risk',
            'abs_value': abs(val)
        })
    
    # Sort by absolute value and return top N
    explanations.sort(key=lambda x: x['abs_value'], reverse=True)
    return explanations[:top_n]


def calibration_and_brier(y_true, y_proba, n_bins=10):
    """
    Compute calibration curve and Brier score.

    Returns:
        (prob_true, prob_pred, brier)
    """
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
    except Exception as e:
        raise RuntimeError("scikit-learn (calibration) is required")

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_proba)
    return prob_true, prob_pred, brier


def prediction_reliability(model, X, y=None, runs=20):
    """
    Run the model multiple times to measure variability in predicted probabilities.

    Returns: numpy array of shape (runs, n_samples) of probabilities and std dev across runs.
    """
    import xgboost as xgb
    preds = []
    for i in range(runs):
        proba = model.predict(xgb.DMatrix(X))
        preds.append(proba)
    preds = np.array(preds)
    avg_var = preds.var()
    avg_std = preds.std()
    return preds, avg_var, avg_std


