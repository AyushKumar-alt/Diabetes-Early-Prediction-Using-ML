"""
Model calibration and reliability utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import xgboost as xgb

def calculate_brier_score(y_true, y_proba):
    """
    Calculate Brier Score for probability calibration.
    Lower is better (0 = perfect, 1 = worst).
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    
    Returns:
    --------
    float : Brier Score
    """
    return brier_score_loss(y_true, y_proba)

def plot_calibration_curve(y_true, y_proba, n_bins=10, save_path=None):
    """
    Plot calibration curve to assess probability calibration.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob_pred, prob_true, marker='o', label="Model calibration", linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated", color='gray', linewidth=2)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def prediction_reliability(model, X, y, runs=20):
    """
    Test prediction reliability by running multiple predictions.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained model
    X : array-like or xgb.DMatrix
        Test data
    y : array-like
        True labels (not used, but kept for consistency)
    runs : int
        Number of prediction runs
    
    Returns:
    --------
    dict : Reliability metrics
        - std_dev: Standard deviation of predictions
        - variance: Variance of predictions
        - reliability_level: Reliability category
    """
    if not isinstance(X, xgb.DMatrix):
        X = xgb.DMatrix(X)
    
    preds = []
    for i in range(runs):
        proba = model.predict(X)
        if isinstance(proba, np.ndarray) and len(proba.shape) > 1:
            proba = proba.flatten()
        preds.append(proba)
    
    preds = np.array(preds)
    
    # Calculate statistics
    mean_preds = np.mean(preds, axis=0)
    std_preds = np.std(preds, axis=0)
    var_preds = np.var(preds, axis=0)
    
    avg_std = np.mean(std_preds)
    avg_var = np.mean(var_preds)
    
    # Determine reliability level
    if avg_std < 0.02:
        reliability_level = "Very Reliable"
    elif avg_std < 0.05:
        reliability_level = "Reliable"
    elif avg_std < 0.10:
        reliability_level = "Slightly Unstable"
    else:
        reliability_level = "Unstable (Retrain Needed)"
    
    return {
        'std_dev': avg_std,
        'variance': avg_var,
        'reliability_level': reliability_level,
        'mean_predictions': mean_preds,
        'std_predictions': std_preds
    }

