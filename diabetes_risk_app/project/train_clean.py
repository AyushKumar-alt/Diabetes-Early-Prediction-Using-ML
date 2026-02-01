"""
Clean model training script - NO SMOTE
Uses scale_pos_weight for class imbalance handling
Includes probability calibration with CalibratedClassifierCV
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, brier_score_loss
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
OPTIMAL_THRESHOLD = 0.30

# Feature columns (must match exactly)
FEATURE_COLUMNS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# Continuous features that need scaling
CONTINUOUS_FEATURES = ['BMI', 'MentHlth', 'PhysHlth', 'Age']


# === Clinically-safe training helpers (NO SMOTE) ===
def train_xgb_multiclass_safe(X_train, y_train, X_val, y_val):
    """
    Train a multiclass XGBoost using sample weights (no SMOTE).
    Returns a fitted XGBClassifier.
    """
    # Compute balanced class weights (inverse frequency)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(np.unique(y_train)),
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


def train_xgb_binary_safe(X_train, y_train, X_val, y_val):
    """
    Train a binary XGBoost model (At-Risk vs Healthy) using sample weights.
    Returns a fitted XGBClassifier.
    """
    # Binary target: 1 = Pre-Diabetes or Diabetes, 0 = No Diabetes
    y_train_bin = (y_train >= 1).astype(int) if y_train.nunique() > 2 else y_train
    y_val_bin = (y_val >= 1).astype(int) if y_val.nunique() > 2 else y_val

    # Class weights: give more weight to the minority "At-Risk" class
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_bin)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        eval_metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0
    )

    model.fit(
        X_train, y_train_bin,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_bin)],
        verbose=False
    )
    return model


def load_and_prepare_data(data_path):
    """
    Load and prepare data for training.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file
    
    Returns:
    --------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Binary target (0=Healthy, 1=At-Risk)
    """
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    
    # Create binary target: 0=Healthy, 1=At-Risk (Pre-DM + DM)
    df['target'] = (df['Diabetes_012'] > 0).astype(int)
    
    # Extract features and target
    X = df[FEATURE_COLUMNS].copy()
    y = df['target'].copy()
    
    print(f"\nClass distribution:")
    print(y.value_counts())
    print(f"\nClass proportions:")
    print(y.value_counts(normalize=True))
    
    return X, y

def scale_features(X_train, X_test, X_val=None):
    """
    Scale continuous features.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Test features
    X_val : pandas.DataFrame, optional
        Validation features
    
    Returns:
    --------
    scaler : StandardScaler
        Fitted scaler
    X_train_scaled : pandas.DataFrame
    X_test_scaled : pandas.DataFrame
    X_val_scaled : pandas.DataFrame (if provided)
    """
    print("\n" + "=" * 80)
    print("SCALING CONTINUOUS FEATURES")
    print("=" * 80)
    
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = X_train.copy()
    X_train_scaled[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
    
    # Transform test data
    X_test_scaled = X_test.copy()
    X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])
    
    # Transform validation data if provided
    if X_val is not None:
        X_val_scaled = X_val.copy()
        X_val_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_val[CONTINUOUS_FEATURES])
        return scaler, X_train_scaled, X_test_scaled, X_val_scaled
    
    return scaler, X_train_scaled, X_test_scaled

def calculate_scale_pos_weight(y_train):
    """
    Calculate scale_pos_weight for XGBoost to handle class imbalance.
    
    Parameters:
    -----------
    y_train : pandas.Series or numpy.ndarray
        Training labels
    
    Returns:
    --------
    float : scale_pos_weight value
    """
    healthy_count = (y_train == 0).sum()
    atrisk_count = (y_train == 1).sum()
    
    scale_pos_weight = healthy_count / atrisk_count
    
    print(f"\nClass counts:")
    print(f"  Healthy (0): {healthy_count:,}")
    print(f"  At-Risk (1): {atrisk_count:,}")
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
    
    return scale_pos_weight

def train_model(X_train, y_train, X_val, y_val, scale_pos_weight):
    """
    Train XGBoost model with calibration.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation labels
    scale_pos_weight : float
        Weight for positive class
    
    Returns:
    --------
    calibrated_model : CalibratedClassifierCV
        Calibrated XGBoost model
    base_model : xgb.XGBClassifier
        Base XGBoost model
    """
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL (NO SMOTE)")
    print("=" * 80)
    
    # Base XGBoost model - train with early stopping first
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    # Train with early stopping
    print("\nTraining base model with early stopping...")
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Best iteration: {base_model.best_iteration}")
    print(f"Best score: {base_model.best_score:.4f}")
    
    # Create a new model with the optimal number of trees for calibration
    # CalibratedClassifierCV doesn't support early_stopping_rounds
    optimal_n_estimators = base_model.best_iteration + 1 if base_model.best_iteration else 500
    
    calibrated_base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        n_estimators=optimal_n_estimators,  # Use optimal number from early stopping
        random_state=RANDOM_STATE,
        n_jobs=-1
        # No early_stopping_rounds for CalibratedClassifierCV
    )
    
    # Train the calibrated base model
    print(f"\nTraining calibrated base model with {optimal_n_estimators} trees...")
    calibrated_base_model.fit(X_train, y_train)
    
    # Calibrate probabilities
    print("\nCalibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(
        calibrated_base_model,
        method='isotonic',  # or 'sigmoid'
        cv=3,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train, y_train)
    
    print("✅ Model training and calibration complete!")
    
    return calibrated_model, base_model

def evaluate_model(model, X_test, y_test, threshold=0.30):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : CalibratedClassifierCV or XGBClassifier
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test labels
    threshold : float
        Decision threshold
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Get calibrated probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Binary predictions at threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)
    
    print(f"\nThreshold: {threshold}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Brier Score: {brier:.4f} (lower is better)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'At-Risk'], digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Healthy  At-Risk")
    print(f"Actual Healthy  {tn:>7}  {fp:>7}")
    print(f"       At-Risk  {fn:>7}  {tp:>7}")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nSensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'brier': brier,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'y_proba': y_proba,
        'y_pred': y_pred
    }

def test_all_no_input(model, scaler, threshold=0.30):
    """
    Test that all "NO" inputs give LOW risk (< 5%).
    
    Parameters:
    -----------
    model : CalibratedClassifierCV
        Trained model
    scaler : StandardScaler
        Fitted scaler
    threshold : float
        Decision threshold
    
    Returns:
    --------
    dict : Test results
    """
    print("\n" + "=" * 80)
    print("TESTING: All NO Inputs → LOW Risk")
    print("=" * 80)
    
    # Create "all NO" patient (healthy baseline)
    all_no_patient = pd.DataFrame([{
        'HighBP': 0,
        'HighChol': 0,
        'CholCheck': 0,
        'BMI': 22.0,  # Normal BMI
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,  # Yes to exercise (positive)
        'Fruits': 1,  # Yes to fruits (positive)
        'Veggies': 1,  # Yes to veggies (positive)
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,  # Has healthcare (positive)
        'NoDocbcCost': 0,
        'GenHlth': 2,  # Very Good health
        'MentHlth': 0,
        'PhysHlth': 0,
        'DiffWalk': 0,
        'Sex': 0,  # Female
        'Age': 5,  # Middle age
        'Education': 5,  # Some college
        'Income': 6  # Good income
    }])
    
    # Ensure correct column order
    all_no_patient = all_no_patient[FEATURE_COLUMNS]
    
    # Scale continuous features
    all_no_patient_scaled = all_no_patient.copy()
    all_no_patient_scaled[CONTINUOUS_FEATURES] = scaler.transform(all_no_patient[CONTINUOUS_FEATURES])
    
    # Predict
    proba = model.predict_proba(all_no_patient_scaled)[0, 1]
    
    print(f"\nAll NO Inputs Test:")
    print(f"  Probability: {proba:.4f} ({proba*100:.2f}%)")
    print(f"  Expected: < 0.05 (< 5%)")
    
    if proba < 0.05:
        print(f"  ✅ PASS: Low risk as expected")
        status = "PASS"
    else:
        print(f"  ❌ FAIL: Risk too high! Model may be miscalibrated.")
        status = "FAIL"
    
    return {
        'probability': proba,
        'status': status,
        'passed': proba < 0.05
    }

def save_model_artifacts(model, scaler, feature_columns, metrics, output_dir='models'):
    """
    Save model and all artifacts.
    
    Parameters:
    -----------
    model : CalibratedClassifierCV
        Trained model
    scaler : StandardScaler
        Fitted scaler
    feature_columns : list
        Feature column names
    metrics : dict
        Evaluation metrics
    output_dir : str
        Output directory
    """
    print("\n" + "=" * 80)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save calibrated model
    model_path = os.path.join(output_dir, 'diabetes_model_calibrated.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Saved calibrated model: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved scaler: {scaler_path}")
    
    # Save feature columns
    columns_path = os.path.join(output_dir, 'model_columns.pkl')
    joblib.dump(feature_columns, columns_path)
    print(f"✅ Saved feature columns: {columns_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'training_metrics.pkl')
    joblib.dump(metrics, metrics_path)
    print(f"✅ Saved metrics: {metrics_path}")
    
    # Save metadata
    metadata = {
        'threshold': OPTIMAL_THRESHOLD,
        'feature_columns': feature_columns,
        'continuous_features': CONTINUOUS_FEATURES,
        'random_state': RANDOM_STATE,
        'model_type': 'XGBoost_Calibrated',
        'training_date': pd.Timestamp.now().isoformat()
    }
    metadata_path = os.path.join(output_dir, 'model_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"✅ Saved metadata: {metadata_path}")
    
    print(f"\nAll artifacts saved to: {output_dir}/")

def main():
    """Main training pipeline."""
    print("=" * 80)
    print("CLEAN DIABETES RISK MODEL TRAINING (NO SMOTE)")
    print("=" * 80)
    
    # Paths - Update these if your data is in a different location
    # Try multiple possible paths. Prefer the repository root CSV by resolving
    # the absolute path relative to this script to avoid permission issues
    # with parent/relative path resolution in some environments.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    abs_repo_csv = os.path.join(repo_root, 'diabetes_012_health_indicators_BRFSS2015.csv')

    possible_paths = [
        abs_repo_csv,
        'data/diabetes_012_health_indicators_BRFSS2015.csv',
        '../diabetes_012_health_indicators_BRFSS2015.csv',
        '../../diabetes_012_health_indicators_BRFSS2015.csv',
        '../../../diabetes_012_health_indicators_BRFSS2015.csv',  # From diabetes_risk_app/project/
        'diabetes_012_health_indicators_BRFSS2015.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(
            f"Data file not found. Tried:\n" + "\n".join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease update data_path in train_clean.py"
        )
    
    output_dir = 'models'
    
    # Load and prepare data
    X, y = load_and_prepare_data(data_path)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train/Val split (for early stopping)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"\nData splits:")
    print(f"  Train: {X_train.shape[0]:,}")
    print(f"  Val:   {X_val.shape[0]:,}")
    print(f"  Test:  {X_test.shape[0]:,}")
    
    # Scale features
    scaler, X_train_scaled, X_test_scaled, X_val_scaled = scale_features(
        X_train, X_test, X_val
    )
    
    # Calculate scale_pos_weight
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    
    # Train model
    calibrated_model, base_model = train_model(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        scale_pos_weight
    )
    
    # Evaluate
    metrics = evaluate_model(calibrated_model, X_test_scaled, y_test, OPTIMAL_THRESHOLD)
    
    # Test all NO inputs
    test_result = test_all_no_input(calibrated_model, scaler, OPTIMAL_THRESHOLD)
    
    if not test_result['passed']:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: All NO inputs test FAILED!")
        print("Model may need recalibration or threshold adjustment.")
        print("=" * 80)
    
    # Save artifacts
    save_model_artifacts(
        calibrated_model, scaler, FEATURE_COLUMNS, metrics, output_dir
    )
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel saved to: {output_dir}/")
    print(f"ROC-AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"All NO test: {test_result['status']} ({test_result['probability']*100:.2f}%)")

if __name__ == "__main__":
    main()

