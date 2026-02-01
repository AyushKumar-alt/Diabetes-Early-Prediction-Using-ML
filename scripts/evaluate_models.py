#!/usr/bin/env python3
"""
Evaluate saved diabetes risk models using the project's dataset and saved artifacts.
Creates a reproducible train/test split using the same settings as the training script
and evaluates the calibrated binary model; optionally evaluates a 3-class XGBoost model
if one is present.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

# Attempt to import helpers from train_clean (preferred)
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'diabetes_risk_app', 'project')))
    import train_clean as tc
    HAS_TRAIN_MODULE = True
except Exception:
    HAS_TRAIN_MODULE = False

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_CANDIDATES = [
    os.path.join(REPO_ROOT, 'diabetes_012_health_indicators_BRFSS2015.csv'),
    os.path.join(REPO_ROOT, 'diabetes_risk_app', 'project', 'diabetes_012_health_indicators_BRFSS2015.csv'),
    os.path.join(REPO_ROOT, 'diabetes_risk_app', 'diabetes_012_health_indicators_BRFSS2015.csv')
]

DATA_PATH = None
for p in CSV_CANDIDATES:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError('Could not locate diabetes_012_health_indicators_BRFSS2015.csv in repository.')

MODELS_DIR = os.path.join(REPO_ROOT, 'diabetes_risk_app', 'models')
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join(REPO_ROOT, 'models') if os.path.isdir(os.path.join(REPO_ROOT,'models')) else MODELS_DIR

print('Data path:', DATA_PATH)
print('Models dir:', MODELS_DIR)

# Load and prepare data using train_clean helpers if available, otherwise implement minimal preparation
if HAS_TRAIN_MODULE:
    FEATURE_COLUMNS = tc.FEATURE_COLUMNS
    CONTINUOUS_FEATURES = tc.CONTINUOUS_FEATURES
    RANDOM_STATE = tc.RANDOM_STATE
    TEST_SIZE = tc.TEST_SIZE
    OPTIMAL_THRESHOLD = getattr(tc, 'OPTIMAL_THRESHOLD', 0.30)

    X, y = tc.load_and_prepare_data(DATA_PATH)
else:
    # Minimal hard-coded columns fallback (keep in sync with training script)
    FEATURE_COLUMNS = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    CONTINUOUS_FEATURES = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    OPTIMAL_THRESHOLD = 0.30

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df['target'] = (df['Diabetes_012'] > 0).astype(int)
    X = df[FEATURE_COLUMNS].copy()
    y = df['target'].copy()

# Train/test split reproducibly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Scaling: prefer saved scaler if available
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print('Loaded scaler from', SCALER_PATH)
    X_test_scaled = X_test.copy()
    X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])
else:
    # Fit new scaler on training data (best-effort)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train[CONTINUOUS_FEATURES])
    X_test_scaled = X_test.copy()
    X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])
    print('Fitted new scaler on training data (no saved scaler found)')

# Load calibrated model
CALIB_MODEL_PATH = os.path.join(MODELS_DIR, 'diabetes_model_calibrated.pkl')
calibrated_model = None
if os.path.exists(CALIB_MODEL_PATH):
    try:
        calibrated_model = joblib.load(CALIB_MODEL_PATH)
        print('Loaded calibrated model from', CALIB_MODEL_PATH)
    except Exception as e:
        print('Failed to load calibrated model:', e)

# Load XGBoost booster (legacy) if present
XGB_JSON_PATH = None
for candidate in [os.path.join(MODELS_DIR, 'bst_binary.json'), os.path.join(MODELS_DIR, 'bst.json')]:
    if os.path.exists(candidate):
        XGB_JSON_PATH = candidate
        break

xgb_booster = None
if XGB_JSON_PATH:
    try:
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(XGB_JSON_PATH)
        print('Loaded xgboost booster from', XGB_JSON_PATH)
    except Exception as e:
        print('Failed to load xgboost booster:', e)

# Helper: evaluate a sklearn-like model (CalibratedClassifierCV or XGB scikit wrapper)
def eval_model_sklearn_like(model, X_t, y_t, threshold=OPTIMAL_THRESHOLD):
    y_proba = model.predict_proba(X_t)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_t, y_proba)
    f1 = f1_score(y_t, y_pred)
    precision = precision_score(y_t, y_pred)
    recall = recall_score(y_t, y_pred)
    cm = confusion_matrix(y_t, y_pred)
    return {
        'y_proba': y_proba,
        'y_pred': y_pred,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'cm': cm
    }

# Evaluate calibrated model
if calibrated_model is not None:
    print('\nEvaluating calibrated binary model...')
    metrics = eval_model_sklearn_like(calibrated_model, X_test_scaled, y_test, threshold=OPTIMAL_THRESHOLD)
    print('ROC-AUC:', f"{metrics['auc']:.4f}")
    print('F1:', f"{metrics['f1']:.4f}")
    print('Confusion matrix:\n', metrics['cm'])
else:
    print('\nNo calibrated model found to evaluate.')

# Evaluate XGBoost booster if it's multiclass (3-class) or binary
if xgb_booster is not None:
    print('\nEvaluating XGBoost booster (legacy model)...')
    dtest = xgb.DMatrix(X_test_scaled)
    y_pred_proba = xgb_booster.predict(dtest)
    y_pred_proba = np.array(y_pred_proba)
    print('Booster predict output shape:', y_pred_proba.shape)
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 2:
        # multi-class
        y_pred_3class = y_pred_proba.argmax(axis=1)
        y_test_bin = None
        try:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=np.unique(y_pred_3class))
            roc_auc_3class = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='macro')
        except Exception:
            roc_auc_3class = None
        f1_macro = f1_score(y_test, y_pred_3class, average='macro')
        print('3-class F1 (macro):', f"{f1_macro:.4f}")
        if roc_auc_3class is not None:
            print('3-class ROC-AUC (ovr macro):', f"{roc_auc_3class:.4f}")
    elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
        # binary probabilities
        y_proba = y_pred_proba[:, 1]
        y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        f1v = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print('Binary booster metrics:')
        print('ROC-AUC:', f"{auc:.4f}")
        print('F1:', f"{f1v:.4f}")
        print('Confusion matrix:\n', cm)
    else:
        # Some boosters return 1D array for binary
        if y_pred_proba.ndim == 1:
            y_proba = y_pred_proba
            y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
            auc = roc_auc_score(y_test, y_proba)
            f1v = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            print('Binary booster metrics:')
            print('ROC-AUC:', f"{auc:.4f}")
            print('F1:', f"{f1v:.4f}")
            print('Confusion matrix:\n', cm)
        else:
            print('Unexpected prediction shape from booster; skipping detailed evaluation.')

# Risk stratification example and production function
print('\nProduction prediction example using calibrated model (if available)')

def categorize_risk(p):
    if p < 0.30:
        return 'Low Risk'
    elif p < 0.60:
        return 'Moderate Risk'
    elif p < 0.80:
        return 'High Risk'
    else:
        return 'Very High Risk'

if calibrated_model is not None:
    example = X_test_scaled.iloc[0:1]
    proba = calibrated_model.predict_proba(example)[0,1]
    print('Example probability:', f"{proba:.4f}")
    print('Risk level:', categorize_risk(proba))

print('\nEvaluation script finished.')
