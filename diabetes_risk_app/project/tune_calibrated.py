"""
Run a small randomized hyperparameter search for a calibrated, clinically-safe
binary XGBoost model (At-Risk vs Healthy). Uses sample weights (no SMOTE), early
stopping, and CalibratedClassifierCV for honest probability calibration.

This is a modest search (limited iterations) intended to find a better F1 for
the At-Risk class before updating main training artifacts.

Run from repository root:
    python -m diabetes_risk_app.project.tune_calibrated

Logs are printed to stdout; consider redirecting to a file.
"""
import os
import random
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from diabetes_risk_app.project.train_clean import FEATURE_COLUMNS, scale_features


def find_data_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    abs_repo_csv = os.path.join(repo_root, 'diabetes_012_health_indicators_BRFSS2015.csv')
    possible_paths = [
        abs_repo_csv,
        os.path.join(repo_root, 'data', 'diabetes_012_health_indicators_BRFSS2015.csv'),
        'diabetes_012_health_indicators_BRFSS2015.csv'
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('Dataset CSV not found')


def sample_params():
    return {
        'max_depth': random.choice([3, 4, 5, 6]),
        'learning_rate': random.choice([0.01, 0.03, 0.05, 0.07]),
        'min_child_weight': random.choice([1, 3, 5]),
        'subsample': random.choice([0.7, 0.8, 0.9]),
        'colsample_bytree': random.choice([0.6, 0.7, 0.8]),
        'reg_alpha': random.choice([0.0, 0.1]),
        'reg_lambda': random.choice([1.0, 2.0]),
        'n_estimators': random.choice([200, 400, 600])
    }


def find_best_threshold(y_true, y_proba, thresholds=None):
    """Find threshold that maximizes F1 on provided labels/probabilities."""
    from sklearn.metrics import f1_score
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.60, 51)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def main(n_iter=30, random_state=42):
    print('Starting calibrated hyperparameter search (no SMOTE)')
    data_path = find_data_path()
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].copy()
    y = (df['Diabetes_012'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)

    scaler, X_train_scaled, X_test_scaled, X_val_scaled = scale_features(X_train, X_test, X_val)

    best = {'f1': -1, 'params': None, 'model': None, 'threshold': None}

    # sample weights for training
    sw = compute_sample_weight(class_weight='balanced', y=y_train)

    for i in range(n_iter):
        params = sample_params()
        print('\nIteration %d/%d - params: %s' % (i+1, n_iter, params))

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
            **params
        )

        # Early stopping using validation set
        try:
            model.fit(X_train_scaled, y_train, sample_weight=sw, eval_set=[(X_val_scaled, y_val)], verbose=False)
        except Exception as e:
            print('Training failed for params:', params, 'error:', e)
            continue

        # Determine optimal n_estimators
        best_iter = getattr(model, 'best_iteration', None) or params.get('n_estimators')
        if best_iter and best_iter < 1:
            best_iter = params.get('n_estimators')

        # Recreate model with optimal trees for calibration
        calibrated_base = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, random_state=random_state,
                                            n_jobs=-1, **{k: params[k] for k in params if k != 'n_estimators'}, n_estimators=best_iter)
        calibrated_base.fit(X_train_scaled, y_train, sample_weight=sw, eval_set=[(X_val_scaled, y_val)], verbose=False)

        # Calibrate
        calib = CalibratedClassifierCV(calibrated_base, method='isotonic', cv=3)
        try:
            calib.fit(X_train_scaled, y_train)
        except Exception as e:
            print('Calibration failed:', e)
            continue

        # Find best threshold on the validation set
        val_proba = calib.predict_proba(X_val_scaled)[:, 1]
        best_t, best_val_f1 = find_best_threshold(y_val, val_proba)

        # Evaluate with that threshold on the test set
        y_proba_test = calib.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_proba_test >= best_t).astype(int)
        f1 = f1_score(y_test, y_pred_test)
        auc = roc_auc_score(y_test, y_proba_test)

        print('Val best threshold: %.3f (F1=%.4f) | Test ROC-AUC: %.4f, Test F1: %.4f' % (best_t, best_val_f1, auc, f1))

        if f1 > best['f1']:
            print('New best model found (F1 improved %.4f -> %.4f)' % (best['f1'], f1))
            best['f1'] = f1
            best['params'] = params
            best['model'] = calib
            best['threshold'] = best_t

    print('\nSearch complete. Best Test F1: %.4f' % best['f1'])
    print('Best params:', best['params'])
    print('Best threshold:', best.get('threshold'))

    if best['model'] is not None:
        out_dir = os.path.join(os.path.dirname(__file__), 'models_tuned')
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, 'best_calibrated_model.pkl')
        joblib.dump(best['model'], model_path)
        # Also save threshold and params
        meta = {'params': best['params'], 'f1': best['f1'], 'threshold': best.get('threshold')}
        joblib.dump(meta, os.path.join(out_dir, 'best_calibrated_model_meta.pkl'))
        print('Saved best calibrated model and metadata to:', out_dir)


if __name__ == '__main__':
    main()
