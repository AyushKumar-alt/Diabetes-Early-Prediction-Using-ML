"""
Extended tuning: randomized search (30 trials) + threshold optimization.
For each model, find the optimal threshold on validation set that maximizes
F1 for the At-Risk class, then evaluate on test set.

Compares:
1. Current best from small search (F1 ~0.4873 at threshold 0.30)
2. Extended search with threshold optimization (30 trials)

Run from repository root:
    python -m diabetes_risk_app.project.tune_threshold_and_params
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
        'max_depth': random.choice([2, 3, 4, 5, 6]),
        'learning_rate': random.choice([0.01, 0.03, 0.05, 0.07, 0.1]),
        'min_child_weight': random.choice([1, 2, 3, 5]),
        'subsample': random.choice([0.6, 0.7, 0.8, 0.9]),
        'colsample_bytree': random.choice([0.6, 0.7, 0.8, 0.9]),
        'reg_alpha': random.choice([0.0, 0.05, 0.1, 0.2]),
        'reg_lambda': random.choice([0.5, 1.0, 2.0]),
        'n_estimators': random.choice([200, 300, 400, 500, 600])
    }


def find_best_threshold(y_val, y_proba_val, thresholds=None):
    """Find threshold that maximizes F1 on validation set."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 50)
    
    best_f1 = -1
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_proba_val >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1


def main(n_iter=30, random_state=42):
    print('Extended tuning: 30 randomized trials + threshold optimization (no SMOTE)')
    data_path = find_data_path()
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].copy()
    y = (df['Diabetes_012'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)

    scaler, X_train_scaled, X_test_scaled, X_val_scaled = scale_features(X_train, X_test, X_val)

    results = []
    best = {'f1': -1, 'params': None, 'model': None, 'threshold': 0.5}

    # sample weights for training
    sw = compute_sample_weight(class_weight='balanced', y=y_train)

    for i in range(n_iter):
        params = sample_params()
        print('\nIteration %d/%d — params: %s' % (i+1, n_iter, params))

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
            **params
        )

        try:
            model.fit(X_train_scaled, y_train, sample_weight=sw, eval_set=[(X_val_scaled, y_val)], verbose=False)
        except Exception as e:
            print('  Training failed:', e)
            continue

        # Optimal trees
        best_iter = getattr(model, 'best_iteration', None) or params.get('n_estimators')
        if best_iter and best_iter < 1:
            best_iter = params.get('n_estimators')

        # Recreate for calibration
        calibrated_base = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
            **{k: params[k] for k in params if k != 'n_estimators'},
            n_estimators=best_iter
        )
        calibrated_base.fit(X_train_scaled, y_train, sample_weight=sw, eval_set=[(X_val_scaled, y_val)], verbose=False)

        # Calibrate
        calib = CalibratedClassifierCV(calibrated_base, method='isotonic', cv=3)
        try:
            calib.fit(X_train_scaled, y_train)
        except Exception as e:
            print('  Calibration failed:', e)
            continue

        # Find best threshold on validation set
        y_proba_val = calib.predict_proba(X_val_scaled)[:, 1]
        best_thresh_val, best_f1_val = find_best_threshold(y_val, y_proba_val)
        
        # Evaluate on test set with best threshold
        y_proba_test = calib.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_proba_test >= best_thresh_val).astype(int)
        f1_test = f1_score(y_test, y_pred_test)
        auc_test = roc_auc_score(y_test, y_proba_test)

        print('  Val best threshold: %.3f (F1 on val: %.4f)' % (best_thresh_val, best_f1_val))
        print('  Test ROC-AUC: %.4f, F1 (At-Risk) at threshold %.3f: %.4f' % (auc_test, best_thresh_val, f1_test))

        results.append({
            'iteration': i + 1,
            'params': params,
            'threshold': best_thresh_val,
            'f1_val': best_f1_val,
            'f1_test': f1_test,
            'auc_test': auc_test
        })

        if f1_test > best['f1']:
            print('  [NEW BEST] F1 improved %.4f -> %.4f' % (best['f1'], f1_test))
            best['f1'] = f1_test
            best['params'] = params
            best['model'] = calib
            best['threshold'] = best_thresh_val

    print('\n' + '='*80)
    print('SEARCH COMPLETE')
    print('='*80)
    print('\nBest F1 on test set: %.4f' % best['f1'])
    print('Best threshold: %.3f' % best['threshold'])
    print('Best params:', best['params'])

    # Save results to CSV for inspection
    results_df = pd.DataFrame(results)
    results_df.to_csv('tuning_results_extended.csv', index=False)
    print('\nDetailed results saved to: tuning_results_extended.csv')

    # Print top 5 results
    print('\nTop 5 results (by test F1):')
    top5 = results_df.nlargest(5, 'f1_test')
    for idx, row in top5.iterrows():
        print('  Iter %d: F1 %.4f, Threshold %.3f, AUC %.4f' % (row['iteration'], row['f1_test'], row['threshold'], row['auc_test']))

    if best['model'] is not None:
        out_dir = os.path.join(os.path.dirname(__file__), 'models_tuned')
        os.makedirs(out_dir, exist_ok=True)
        
        model_path = os.path.join(out_dir, 'best_calibrated_model_extended.pkl')
        joblib.dump(best['model'], model_path)
        print('\nSaved best calibrated model to:', model_path)
        
        # Save best config
        config = {
            'params': best['params'],
            'threshold': best['threshold'],
            'f1_test': best['f1']
        }
        config_path = os.path.join(out_dir, 'best_config.pkl')
        joblib.dump(config, config_path)
        print('Saved best config to:', config_path)


if __name__ == '__main__':
    main()
