"""
Train and evaluate the clinically-safe binary model (At-Risk vs Healthy).
This script uses the safe helper `train_xgb_binary_safe` defined in
`diabetes_risk_app.project.train_clean` and prints honest evaluation metrics.

Run from repository root:
    python -m diabetes_risk_app.project.medical_safe_train_and_eval

"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from diabetes_risk_app.project.train_clean import (
    FEATURE_COLUMNS, CONTINUOUS_FEATURES, scale_features,
    train_xgb_binary_safe
)


def find_data_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    abs_repo_csv = os.path.join(repo_root, 'diabetes_012_health_indicators_BRFSS2015.csv')
    possible_paths = [
        abs_repo_csv,
        os.path.join(os.path.dirname(__file__), 'data', 'diabetes_012_health_indicators_BRFSS2015.csv'),
        os.path.join(repo_root, 'data', 'diabetes_012_health_indicators_BRFSS2015.csv'),
        'diabetes_012_health_indicators_BRFSS2015.csv'
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('Could not find dataset csv. Update paths in script.')


def main():
    print("Training final medical-grade binary model...")

    data_path = find_data_path()
    df = pd.read_csv(data_path)

    # Prepare features/target consistent with train_clean
    X = df[FEATURE_COLUMNS].copy()
    y = (df['Diabetes_012'] > 0).astype(int)

    # Train/Val/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Scale
    scaler, X_train_scaled, X_test_scaled, X_val_scaled = scale_features(X_train, X_test, X_val)

    # Train using the safe binary trainer
    bst_binary_final = train_xgb_binary_safe(X_train_scaled, y_train, X_val_scaled, y_val)

    # Evaluate
    y_test_bin = (y_test >= 1).astype(int)
    y_pred_bin = bst_binary_final.predict(X_test_scaled)
    y_proba_bin = bst_binary_final.predict_proba(X_test_scaled)[:, 1]

    print("\nREAL-WORLD PERFORMANCE (NO SMOTE):")
    print(classification_report(y_test_bin, y_pred_bin, target_names=['Healthy', 'At-Risk']))
    print("ROC-AUC:", round(roc_auc_score(y_test_bin, y_proba_bin), 4))
    print("F1 At-Risk class:", round(f1_score(y_test_bin, y_pred_bin), 4))


if __name__ == '__main__':
    main()
