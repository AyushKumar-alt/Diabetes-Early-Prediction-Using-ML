#!/usr/bin/env python
"""
Deployment Verification Script
Checks that all components are ready for production
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_check(passed, message):
    symbol = "✅" if passed else "❌"
    print(f"{symbol} {message}")
    return passed

def verify_file_structure():
    """Check that all required files exist"""
    print_header("1. FILE STRUCTURE VERIFICATION")
    
    required_files = {
        "train_clean.py": "Training pipeline",
        "predict.py": "Inference module",
        "requirements.txt": "Dependencies",
        "app/app.py": "Streamlit dashboard",
        "README.md": "Documentation",
        "SETUP_GUIDE.md": "Setup instructions",
    }
    
    required_dirs = {
        "data": "Data directory",
        "models": "Model artifacts directory",
        "notebooks": "Notebooks directory",
        "app": "App directory",
    }
    
    all_good = True
    
    print("\nRequired files:")
    for file, desc in required_files.items():
        exists = os.path.exists(file)
        all_good &= print_check(exists, f"{file:<30} ({desc})")
    
    print("\nRequired directories:")
    for dir_name, desc in required_dirs.items():
        exists = os.path.isdir(dir_name)
        all_good &= print_check(exists, f"{dir_name}/<30} ({desc})")
    
    return all_good

def verify_dependencies():
    """Check that key dependencies can be imported"""
    print_header("2. DEPENDENCIES VERIFICATION")
    
    required_packages = {
        "pandas": "Data manipulation",
        "numpy": "Numerical operations",
        "xgboost": "XGBoost model",
        "sklearn": "Scikit-learn utilities",
        "streamlit": "Streamlit app",
        "shap": "SHAP explainability",
        "joblib": "Model serialization",
    }
    
    all_good = True
    
    for package, desc in required_packages.items():
        try:
            __import__(package)
            all_good &= print_check(True, f"{package:<20} ({desc})")
        except ImportError as e:
            all_good &= print_check(False, f"{package:<20} ({desc}) - {str(e)}")
    
    return all_good

def verify_model_artifacts():
    """Check that trained model artifacts exist"""
    print_header("3. MODEL ARTIFACTS VERIFICATION")
    
    required_artifacts = {
        "models/diabetes_model_calibrated.pkl": "Trained model",
        "models/scaler.pkl": "Feature scaler",
        "models/model_columns.pkl": "Feature columns",
        "models/model_metadata.pkl": "Model metadata",
        "models/training_metrics.pkl": "Training metrics",
    }
    
    all_good = True
    
    for artifact, desc in required_artifacts.items():
        exists = os.path.exists(artifact)
        if not exists:
            print_check(False, f"{artifact:<40} ({desc}) - NOT FOUND")
            all_good = False
        else:
            try:
                joblib.load(artifact)
                all_good &= print_check(True, f"{artifact:<40} ({desc}) - OK")
            except Exception as e:
                all_good &= print_check(False, f"{artifact:<40} ({desc}) - ERROR: {str(e)}")
    
    return all_good

def verify_model_performance():
    """Check model performance metrics"""
    print_header("4. MODEL PERFORMANCE VERIFICATION")
    
    try:
        metrics = joblib.load("models/training_metrics.pkl")
        
        print("\nKey Metrics:")
        all_good = True
        
        # ROC-AUC
        auc = metrics.get('auc', 0)
        auc_ok = auc > 0.75
        all_good &= print_check(auc_ok, f"ROC-AUC: {auc:.4f} {'(> 0.75)' if auc_ok else '(< 0.75 - NEEDS IMPROVEMENT)'}")
        
        # F1 Score
        f1 = metrics.get('f1', 0)
        f1_ok = f1 > 0.40
        all_good &= print_check(f1_ok, f"F1 Score: {f1:.4f} {'(> 0.40)' if f1_ok else '(< 0.40 - NEEDS IMPROVEMENT)'}")
        
        # Brier Score
        brier = metrics.get('brier', 1)
        brier_ok = brier < 0.20
        all_good &= print_check(brier_ok, f"Brier Score: {brier:.4f} {'(< 0.20)' if brier_ok else '(> 0.20 - NEEDS IMPROVEMENT)'}")
        
        # Sensitivity
        sensitivity = metrics.get('sensitivity', 0)
        sensitivity_ok = sensitivity > 0.50
        all_good &= print_check(sensitivity_ok, f"Sensitivity: {sensitivity:.4f} {'(> 0.50)' if sensitivity_ok else '(< 0.50 - NEEDS IMPROVEMENT)'}")
        
        return all_good
        
    except Exception as e:
        print_check(False, f"Could not load metrics: {str(e)}")
        return False

def verify_all_no_test():
    """Verify the all NO inputs test passed"""
    print_header("5. ALL NO INPUTS TEST VERIFICATION")
    
    try:
        from predict import predict_risk
        import warnings
        warnings.filterwarnings('ignore')
        
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
        
        result = predict_risk(all_no_patient)
        prob = result['probability']
        risk_level = result['risk_level']
        
        test_passed = prob < 0.05 and risk_level == "Low Risk"
        
        print(f"\nAll NO inputs prediction:")
        all_good = print_check(test_passed, f"Probability: {prob:.4f} ({prob*100:.2f}%) - Risk: {risk_level}")
        
        if prob >= 0.05:
            print("\n⚠️  WARNING: Probability >= 5%")
            print("   Model may be miscalibrated")
            print("   Recommendation: Retrain model")
        
        return test_passed
        
    except Exception as e:
        print_check(False, f"Could not run all NO test: {str(e)}")
        return False

def verify_feature_columns():
    """Verify feature columns are consistent"""
    print_header("6. FEATURE COLUMNS VERIFICATION")
    
    try:
        feature_columns = joblib.load("models/model_columns.pkl")
        
        expected_count = 21
        actual_count = len(feature_columns)
        
        print(f"\nFeature columns:")
        all_good = print_check(actual_count == expected_count, f"Count: {actual_count} (expected {expected_count})")
        
        print(f"\nFeatures:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i:2d}. {col}")
        
        return all_good
        
    except Exception as e:
        print_check(False, f"Could not load feature columns: {str(e)}")
        return False

def verify_scaler():
    """Verify scaler is loaded and working"""
    print_header("7. SCALER VERIFICATION")
    
    try:
        scaler = joblib.load("models/scaler.pkl")
        
        # Check it has expected attributes
        has_mean = hasattr(scaler, 'mean_')
        has_scale = hasattr(scaler, 'scale_')
        
        all_good = print_check(has_mean, "Scaler has mean_ attribute")
        all_good &= print_check(has_scale, "Scaler has scale_ attribute")
        
        if has_mean and has_scale:
            print(f"\nScaler statistics:")
            print(f"  Number of features: {len(scaler.mean_)}")
            print(f"  Mean (first 5): {scaler.mean_[:5]}")
            print(f"  Scale (first 5): {scaler.scale_[:5]}")
        
        return all_good
        
    except Exception as e:
        print_check(False, f"Could not load scaler: {str(e)}")
        return False

def main():
    """Run all verification checks"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + "DIABETES RISK PREDICTION - DEPLOYMENT VERIFICATION".center(78) + "║")
    print("║" + "Production Readiness Checklist".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    checks = [
        ("File Structure", verify_file_structure),
        ("Dependencies", verify_dependencies),
        ("Model Artifacts", verify_model_artifacts),
        ("Model Performance", verify_model_performance),
        ("All NO Inputs Test", verify_all_no_test),
        ("Feature Columns", verify_feature_columns),
        ("Scaler", verify_scaler),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_header(f"ERROR in {name}")
            print(f"❌ Unexpected error: {str(e)}")
            results[name] = False
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nChecks Passed: {passed}/{total}")
    print("\nDetailed Results:")
    for name, result in results.items():
        symbol = "✅" if result else "❌"
        print(f"  {symbol} {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY FOR PRODUCTION DEPLOYMENT")
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE DEPLOYMENT")
        print("\nFailures:")
        for name, result in results.items():
            if not result:
                print(f"  • {name}")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
