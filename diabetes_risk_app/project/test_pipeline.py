"""
Test script to validate the clean pipeline.
Tests that all NO inputs give LOW risk.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import predict_risk, validate_all_no_input

def test_all_no_inputs():
    """Test that all NO inputs give LOW risk."""
    print("=" * 80)
    print("TEST: All NO Inputs → LOW Risk")
    print("=" * 80)
    
    # Create all NO patient (healthy baseline)
    all_no_patient = {
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
    }
    
    try:
        result = predict_risk(all_no_patient)
        
        print(f"\nResult:")
        print(f"  Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Prediction: {result['prediction']}")
        
        print(f"\nExpected:")
        print(f"  Probability: < 0.05 (< 5%)")
        print(f"  Risk Level: Low Risk")
        print(f"  Prediction: Healthy")
        
        # Validation
        passed = (
            result['probability'] < 0.05 and
            result['risk_level'] == "Low Risk" and
            result['prediction'] == "Healthy"
        )
        
        if passed:
            print("\n✅ PASS: All NO inputs correctly predict LOW risk")
            return True
        else:
            print("\n❌ FAIL: Model predicts incorrect risk level")
            print(f"   Probability too high: {result['probability']*100:.2f}% (expected < 5%)")
            return False
            
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Model not found. Run train_clean.py first.")
        print(f"   {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_high_risk_patient():
    """Test a high-risk patient."""
    print("\n" + "=" * 80)
    print("TEST: High-Risk Patient")
    print("=" * 80)
    
    high_risk_patient = {
        'HighBP': 1,
        'HighChol': 1,
        'CholCheck': 1,
        'BMI': 35.0,  # Obese
        'Smoker': 1,
        'Stroke': 0,
        'HeartDiseaseorAttack': 1,
        'PhysActivity': 0,  # No exercise
        'Fruits': 0,  # No fruits
        'Veggies': 0,  # No veggies
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 0,  # No healthcare
        'NoDocbcCost': 1,  # Can't afford doctor
        'GenHlth': 4,  # Poor health
        'MentHlth': 10,
        'PhysHlth': 15,
        'DiffWalk': 1,  # Difficulty walking
        'Sex': 1,  # Male
        'Age': 10,  # Older
        'Education': 2,  # Low education
        'Income': 2  # Low income
    }
    
    try:
        result = predict_risk(high_risk_patient)
        
        print(f"\nResult:")
        print(f"  Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Prediction: {result['prediction']}")
        
        # Should be at-risk
        if result['prediction'] == "At-Risk" and result['probability'] > 0.30:
            print("\n✅ PASS: High-risk patient correctly identified")
            return True
        else:
            print("\n⚠️  WARNING: High-risk patient may not be correctly identified")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("CLEAN PIPELINE VALIDATION TESTS")
    print("=" * 80)
    
    # Test 1: All NO inputs
    test1_passed = test_all_no_inputs()
    
    # Test 2: High-risk patient
    test2_passed = test_high_risk_patient()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"All NO Inputs Test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"High-Risk Patient Test: {'✅ PASS' if test2_passed else '⚠️  WARNING'}")
    
    if test1_passed:
        print("\n✅ Pipeline validation PASSED")
    else:
        print("\n❌ Pipeline validation FAILED - Model needs retraining/calibration")

