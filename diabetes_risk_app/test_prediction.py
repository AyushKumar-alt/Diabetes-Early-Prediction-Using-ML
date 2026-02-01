"""
Simple test script to verify the prediction system works.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.predict import predict_risk

# Test patient data
test_patient = {
    'HighBP': 1,
    'HighChol': 1,
    'CholCheck': 1,
    'BMI': 30.5,
    'Smoker': 0,
    'Stroke': 0,
    'HeartDiseaseorAttack': 0,
    'PhysActivity': 1,
    'Fruits': 1,
    'Veggies': 1,
    'HvyAlcoholConsump': 0,
    'AnyHealthcare': 1,
    'NoDocbcCost': 0,
    'GenHlth': 3,
    'MentHlth': 0,
    'PhysHlth': 0,
    'DiffWalk': 0,
    'Sex': 1,
    'Age': 7,
    'Education': 4,
    'Income': 6
}

if __name__ == "__main__":
    print("Testing Diabetes Risk Prediction System...")
    print("=" * 60)
    
    try:
        result = predict_risk(test_patient, return_explanation=False)
        
        print("\n✅ Prediction successful!")
        print(f"\nResults:")
        print(f"  Risk Level: {result['risk_level']} {result['risk_icon']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Confidence: {result['confidence']}")
        print(f"\nRecommendation:")
        print(f"  {result['recommendation']}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

