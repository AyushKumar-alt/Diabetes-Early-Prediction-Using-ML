# Diabetes Risk Prediction System

Production-ready diabetes risk prediction system using XGBoost binary classification.

## 🎯 Features

- ✅ Binary classification (Healthy vs At-Risk)
- ✅ Optimized threshold = 0.30 (69.2% sensitivity)
- ✅ Multi-Level Risk Stratification (4 levels)
- ✅ Personalized medical recommendations
- ✅ Feature importance visualization
- ✅ SHAP-based explainability
- ✅ Streamlit web interface

## 📁 Project Structure

```
diabetes_risk_app/
│
├── models/
│   └── bst_binary.json          # Trained XGBoost model
│
├── data/
│   └── input_example.csv        # Sample input data
│
├── utils/
│   ├── preprocess.py            # Data preprocessing
│   ├── risk_stratification.py   # Risk level mapping
│   └── explainability.py        # SHAP explanations
│
├── app/
│   └── streamlit_app.py         # Web application
│
├── core/
│   └── predict.py               # Prediction engine
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Model Performance

- **ROC-AUC**: 0.81
- **Sensitivity (Recall)**: 69.2%
- **Specificity**: 76.5%
- **F1 Score**: 0.49
- **Optimal Threshold**: 0.30

## 🎨 Risk Levels

| Risk Level | Probability Range | Recommendation |
|------------|------------------|----------------|
| **Low Risk** | < 20% | Maintain healthy lifestyle |
| **Moderate Risk** | 20-40% | Lifestyle modifications, follow-up in 6 months |
| **High Risk** | 40-70% | Schedule diabetes screening (HbA1c/FPG) |
| **Very High Risk** | > 70% | Immediate medical testing recommended |

## 💻 Usage Examples

### Python API

```python
from core.predict import predict_risk

# Patient data
patient_data = {
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

# Predict
result = predict_risk(patient_data, return_explanation=True)

print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Recommendation: {result['recommendation']}")
```

### Batch Prediction

```python
from core.predict import predict_batch
import pandas as pd

# Load data
df = pd.read_csv('data/input_example.csv')

# Predict for all rows
results = predict_batch(df.to_dict('records'))

for i, result in enumerate(results):
    print(f"Patient {i+1}: {result['risk_level']}")
```

## 🔧 Configuration

### Adjusting Threshold

Edit `core/predict.py`:

```python
OPTIMAL_THRESHOLD = 0.30  # Change this value
```

### Model Path

By default, the model is loaded from `models/bst_binary.json`. To use a different path:

```python
result = predict_risk(patient_data, model_path='path/to/model.json')
```

## 📋 Input Features

The model requires 21 health indicators:

1. **Demographics**: Age, Sex, Education, Income
2. **Health Conditions**: HighBP, HighChol, Stroke, HeartDiseaseorAttack
3. **Physical Health**: BMI, GenHlth, PhysHlth, MentHlth, DiffWalk
4. **Lifestyle**: PhysActivity, Smoker, Fruits, Veggies, HvyAlcoholConsump
5. **Healthcare**: AnyHealthcare, NoDocbcCost, CholCheck

## ⚠️ Important Notes

- This is a **screening tool**, not a diagnostic tool
- Results should be confirmed with clinical testing
- Always consult healthcare providers for medical decisions
- Model trained on BRFSS 2015 data - may not generalize to all populations

## 📚 References

- Model trained on BRFSS 2015 Diabetes Health Indicators dataset
- XGBoost documentation: https://xgboost.readthedocs.io/
- Streamlit documentation: https://docs.streamlit.io/

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

