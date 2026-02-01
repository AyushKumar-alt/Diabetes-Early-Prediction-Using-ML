# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

Open terminal in the `diabetes_risk_app` folder and run:

```bash
pip install -r requirements.txt
```

### Step 2: Test the System

Verify everything works:

```bash
python test_prediction.py
```

You should see:
```
✅ Prediction successful!
Results:
  Risk Level: [some level]
  Prediction: [Healthy/At-Risk]
  ...
```

### Step 3: Run the Web App

```bash
streamlit run app/streamlit_app.py
```

The app will open automatically in your browser!

## 📝 Using the Web Interface

1. **Fill in Patient Data**: Use the sidebar to enter health indicators
2. **Click "Predict Risk"**: Get instant risk assessment
3. **View Results**: 
   - Risk level and probability
   - Personalized recommendation
   - Feature importance
   - SHAP explanations

## 🐍 Using the Python API

```python
from core.predict import predict_risk

patient_data = {
    'HighBP': 1,
    'HighChol': 1,
    'BMI': 30.5,
    # ... (all 21 features)
}

result = predict_risk(patient_data)
print(result['risk_level'])
print(result['recommendation'])
```

## ⚠️ Troubleshooting

### Import Errors
- Make sure you're in the `diabetes_risk_app` directory
- Verify all dependencies are installed: `pip list`

### Model Not Found
- Check that `models/bst_binary.json` exists
- Verify file path in `core/predict.py`

### SHAP Errors
- SHAP is optional - the app will work without it
- Install with: `pip install shap`

## 📚 Next Steps

- Read `README.md` for full documentation
- Customize risk thresholds in `core/predict.py`
- Modify UI in `app/streamlit_app.py`

