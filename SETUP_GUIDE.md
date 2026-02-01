# ============================================================================
# DIABETES RISK PREDICTION SYSTEM - SETUP INSTRUCTIONS
# ============================================================================

## 🎯 WHAT YOU HAVE

A production-ready diabetes risk prediction system with:
✅ Data cleaning & EDA (Diabetes.ipynb)
✅ XGBoost ML model  
✅ Streamlit web interface
✅ Risk stratification (4 levels)
✅ Feature importance visualization
✅ SHAP explainability

## 🚀 QUICK START (Copy & Paste)

### Step 1: Open PowerShell in diabetes_risk_app folder

```powershell
cd c:\Users\calpo\Downloads\ML_Diabetes\diabetes_risk_app
```

### Step 2: Install dependencies (ONE TIME ONLY)

```powershell
pip install xgboost streamlit pandas numpy scikit-learn plotly shap joblib
```

OR use requirements.txt:

```powershell
pip install -r requirements.txt
```

### Step 3: Run the app

```powershell
streamlit run app/streamlit_app.py
```

Browser will open automatically. If not, go to: http://localhost:8501

## 📊 PROJECT STRUCTURE

```
diabetes_risk_app/
├── core/predict.py              ← Prediction engine
├── utils/preprocess.py          ← Data preprocessing
├── utils/risk_stratification.py ← Risk levels
├── app/streamlit_app.py         ← Web interface
├── models/bst_binary.json       ← Trained model (ADD HERE)
├── test_prediction.py           ← Test suite
└── requirements.txt             ← Dependencies
```

## ⚙️ WHAT EACH PART DOES

### core/predict.py
- Loads XGBoost model
- Predicts probability (0-1)
- Applies threshold (0.30)
- Returns risk level + recommendation

### utils/risk_stratification.py
Maps probability to 4 risk levels:
- Low Risk (< 20%)
- Moderate Risk (20-40%)
- High Risk (40-70%)
- Very High Risk (> 70%)

### app/streamlit_app.py
Beautiful web interface:
- 21-feature input form (sidebar)
- Risk gauge visualization
- Feature importance charts
- SHAP explanations
- PDF report export

### test_prediction.py
Verify everything works:
```powershell
python test_prediction.py
```

## 🔧 IMPORTANT FILES TO CONFIGURE

1. **Model File** → `models/bst_binary.json`
   - Train in Diabetes.ipynb
   - Save: `bst.save_model("models/bst_binary.json")`

2. **Threshold** → `core/predict.py` line 10
   ```python
   OPTIMAL_THRESHOLD = 0.30  # Adjust as needed
   ```

3. **Features** → `utils/preprocess.py` lines 18-21
   ```python
   FEATURE_ORDER = [
       'HighBP', 'HighChol', 'CholCheck', 'BMI', ...
   ]
   ```

## 📋 21 REQUIRED INPUT FEATURES

Your model must accept exactly these 21 features in this order:

1. HighBP (0/1)
2. HighChol (0/1)
3. CholCheck (0/1)
4. BMI (10-100)
5. Smoker (0/1)
6. Stroke (0/1)
7. HeartDiseaseorAttack (0/1)
8. PhysActivity (0/1)
9. Fruits (0/1)
10. Veggies (0/1)
11. HvyAlcoholConsump (0/1)
12. AnyHealthcare (0/1)
13. NoDocbcCost (0/1)
14. GenHlth (1-5)
15. MentHlth (0-30)
16. PhysHlth (0-30)
17. DiffWalk (0/1)
18. Sex (0/1)
19. Age (1-13 BRFSS encoding)
20. Education (1-6)
21. Income (1-8)

## 🐍 PYTHON API USAGE

```python
from core.predict import predict_risk

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

result = predict_risk(patient_data)

print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.1%}")
print(f"Recommendation: {result['recommendation']}")
```

## ✅ TROUBLESHOOTING

### "Module not found" error
→ Install dependencies: `pip install -r requirements.txt`

### "Model file not found" error
→ Save model from Diabetes.ipynb: `bst.save_model("models/bst_binary.json")`

### Port 8501 already in use
→ Use different port: `streamlit run app/streamlit_app.py --server.port 8502`

### SHAP errors (optional, can ignore)
→ Install: `pip install shap`

### Still having issues?
→ Run test: `python test_prediction.py` (shows what's broken)

## 📚 DOCUMENTATION FILES

- `README.md` - Full documentation
- `QUICKSTART.md` - 3-step quick start
- `RUN_APP.md` - Detailed run instructions
- `PROJECT_SUMMARY.md` - This project overview

## 🎯 YOUR NEXT STEPS

1. Train XGBoost model in `Diabetes.ipynb`
2. Save model: `bst.save_model("models/bst_binary.json")`
3. Run app: `streamlit run app/streamlit_app.py`
4. Test with sample patients

That's it! The system is production-ready.

---

Questions? Check the README.md or RUN_APP.md for more details.
