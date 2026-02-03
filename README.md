# 🩺 DIABETES RISK PREDICTION SYSTEM - COMPLETE PROJECT GUIDE

## 📦 WHAT YOU HAVE

A complete, production-ready diabetes risk prediction system built with:
- XGBoost ML model (binary classification)
- Streamlit web interface (21-feature input)
- Risk stratification (4 medical levels)
- SHAP explainability
- Automated testing

## 📁 PROJECT STRUCTURE

```
c:\Users\calpo\Downloads\ML_Diabetes\
│
├── 📓 Diabetes.ipynb                              ← Main notebook
│   ├── Cell 1: Load packages
│   ├── Cell 2: Load CSV (253,680 rows)
│   ├── Cell 3: Data info & missingness
│   ├── Cell 4-8: Data exploration
│   ├── Cell 9: Outlier detection (IQR, z-score)
│   ├── Cell 10: EDA & cleaning
│   ├── Cell 11-18: Model training (XGBoost, SMOTE)
│   └── Cell 19: Save processed data
│
├── 📊 diabetes_012_health_indicators_BRFSS2015.csv  ← Raw data
│
├── 📄 PROJECT_SUMMARY.md                        ← Project overview
├── 📄 SETUP_GUIDE.md                            ← Setup instructions
│
└── 📂 diabetes_risk_app/                        ← PRODUCTION APP
    │
    ├── 🎯 app/
    │   ├── __init__.py
    │   └── streamlit_app.py  ← RUN THIS: streamlit run app/streamlit_app.py
    │       • 21-feature input form (sidebar)
    │       • Risk gauge visualization
    │       • Feature importance charts
    │       • SHAP explanations
    │       • Report export
    │
    ├── ⚙️ core/
    │   ├── __init__.py
    │   └── predict.py
    │       • load_model() - Load XGBoost
    │       • predict_risk() - Main prediction
    │       • batch_predict() - Batch processing
    │
    ├── 🔧 utils/
    │   ├── __init__.py
    │   ├── preprocess.py
    │   │   • Feature ordering (21 features)
    │   │   • StandardScaler for continuous features
    │   │   • DMatrix creation
    │   │
    │   ├── risk_stratification.py
    │   │   • Low Risk (< 20%)
    │   │   • Moderate Risk (20-40%)
    │   │   • High Risk (40-70%)
    │   │   • Very High Risk (> 70%)
    │   │
    │   └── explainability.py
    │       • SHAP feature importance
    │       • Patient comparison
    │       • Feature impact analysis
    │
    ├── 📊 models/
    │   └── bst_binary.json  ← SAVE YOUR MODEL HERE
    │       (from Diabetes.ipynb: bst.save_model("models/bst_binary.json"))
    │
    ├── 📈 data/
    │   └── input_example.csv  ← Sample input (4 examples)
    │
    ├── 🧪 test_prediction.py
    │   • Run: python test_prediction.py
    │   • Tests all components
    │
    ├── 📋 requirements.txt  ← Install: pip install -r requirements.txt
    │
    ├── 📖 README.md  ← Full documentation
    ├── 📖 RUN_APP.md  ← How to run
    └── 📖 QUICKSTART.md  ← 3-step start
```

---

## 🚀 QUICK START (3 STEPS)

### Step 1: Install dependencies
```powershell
cd c:\Users\calpo\Downloads\ML_Diabetes\diabetes_risk_app
pip install -r requirements.txt
```

### Step 2: Train model (in Diabetes.ipynb)
```python
# At the end of your notebook:
bst.save_model("models/bst_binary.json")
```

### Step 3: Run the app
```powershell
streamlit run app/streamlit_app.py
```

**Opens browser: http://localhost:8501**

---

## 📊 HOW THE SYSTEM WORKS

### Input (21 Features)
Patient health data (demographics, health conditions, lifestyle, etc.)
        ↓
### Preprocessing (utils/preprocess.py)
Validate → Order → Scale → DMatrix
        ↓
### Model (core/predict.py)
Load XGBoost → Predict probability (0-1)
        ↓
### Threshold Application
Probability ≥ 0.30 → "At-Risk", else "Healthy"
        ↓
### Risk Stratification (utils/risk_stratification.py)
Map probability to 4 levels + recommendation
        ↓
### Output
Risk Level + Probability + Recommendation + Feature Importance
        ↓
### Visualization (app/streamlit_app.py)
Beautiful web UI with charts and export

---

## 🎯 21 REQUIRED FEATURES

```
Demographics (4):       Age, Sex, Education, Income
Health Conditions (4):  HighBP, HighChol, Stroke, HeartDiseaseorAttack
Physical Health (5):    BMI, GenHlth, PhysHlth, MentHlth, DiffWalk
Lifestyle (5):          PhysActivity, Smoker, Fruits, Veggies, HvyAlcoholConsump
Healthcare (3):         AnyHealthcare, NoDocbcCost, CholCheck
```

---

## 💻 USAGE EXAMPLES

### WEB INTERFACE
1. Run: `streamlit run app/streamlit_app.py`
2. Fill in patient data (sidebar)
3. Click "Predict Risk"
4. View results, charts, recommendations
5. Export report

### PYTHON API
```python
from core.predict import predict_risk

patient_data = {
    'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 30.0,
    'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
    'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'GenHlth': 3, 'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0,
    'Sex': 1, 'Age': 7, 'Education': 4, 'Income': 6
}

result = predict_risk(patient_data)
print(f"Risk: {result['risk_level']}")
print(f"Prob: {result['probability']:.1%}")
print(f"Rec:  {result['recommendation']}")
```

### BATCH PROCESSING
```python
from core.predict import predict_batch
import pandas as pd

df = pd.read_csv('data/input_example.csv')
results = predict_batch(df.to_dict('records'))
```

---

## ⚙️ KEY FILES & WHAT THEY DO

| File | Purpose | Key Function |
|------|---------|-------------|
| `core/predict.py` | Prediction engine | `predict_risk(patient_data)` |
| `utils/preprocess.py` | Data preprocessing | `preprocess_input(patient_data)` |
| `utils/risk_stratification.py` | Risk mapping | `stratify_risk(probability)` |
| `utils/explainability.py` | SHAP explanations | `explain_prediction(model, dmatrix)` |
| `app/streamlit_app.py` | Web UI | Full Streamlit application |
| `test_prediction.py` | Testing | Automated test suite |

---

## 🔧 CONFIGURATION

### Change Risk Threshold
Edit `core/predict.py` line 10:
```python
OPTIMAL_THRESHOLD = 0.30  # Change this value (0.0-1.0)
```

### Change Model Path
```python
result = predict_risk(patient_data, model_path='custom/path/model.json')
```

### Adjust Risk Levels
Edit `utils/risk_stratification.py` `stratify_risk()` function

---

## 📋 RISK STRATIFICATION LEVELS

| Level | Probability | Recommendation |
|-------|------------|-----------------|
| **Low Risk** | < 20% | Maintain healthy lifestyle |
| **Moderate Risk** | 20-40% | Lifestyle changes, recheck in 6 months |
| **High Risk** | 40-70% | Schedule diabetes screening (HbA1c/FPG) |
| **Very High Risk** | > 70% | Immediate medical testing required |

---

## 🧪 TESTING

```powershell
cd c:\Users\calpo\Downloads\ML_Diabetes\diabetes_risk_app
python test_prediction.py
```

Expected output:
```
✅ Imports ........................... ✅ PASS
✅ Model Loading ..................... ✅ PASS
✅ Risk Stratification ............... ✅ PASS
✅ Preprocessing ..................... ✅ PASS
✅ Prediction ........................ ✅ PASS
================================================
Tests Passed: 5/5
🎉 All tests passed! The app is ready to run.
```

---

## ⚠️ IMPORTANT NOTES

1. **MODEL NOT INCLUDED**: Save from Diabetes.ipynb
   ```python
   bst.save_model("models/bst_binary.json")
   ```

2. **NOT A DIAGNOSTIC TOOL**: Screen only, consult healthcare providers

3. **FEATURE VALIDATION**: All 21 features must be provided

4. **OPTIONAL SHAP**: Works without it (for explanations)

5. **THRESHOLD TUNING**: 0.30 optimized for sensitivity (69.2%)

---

## 🆘 TROUBLESHOOTING

| Error | Solution |
|-------|----------|
| "streamlit: command not found" | `pip install streamlit` |
| "Model file not found" | Save from notebook: `bst.save_model("models/bst_binary.json")` |
| "ModuleNotFoundError" | `pip install -r requirements.txt` |
| Port 8501 in use | `streamlit run app/streamlit_app.py --server.port 8502` |
| SHAP not working | Optional - app works without it (`pip install shap`) |

---

## 📚 FILE DESCRIPTIONS

### app/streamlit_app.py
Full-featured web interface with:
- Organized sidebar with 21 input fields
- Risk gauge visualization
- Feature importance bar charts
- SHAP explainability
- Recommendation boxes
- Report export (TXT download)

### core/predict.py
Main prediction engine:
- `load_model()`: Loads XGBoost from JSON
- `predict_risk()`: Single prediction
- `batch_predict()`: Multiple predictions
- Returns: probability, risk_level, recommendation, confidence

### utils/preprocess.py
Data preprocessing:
- Input validation
- Feature ordering (21 fixed order)
- StandardScaler for continuous features
- DMatrix creation for XGBoost

### utils/risk_stratification.py
Medical risk classification:
- 4 levels based on probability
- Personalized recommendations
- Risk colors and icons
- Severity scores

### utils/explainability.py
SHAP-based explanations:
- Feature importance extraction
- SHAP value computation
- Patient comparison
- Feature impact analysis

---

## 🎓 NEXT STEPS

1. **Train Model** (Diabetes.ipynb):
   - Run all cells through model training
   - Save: `bst.save_model("models/bst_binary.json")`

2. **Run App**: 
   - `streamlit run app/streamlit_app.py`

3. **Test System**:
   - `python test_prediction.py`

4. **Customize** (Optional):
   - Adjust threshold in `core/predict.py`
   - Change risk levels in `utils/risk_stratification.py`
   - Modify UI in `app/streamlit_app.py`

5. **Deploy** (Production):
   - Streamlit Cloud
   - Docker container
   - AWS/GCP/Azure
   - FastAPI REST API

---

## 📞 SUPPORT

- README.md - Full documentation
- RUN_APP.md - Detailed instructions
- QUICKSTART.md - 3-step quick start
- test_prediction.py - System validation

---

**Status**: ✅ READY TO USE

**Next Action**: 
1. Train model in Diabetes.ipynb
2. Save to models/bst_binary.json
3. Run: streamlit run app/streamlit_app.py

---

Created: November 15, 2025
