# 📋 Complete Setup & Deployment Guide

## 🎯 Overview

This is a **production-ready, medically-valid** diabetes risk prediction system:
- ✅ **NO SMOTE** - Uses `scale_pos_weight` for class imbalance
- ✅ **Calibrated Probabilities** - `CalibratedClassifierCV` for reliability
- ✅ **Validated Edge Cases** - All NO inputs → LOW risk (< 5%)
- ✅ **Proper Thresholding** - Optimized for screening (0.30)
- ✅ **SHAP Explainability** - Shows feature contributions per patient
- ✅ **Reproducible** - Deterministic pipeline with saved artifacts

---

## 🚀 Step-by-Step Setup

### Step 1: Clone/Copy Project

```bash
cd path/to/your/projects
# Copy the project folder here
```

### Step 2: Install Dependencies

```bash
cd project
pip install -r requirements.txt
```

**What gets installed:**
- `xgboost` - Model training
- `scikit-learn` - Preprocessing, calibration, metrics
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `streamlit` - Web dashboard
- `shap` - Explainability
- `plotly` - Interactive visualizations
- `joblib` - Model serialization

### Step 3: Place Your Data

The CSV file should be at:
```
project/data/diabetes_012_health_indicators_BRFSS2015.csv
```

Or update the path in `train_clean.py`:
```python
data_path = 'path/to/your/diabetes_012_health_indicators_BRFSS2015.csv'
```

### Step 4: Train the Model

```bash
python train_clean.py
```

**What happens:**
1. ✅ Loads and cleans data
2. ✅ Stratified train/val/test split
3. ✅ Feature scaling (continuous only)
4. ✅ Trains XGBoost with `scale_pos_weight` (NO SMOTE)
5. ✅ Calibrates probabilities with CalibratedClassifierCV
6. ✅ Evaluates on test set
7. ✅ **Validates**: All NO inputs → LOW risk (< 5%)
8. ✅ Saves model artifacts to `models/`

**Output:**
```
models/
├── diabetes_model_calibrated.pkl   # Trained & calibrated model
├── scaler.pkl                      # Feature scaler
├── model_columns.pkl               # Feature column order
├── model_metadata.pkl              # Threshold, config
└── training_metrics.pkl            # Evaluation metrics
```

### Step 5: Run Streamlit App

```bash
streamlit run app/app.py
```

**Opens at:** http://localhost:8501 (default Streamlit port)

---

## ✅ Validation Checklist

After training, verify:

- [ ] **Data Loaded**: ✓ Shows sample size and class distribution
- [ ] **Model Trained**: ✓ Shows best iteration
- [ ] **Calibration Complete**: ✓ Calibration method displayed
- [ ] **All NO Test**: ✓ **MUST PASS** - Check green checkmark
- [ ] **ROC-AUC > 0.75**: ✓ Look for score in output
- [ ] **Brier Score < 0.20**: ✓ Lower is better
- [ ] **Model Saved**: ✓ Check `models/` directory

### Critical: All NO Inputs Test

**What it tests:**
A patient with all "No" answers should get LOW risk (< 5%)

**If it PASSES:**
```
✅ PASS: Low risk as expected
Probability: 0.0234 (2.34%)
```

**If it FAILS:**
```
❌ FAIL: Risk too high! Model may be miscalibrated.
Probability: 0.8541 (85.41%)
```

**If it fails:**
1. Check model calibration in `train_clean.py`
2. Verify feature scaling is correct
3. Review training data quality
4. May need to adjust calibration method (isotonic vs sigmoid)

---

## 🏥 Streamlit App Usage

### Input Form

The left sidebar has all 21 health indicators:

**Demographics:**
- Age Group (1-13)
- Sex (Male/Female)
- Education Level
- Income Level

**Health Conditions:**
- High Blood Pressure (Yes/No)
- High Cholesterol (Yes/No)
- Cholesterol Check (Past 5 years)
- History of Stroke
- Heart Disease or Attack

**Physical Health:**
- BMI (numeric)
- General Health (1-5 scale)
- Physical Health Days (0-30)
- Mental Health Days (0-30)
- Difficulty Walking/Climbing Stairs

**Lifestyle:**
- Physical Activity (Yes/No)
- Smoking Status
- Fruits (1+ times/day)
- Vegetables (1+ times/day)
- Heavy Alcohol Consumption

**Healthcare:**
- Has Healthcare Coverage
- Could Not See Doctor (Due to Cost)

### Output Display

After clicking **"Predict Risk"**:

1. **Risk Level Badge**
   - Color-coded (green/yellow/orange/red)
   - Clear text label

2. **Probability**
   - Percentage (0-100%)
   - Confidence level

3. **Recommendation**
   - Personalized based on risk level
   - Clinical guidance

4. **Feature Importance (SHAP)**
   - Top 10 factors driving risk
   - Shows contribution direction

---

## 🐛 Troubleshooting

### Error: "Data file not found"

**Problem:** CSV file not in expected location

**Fix:**
```python
# Edit train_clean.py
data_path = 'path/to/diabetes_012_health_indicators_BRFSS2015.csv'
```

Or place file in `project/data/`

### Error: "Model not found"

**Problem:** Trying to run app without training first

**Fix:**
```bash
python train_clean.py  # Train model first
streamlit run app/app.py  # Then run app
```

### Error: "All NO inputs test FAILED"

**Problem:** Model predictions are miscalibrated

**Diagnostic steps:**
1. Check training data quality (no obvious errors)
2. Verify scaler parameters are reasonable
3. Check calibration method (isotonic recommended)
4. Review feature distributions

**Fix options:**
1. Try different calibration method:
   ```python
   # In train_clean.py, change:
   method='sigmoid'  # instead of 'isotonic'
   ```

2. Adjust threshold:
   ```python
   # In predict.py, change:
   DEFAULT_THRESHOLD = 0.25  # or 0.35
   ```

### Error: "Missing features" in prediction

**Problem:** Feature order mismatch

**Fix:**
```python
# Always load from saved file
feature_columns = joblib.load('models/model_columns.pkl')
# Don't hardcode feature names
```

### App shows "999% risk" for valid patient

**Problem:** Features not scaled correctly or scaler not loaded

**Fix:**
```python
# In predict.py, verify:
scaler = joblib.load('models/scaler.pkl')
# And ensure CONTINUOUS_FEATURES matches training
```

---

## 📊 Understanding the Output

### Risk Levels

| Probability | Risk Level | Recommendation |
|------------|------------|---------------|
| < 20% | 🟢 Low Risk | Maintain healthy lifestyle |
| 20-40% | 🟡 Moderate Risk | Lifestyle modifications, follow-up |
| 40-70% | 🟠 High Risk | Schedule diabetes screening |
| > 70% | 🔴 Very High Risk | Immediate medical attention |

### SHAP Feature Importance

Shows top 10 factors affecting risk:

**Red arrow → increases risk:**
- High BMI
- High cholesterol
- Sedentary lifestyle

**Blue arrow → decreases risk:**
- Physical activity
- Good diet
- Healthcare access

---

## 🔒 Medical Safety & Best Practices

### Critical Points

1. **This is a SCREENING tool, NOT a diagnostic tool**
   - Positive results need medical confirmation
   - Never replace clinical judgment

2. **Data Privacy**
   - Don't log sensitive patient data
   - Use only for intended medical purposes

3. **Model Monitoring**
   - Track prediction distributions
   - Monitor for model drift
   - Retrain quarterly if possible

4. **Feature Validation**
   - Always check feature values are reasonable
   - All binary features should be 0 or 1
   - BMI should be 10-100

5. **Threshold Management**
   - Default 0.30 optimized for screening
   - Can adjust based on clinical needs
   - Document any changes

---

## 📈 Model Interpretation

### Why NO SMOTE?

**Problems with SMOTE:**
- Creates synthetic patients (not real)
- Can overfit to synthetic patterns
- Often causes miscalibrated probabilities
- Can result in 99% risk for healthy patients

**Our solution:**
- Use `scale_pos_weight` (tells XGBoost to weight minority class)
- Only trains on real data
- Better calibrated probabilities
- Production-safe

### Why Calibration?

**Without calibration:**
- Model might output 0.55 probability
- But only 30% of those patients actually have diabetes

**With calibration:**
- Model outputs 0.30 probability
- Actually 30% of those patients have diabetes
- Probabilities are trustworthy

### Feature Scaling

**Continuous features (scaled):**
- BMI, MentHlth, PhysHlth, Age
- StandardScaler: mean=0, std=1

**Binary features (not scaled):**
- All others: 0 or 1 values
- Scaling would distort them

---

## 🚀 Deployment Options

### Option 1: Local Streamlit (Development)

```bash
streamlit run app/app.py
```
- Available at http://localhost:8501
- Good for testing and demo

### Option 2: Cloud Deployment

**Using Streamlit Cloud (free):**
1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect repository
4. App deploys automatically

**Using AWS/Azure/GCP:**
1. Deploy to EC2 or App Service
2. Use Docker for containerization
3. Add authentication layer

### Option 3: API Deployment

Convert `predict.py` to a REST API:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict_endpoint(patient_data: dict):
    return predict_risk(patient_data)
```

Deploy with: `uvicorn app:app`

---

## 📋 File Reference

| File | Purpose |
|------|---------|
| `train_clean.py` | Train model, save artifacts |
| `predict.py` | Load model, make predictions |
| `app/app.py` | Streamlit dashboard |
| `models/diabetes_model_calibrated.pkl` | Trained model |
| `models/scaler.pkl` | Feature scaler |
| `models/model_columns.pkl` | Feature order |
| `requirements.txt` | Dependencies |
| `WARNINGS.md` | Critical warnings |
| `README.md` | Quick reference |

---

## 🎯 Success Criteria

Your setup is complete when:

✅ `python train_clean.py` completes successfully
✅ All NO inputs test shows "PASS" and < 5% probability
✅ ROC-AUC > 0.75
✅ `streamlit run app/app.py` opens dashboard
✅ Can input patient data and get predictions
✅ SHAP explanations display correctly
✅ All artifacts saved in `models/` directory

---

## 📞 Support

For issues:
1. Check `WARNINGS.md` for common problems
2. Review error messages carefully
3. Verify all dependencies installed
4. Check feature columns and scaling
5. Ensure data format is correct

---

## 🎓 Next Steps

1. **Train Model:** `python train_clean.py`
2. **Run App:** `streamlit run app/app.py`
3. **Test with Sample Patients:** Try different inputs
4. **Verify SHAP Explanations:** Check feature importance
5. **Monitor Performance:** Track metrics over time

Good luck! 🚀
