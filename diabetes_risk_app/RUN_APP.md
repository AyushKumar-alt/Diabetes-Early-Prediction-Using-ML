# 🚀 How to Run the Diabetes Risk Prediction App

## Method 1: Quick Start (Recommended)

### Step 1: Open Terminal in VS Code
1. Open VS Code in the `diabetes_risk_app` folder
2. Press `` Ctrl + ` `` (backtick) to open terminal, OR
3. Go to **Terminal → New Terminal**

### Step 2: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

This installs:
- streamlit (web framework)
- xgboost (ML model)
- pandas, numpy (data processing)
- plotly (visualizations)
- shap (explanations)

### Step 3: Run the App
```bash
streamlit run app/streamlit_app.py
```

**That's it!** The app will:
- Start a local server
- Automatically open in your browser at `http://localhost:8501`
- Show the Diabetes Risk Prediction interface

---

## Method 2: Using Command Prompt/PowerShell

### Step 1: Navigate to the App Folder
```powershell
cd C:\Users\calpo\Downloads\ML_Diabetes\diabetes_risk_app
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the App
```powershell
streamlit run app/streamlit_app.py
```

---

## 📱 Using the App

Once the app opens in your browser:

1. **Fill in Patient Data** (left sidebar):
   - Demographics (Age, Sex, Education, Income)
   - Health Conditions (Blood Pressure, Cholesterol, etc.)
   - Physical Health (BMI, General Health, etc.)
   - Lifestyle Factors (Exercise, Smoking, Diet)
   - Healthcare Access

2. **Click "🔍 Predict Risk"** button

3. **View Results**:
   - Risk Level (Low/Moderate/High/Very High)
   - Probability percentage
   - Personalized recommendation
   - Feature importance charts
   - SHAP explanations

---

## 🧪 Test Before Running (Optional)

Test that everything works:
```bash
python test_prediction.py
```

Expected output:
```
✅ Prediction successful!
Results:
  Risk Level: [some level]
  Prediction: [Healthy/At-Risk]
  ...
```

---

## ⚠️ Troubleshooting

### Issue: "streamlit: command not found"
**Solution**: Install streamlit
```bash
pip install streamlit
```

### Issue: "ModuleNotFoundError"
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Model file not found"
**Solution**: Verify the model exists
```bash
# Check if file exists
dir models\bst_binary.json
```

### Issue: Port 8501 already in use
**Solution**: Use a different port
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Issue: Browser doesn't open automatically
**Solution**: Manually open browser and go to:
```
http://localhost:8501
```

---

## 🛑 Stopping the App

- Press `Ctrl + C` in the terminal
- Or close the terminal window

---

## 📝 Quick Reference

| Action | Command |
|--------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run app | `streamlit run app/streamlit_app.py` |
| Test system | `python test_prediction.py` |
| Stop app | `Ctrl + C` |

---

## 🎯 Next Steps

- Read `README.md` for full documentation
- Check `QUICKSTART.md` for more details
- Customize the app in `app/streamlit_app.py`

