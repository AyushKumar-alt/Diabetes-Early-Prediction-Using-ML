# 🎯 New Features Guide

## Quick Reference for All New Features

### 1. 🔍 Explain My Prediction (XAI)

**What it does**: Shows exactly why the model predicted a specific risk level.

**How to use**:
1. After getting a prediction, scroll down to "Explain My Risk Score (XAI)" section
2. Click "🔍 Explain My Prediction" button
3. View:
   - Top 5 factors affecting your risk
   - Feature contribution bar chart
   - Detailed table with all contributions

**Example Output**:
```
🔺 HighBP (High Blood Pressure) increases your risk (+0.123)
🔺 BMI (Body Mass Index) increases your risk (+0.091)
🔻 PhysActivity (Physical Activity) reduces your risk (-0.031)
```

### 2. 📄 PDF Report Generation

**What it does**: Creates a professional PDF report you can download and share.

**How to use**:
1. After prediction, scroll to "Export Report" section
2. Click "📄 Download PDF Report"
3. Click "Download PDF" button that appears
4. Save the PDF file

**Includes**:
- Patient information
- Risk assessment
- Recommendations
- Feature contributions (if SHAP was generated)
- Professional formatting

### 3. 📊 Feature Contribution Chart

**What it does**: Visual bar chart showing which features contribute most to the prediction.

**How to use**:
- Automatically shown when you click "Explain My Prediction"
- Red bars = features that increase risk
- Green bars = features that decrease risk

### 4. 🧭 Enhanced Confidence Levels

**What it does**: Better confidence calculation with 4 levels.

**Levels**:
- 🟢 **Low** (< 30%): Low confidence
- 🟡 **Moderate** (30-50%): Moderate confidence
- 🟠 **High** (50-70%): High confidence
- 🔴 **Very High** (> 70%): Very high confidence

### 5. 📋 Detailed Feature Contributions Table

**What it does**: Complete table showing all feature contributions with descriptions.

**Shows**:
- Feature name
- Human-readable description
- SHAP contribution value
- Effect (increases/reduces risk)

## 🎨 Visual Features

### Risk Gauge Meter
- Color-coded probability gauge
- Shows risk zones (Low/Moderate/High/Very High)
- Threshold indicator at 30%

### Feature Importance Charts
- Top 10 most important features globally
- Individual prediction contributions
- Interactive Plotly charts

## 📥 Export Options

### Text Report (TXT)
- Simple markdown format
- Quick download
- Easy to read

### PDF Report
- Professional formatting
- Includes SHAP explanations
- Suitable for clinical use
- Print-ready

## 🔧 For Developers

### Using Calibration Tools

```python
from utils.calibration import calculate_brier_score, plot_calibration_curve

# Calculate Brier Score
brier = calculate_brier_score(y_test, y_proba)
print(f"Brier Score: {brier}")  # Lower is better

# Plot calibration curve
fig = plot_calibration_curve(y_test, y_proba, n_bins=10)
plt.show()
```

### Using Reliability Test

```python
from utils.calibration import prediction_reliability

reliability = prediction_reliability(model, X_test, y_test, runs=20)
print(f"Reliability: {reliability['reliability_level']}")
print(f"Std Dev: {reliability['std_dev']:.4f}")
```

### Using SHAP Explanations

```python
from utils.explainability import get_patient_shap_explanation, shap_bar_chart

# Get explanations
explanations = get_patient_shap_explanation(model, patient_data, top_n=10)

# Generate chart
fig = shap_bar_chart(model, patient_data, top_n=10, save_path="chart.png")
```

## 📚 Understanding the Output

### SHAP Values
- **Positive values** = Feature increases risk
- **Negative values** = Feature decreases risk
- **Larger absolute value** = Stronger effect

### Confidence Levels
- Based on probability distance from threshold
- Helps assess prediction reliability
- Useful for clinical decision-making

### Risk Levels
- **Low Risk** (< 20%): Maintain healthy lifestyle
- **Moderate Risk** (20-40%): Lifestyle modifications recommended
- **High Risk** (40-70%): Schedule diabetes screening
- **Very High Risk** (> 70%): Immediate medical testing

## ⚠️ Important Notes

1. **SHAP explanations** require the `shap` package
2. **PDF generation** requires the `reportlab` package
3. **Charts** work with Plotly or Matplotlib
4. All features gracefully handle missing dependencies

## 🚀 Next Steps

1. Install any missing dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app/streamlit_app.py`
3. Try the "Explain My Prediction" feature
4. Generate a PDF report
5. Explore the feature contributions

