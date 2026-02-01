# Clean Diabetes Risk Prediction Pipeline

Production-ready ML pipeline **WITHOUT SMOTE** - uses `scale_pos_weight` and probability calibration.

## 🎯 Key Features

- ✅ **NO SMOTE** - Uses `scale_pos_weight` for class imbalance
- ✅ **Calibrated Probabilities** - `CalibratedClassifierCV` for reliable probabilities
- ✅ **Proper Thresholding** - Optimized threshold (0.30)
- ✅ **Validated** - All NO inputs → LOW risk (< 5%)
- ✅ **Production-Ready** - Clean, modular, reproducible

## 📁 Project Structure

```
project/
├── data/                    # Data files
├── models/                  # Trained models and artifacts
│   ├── diabetes_model_calibrated.pkl
│   ├── scaler.pkl
│   ├── model_columns.pkl
│   ├── model_metadata.pkl
│   └── training_metrics.pkl
├── notebooks/               # Jupyter notebooks
├── app/
│   └── app.py              # Streamlit application
├── train_clean.py          # Training script (NO SMOTE)
├── predict.py              # Prediction module
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
python train_clean.py
```

This will:
- Load and clean data
- Train XGBoost with `scale_pos_weight` (NO SMOTE)
- Calibrate probabilities
- Validate that all NO inputs give LOW risk
- Save all artifacts to `models/`

### 3. Run Streamlit App

```bash
streamlit run app/app.py
```

### 4. Test Prediction

```python
from predict import predict_risk

patient_data = {
    'HighBP': 0,
    'HighChol': 0,
    # ... (all features)
}

result = predict_risk(patient_data)
print(f"Risk: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
```

## ⚠️ Important Notes

### Why NO SMOTE?

SMOTE can cause issues:
- **Synthetic samples** may not represent real patients
- **Overfitting** to synthetic patterns
- **Miscalibrated probabilities** (e.g., 99% risk for healthy patients)
- **Not suitable for production** inference

### Solution: `scale_pos_weight`

Instead of SMOTE, we use:
- `scale_pos_weight = healthy_count / atrisk_count`
- This tells XGBoost to pay more attention to the minority class
- **No synthetic data** - only real training samples
- **Better calibrated** probabilities

### Probability Calibration

- Uses `CalibratedClassifierCV` with isotonic calibration
- Ensures probabilities are reliable (e.g., 30% means 30% chance)
- Critical for medical applications

## 📊 Model Performance

After training, you'll see:
- ROC-AUC score
- F1, Precision, Recall
- Brier Score (calibration quality)
- Validation: All NO inputs → LOW risk test

## 🔍 Validation

The training script automatically tests:
- **All NO inputs** should give < 5% probability
- **Risk level** should be "Low Risk"
- If this fails, model needs recalibration

## 📝 Feature Columns

The model expects exactly these 21 features in this order:

1. HighBP, HighChol, CholCheck, BMI, Smoker, Stroke
2. HeartDiseaseorAttack, PhysActivity, Fruits, Veggies
3. HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth
4. MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

Saved in `models/model_columns.pkl` for consistency.

## 🎯 Risk Levels

| Probability | Risk Level | Recommendation |
|------------|------------|---------------|
| < 20% | Low Risk | Maintain healthy lifestyle |
| 20-40% | Moderate Risk | Lifestyle modifications, follow-up in 6 months |
| 40-70% | High Risk | Schedule diabetes screening |
| > 70% | Very High Risk | Immediate medical testing |

## ⚠️ Potential Issues & Warnings

### 1. **Data Path**
- Update `data_path` in `train_clean.py` if CSV is in different location
- Default: `../diabetes_012_health_indicators_BRFSS2015.csv`

### 2. **Model Directory**
- Ensure `models/` directory exists before training
- Script will create it automatically

### 3. **Column Ordering**
- Always use `model_columns.pkl` for inference
- Don't hardcode feature order

### 4. **Scaler**
- Must use the same scaler from training
- Saved in `models/scaler.pkl`

### 5. **Threshold**
- Default: 0.30 (optimized for screening)
- Can be adjusted in `predict.py` or metadata

### 6. **All NO Input Test**
- If this fails, check:
  - Model calibration quality
  - Threshold may need adjustment
  - Feature scaling may be incorrect

### 7. **SHAP Explanations**
- Requires `shap` package
- May be slow for large datasets
- Optional feature

## 🔧 Troubleshooting

### "Model not found" error
→ Run `train_clean.py` first to generate model artifacts

### "All NO inputs test FAILED"
→ Model may need recalibration or threshold adjustment

### Import errors
→ Install dependencies: `pip install -r requirements.txt`

### Probability seems wrong
→ Check that scaler and feature columns match training

## 📚 Best Practices

1. **Always validate** with all NO inputs test
2. **Use saved artifacts** (scaler, columns) from training
3. **Never use SMOTE** on inference data
4. **Monitor calibration** with Brier Score
5. **Document threshold** used for predictions

## 🎉 Benefits of This Approach

✅ **Medically Valid** - No synthetic data
✅ **Well-Calibrated** - Reliable probabilities
✅ **Production-Safe** - Reproducible and consistent
✅ **Transparent** - Clear feature contributions
✅ **Validated** - All NO inputs → LOW risk

