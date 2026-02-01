# ⚠️ Important Warnings & Potential Issues

## 🚨 Critical Issues to Watch For

### 1. **SMOTE Problem (FIXED)**
**Issue**: Previous model used SMOTE, causing 99% risk for all NO inputs
**Solution**: This pipeline uses `scale_pos_weight` instead - NO SMOTE anywhere
**Status**: ✅ Fixed in clean pipeline

### 2. **Data Path**
**Warning**: Update `data_path` in `train_clean.py` if your CSV is in a different location
```python
data_path = '../diabetes_012_health_indicators_BRFSS2015.csv'  # Update this
```

### 3. **Column Ordering**
**Critical**: Always use `model_columns.pkl` for feature ordering
- Don't hardcode feature names
- Load from saved file: `joblib.load('models/model_columns.pkl')`
- Ensures consistency between training and inference

### 4. **Scaler Mismatch**
**Warning**: Must use the EXACT scaler from training
- Saved in `models/scaler.pkl`
- Don't create new scaler for inference
- Scaling must match training exactly

### 5. **All NO Inputs Test**
**Validation**: If this test fails:
- Model probabilities may be miscalibrated
- Check Brier Score (should be < 0.20)
- May need to adjust calibration method
- Threshold may need tuning

### 6. **Probability Calibration**
**Important**: CalibratedClassifierCV is critical
- Without calibration, probabilities may be unreliable
- Isotonic calibration works best for this dataset
- Recalibrate if adding new training data

### 7. **Threshold Selection**
**Note**: Default threshold is 0.30
- Optimized for 70% recall (screening standard)
- Can be adjusted based on clinical needs
- Saved in `model_metadata.pkl`

### 8. **Class Imbalance**
**Handled by**: `scale_pos_weight`
- Automatically calculated from training data
- No manual tuning needed
- More stable than SMOTE

### 9. **Feature Scaling**
**Critical**: Only scale continuous features
- Continuous: BMI, MentHlth, PhysHlth, Age
- Binary features: Don't scale (0/1 values)
- Scaling applied only to specified columns

### 10. **Model Versioning**
**Best Practice**: Save model version with artifacts
- Include training date in metadata
- Track model performance metrics
- Version control model files

## 🔍 Validation Checklist

Before deploying, verify:

- [ ] All NO inputs → LOW risk (< 5%)
- [ ] ROC-AUC > 0.75
- [ ] Brier Score < 0.20
- [ ] Feature columns match exactly
- [ ] Scaler loaded correctly
- [ ] Threshold set appropriately
- [ ] No SMOTE in inference pipeline

## 🐛 Common Errors

### Error: "Model file not found"
**Fix**: Run `train_clean.py` first

### Error: "Missing features"
**Fix**: Check feature names match `model_columns.pkl`

### Error: "All NO test failed"
**Fix**: 
1. Check model calibration
2. Verify scaler is correct
3. Review training data quality

### Error: "Probability > 50% for healthy patient"
**Fix**: 
1. Recalibrate model
2. Check feature values (should be 0/1 for binaries)
3. Verify scaling is correct

## 📋 Pre-Deployment Checklist

1. ✅ Train model with `train_clean.py`
2. ✅ Verify all NO inputs test passes
3. ✅ Check ROC-AUC > 0.75
4. ✅ Validate Brier Score < 0.20
5. ✅ Test with sample patients
6. ✅ Verify SHAP explanations work
7. ✅ Test Streamlit app
8. ✅ Document model version and date

## 🎯 Success Criteria

Model is ready for production when:
- ✅ All NO inputs → < 5% probability, LOW risk
- ✅ ROC-AUC > 0.75
- ✅ Brier Score < 0.20
- ✅ Sensible predictions for edge cases
- ✅ SHAP explanations make medical sense

