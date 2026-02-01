# 🚀 Quick Start Guide

## Step 1: Install Dependencies

```bash
cd project
pip install -r requirements.txt
```

## Step 2: Prepare Data

Copy your CSV file to `project/data/`:
```bash
# From project root
cp ../diabetes_012_health_indicators_BRFSS2015.csv data/
```

Or update the path in `train_clean.py` (line ~430).

## Step 3: Train Model

```bash
python train_clean.py
```

This will:
- ✅ Load and clean data
- ✅ Train XGBoost with `scale_pos_weight` (NO SMOTE)
- ✅ Calibrate probabilities
- ✅ Test that all NO inputs → LOW risk
- ✅ Save model artifacts to `models/`

**Expected output:**
```
✅ TRAINING COMPLETE
ROC-AUC: 0.81xx
F1 Score: 0.49xx
All NO test: PASS (< 5%)
```

## Step 4: Test Pipeline

```bash
python test_pipeline.py
```

Should show:
```
✅ PASS: All NO inputs correctly predict LOW risk
```

## Step 5: Run Streamlit App

```bash
streamlit run app/app.py
```

Open browser at `http://localhost:8501`

## ✅ Success Criteria

- [ ] Training completes without errors
- [ ] All NO inputs test passes (< 5% probability)
- [ ] ROC-AUC > 0.75
- [ ] Streamlit app runs
- [ ] Predictions make sense

## ⚠️ Troubleshooting

### "Data file not found"
→ Update `data_path` in `train_clean.py` or copy CSV to `data/`

### "Model not found" (when running app)
→ Run `train_clean.py` first

### "All NO test failed"
→ Model may need recalibration - check training output

