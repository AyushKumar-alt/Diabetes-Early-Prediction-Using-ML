# Installation Instructions

## ✅ Most Packages Already Installed

The good news: Most required packages are already installed:
- ✅ xgboost
- ✅ scikit-learn
- ✅ pandas
- ✅ numpy
- ✅ streamlit
- ✅ plotly
- ✅ shap
- ✅ joblib

## ⚠️ PyArrow Issue (Optional)

`pyarrow` failed to build because it requires `cmake`. This is **optional** for Streamlit.

### Option 1: Skip PyArrow (Recommended)
Streamlit will work without it. Just proceed with training:
```bash
cd project
python train_clean.py
```

### Option 2: Install Pre-built PyArrow
Try installing a pre-built wheel:
```bash
pip install pyarrow --only-binary :all:
```

### Option 3: Install CMake (if you need PyArrow)
1. Download CMake from: https://cmake.org/download/
2. Install it
3. Then: `pip install pyarrow`

## 🚀 Quick Start

Since most packages are installed, you can proceed:

```bash
# Navigate to project directory
cd project

# Train the model
python train_clean.py

# Test the pipeline
python test_pipeline.py

# Run Streamlit app
streamlit run app/app.py
```

The PyArrow issue won't prevent you from training or using the model!

