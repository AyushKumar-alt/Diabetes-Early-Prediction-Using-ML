🩺 AI-Based Diabetes Risk Prediction System
Early Detection of Diabetes & Pre-Diabetes using Machine Learning
📌 Project Overview

Diabetes is one of the most rapidly increasing chronic diseases worldwide, affecting more than 537 million people. Early detection is crucial for prevention, yet nearly 50% of cases remain undiagnosed due to lack of awareness and access to clinical testing.

This project presents a Machine Learning–based screening tool that predicts the risk probability of diabetes and pre-diabetes using non-invasive survey-based indicators such as BMI, lifestyle habits, mental & physical health scores, and general health assessment.

The system identifies Healthy vs At-Risk (Pre-Diabetic + Diabetic) individuals, provides personalized recommendations, and automatically generates PDF patient reports.

🎯 Key Features

🧠 Calibrated XGBoost Machine Learning Model

📈 Optimized decision threshold (0.30) for medical sensitivity

⚕️ Medical-Boost Logic for clinical safety

📑 Auto-Generated PDF Reports

📊 Visual Analytics: Confusion Matrices, ROC-AUC, Feature Importance

🌐 Streamlit-based UI for real-time screening

📂 Fully reproducible ML pipeline

🤝 Built for real-world clinical utility & early screening

🧪 Machine Learning Models & Results
✔ Final Selected Model: Calibrated XGBoost (Binary: Healthy vs At-Risk)
Metric	Value
ROC-AUC	0.81
Recall (At-Risk)	0.69
Precision (At-Risk)	0.45
F1-Score (At-Risk)	0.49

Optimized using threshold tuning (0.30) to reduce false negatives
Ideal for mass screening rather than diagnostic accuracy.

Baseline Comparison
Model	Recall (At-Risk)	F1-Score	Notes
Logistic Regression	0.49	0.47	Linear model baseline
XGBoost (final)	0.69	0.49	Best clinical performance
3-Class Model	Failed (Pre-DM recall ≈ 0%)	—	Abandoned due to class overlap
📂 Project Structure
📦 Diabetes-Risk-Prediction
├── data/
│   ├── cleaned_health_data.csv
├── models/
│   ├── xgb_calibrated.pkl
│   ├── scaler.pkl
├── diabetes_risk_app/
│   ├── app.py
│   ├── pdf_generator.py
│   ├── recommender.py
│   ├── utils.py
├── results/
│   ├── confusion_matrices/
│   ├── roc_auc_curve.png
└── ML_Project.ipynb

🚀 How to Run the Application
1️⃣ Create Virtual Environment
python -m venv venv
source venv/Scripts/activate   # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run diabetes_risk_app/app.py

🩻 Model Deployment Logic
User Inputs → Preprocessing → Calibrated XGBoost Model →
Medical Boost Adjustment → Risk Prediction + PDF Report

🧪 Medical Boost

Used to increase clinical safety by adding a boost (up to +20%) in cases with:

BMI > 35

Age ≥ 60

History of heart disease or stroke

Poor general health rating

📊 Visualizations Included

ROC-AUC Curve

Confusion Matrix (Normalized)

Feature Importance Plot

Probability Distribution vs Threshold

3-class failure comparison chart

📦 Dataset Details

CDC BRFSS 2015 Public Health Dataset

229,772 rows × 22 non-invasive features

Real-world imbalance handled with threshold tuning (not SMOTE)

🧠 Tech Stack
Category	Tools
ML Models	XGBoost, Logistic Regression
Deployment	Streamlit
Explainability	SHAP
Report Generation	ReportLab
Scaling	StandardScaler
Metrics	ROC-AUC, F1, Precision, Confusion Matrix
🛠 Future Improvements

Add deep learning models for comparative performance

Integrate real clinical lab data (HbA1c, Fasting Glucose)

Build mobile-friendly version

Add multi-language support

Deploy on cloud (AWS / Azure)

📜 License

MIT License — Free for research, education & development.

👥 Team
Name	ID
Ayush Kumar	2023UG000116
Pruthviraj Shinde	2023UG000103
Chandrapal	2023UG000118
🙌 Acknowledgements

CDC BRFSS Open Dataset (U.S. Centres for Disease Control)

Vidyashilp University — School of Computational & Data Sciences

⭐ If you found this helpful

Please star the repository ⭐ to support our research
