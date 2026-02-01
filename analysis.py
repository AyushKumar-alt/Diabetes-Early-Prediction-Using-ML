import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import shap
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Load Data
data_path = 'diabetes_012_health_indicators_BRFSS2015.csv'
print(f"Loading data from {data_path}...")
try:
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File {data_path} not found. Please ensure the file is in the same directory.")
    exit()

# --- EDA Plots ---
print("\nGenerating EDA Plots...")

# 1. Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Diabetes_012', data=df)
plt.title('Distribution of Diabetes Classes')
plt.xlabel('Diabetes Class (0: No Diabetes, 1: Pre-diabetes, 2: Diabetes)')
plt.ylabel('Count')
plt.savefig('eda_target_distribution.png')
plt.close()
print("Saved eda_target_distribution.png")

# 2. Correlation Heatmap
plt.figure(figsize=(15, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.savefig('eda_correlation_heatmap.png')
plt.close()
print("Saved eda_correlation_heatmap.png")

# 3. Feature Distributions (Selected important features)
features_to_plot = ['BMI', 'Age', 'GenHlth', 'PhysHlth']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('eda_feature_distributions.png')
plt.close()
print("Saved eda_feature_distributions.png")

# 4. Bivariate Analysis: BMI vs Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='BMI', data=df)
plt.title('BMI Distribution by Diabetes Class')
plt.xlabel('Diabetes Class')
plt.ylabel('BMI')
plt.savefig('eda_bmi_vs_diabetes.png')
plt.close()
print("Saved eda_bmi_vs_diabetes.png")

# 5. Bivariate Analysis: HighBP vs Diabetes
plt.figure(figsize=(8, 6))
sns.countplot(x='Diabetes_012', hue='HighBP', data=df)
plt.title('Diabetes Class Distribution by High Blood Pressure')
plt.xlabel('Diabetes Class')
plt.legend(title='HighBP (0: No, 1: Yes)')
plt.savefig('eda_highbp_vs_diabetes.png')
plt.close()
print("Saved eda_highbp_vs_diabetes.png")


# --- Model Training and Evaluation ---
print("\nTraining Models and Generating Evaluation Plots...")

# Prepare Data
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Note: SMOTE disabled for medical-grade deployment — use sample weights instead
print("SMOTE has been disabled. Using balanced sample weights for safe training...")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 1. Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train, sample_weight=sample_weights)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)

# Evaluation Plots for LR
# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('lr_confusion_matrix.png')
plt.close()
print("Saved lr_confusion_matrix.png")

# ROC Curve (One-vs-Rest for multiclass)
# For simplicity in visualization, we can plot for each class
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob_lr[:, i])
    roc_auc[i] = roc_auc_score(y_test == i, y_prob_lr[:, i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Multi-class')
plt.legend(loc="lower right")
plt.savefig('lr_roc_curve.png')
plt.close()
print("Saved lr_roc_curve.png")

# Feature Importance for LR
feature_importance_lr = abs(lr.coef_[0])
feature_importance_lr = 100.0 * (feature_importance_lr / feature_importance_lr.max())
sorted_idx = np.argsort(feature_importance_lr)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 10))
plt.barh(pos, feature_importance_lr[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Logistic Regression Feature Importance')
plt.savefig('lr_feature_importance.png')
plt.close()
print("Saved lr_feature_importance.png")


# 2. XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)

# Evaluation Plots for XGBoost
# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens')
plt.title('XGBoost Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('xgb_confusion_matrix.png')
plt.close()
print("Saved xgb_confusion_matrix.png")

# ROC Curve
fpr_xgb = dict()
tpr_xgb = dict()
roc_auc_xgb = dict()
for i in range(n_classes):
    fpr_xgb[i], tpr_xgb[i], _ = roc_curve(y_test == i, y_prob_xgb[:, i])
    roc_auc_xgb[i] = roc_auc_score(y_test == i, y_prob_xgb[:, i])

plt.figure(figsize=(10, 8))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_xgb[i], tpr_xgb[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc_xgb[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Multi-class')
plt.legend(loc="lower right")
plt.savefig('xgb_roc_curve.png')
plt.close()
print("Saved xgb_roc_curve.png")

# Feature Importance for XGBoost
plt.figure(figsize=(12, 10))
xgb.plot_importance(xgb_model, max_num_features=20, height=0.5)
plt.title('XGBoost Feature Importance')
plt.savefig('xgb_feature_importance.png')
plt.close() # plot_importance creates its own figure, but good to be safe
print("Saved xgb_feature_importance.png")

# SHAP Summary
# print("Generating SHAP Summary Plot (this might take a while)...")
# # Using a sample for SHAP to speed it up
# X_sample = X_test_scaled[:1000]
# explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(X_sample)

# plt.figure()
# shap.summary_plot(shap_values, X_sample, feature_names=X.columns, show=False)
# plt.title('SHAP Summary Plot (XGBoost)')
# plt.savefig('xgb_shap_summary.png', bbox_inches='tight')
# plt.close()
# print("Saved xgb_shap_summary.png")

print("\nAnalysis Complete. All plots saved.")
