import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("files/water_potability.csv")

df.fillna(df.median(), inplace=True)

X = df.drop(columns=['Potability'])
y = df['Potability']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion matrices
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
svm_acc = accuracy_score(y_test, y_pred_svm)
print(f"Logistic Regression Accuracy: {log_reg_acc}")
print(f"SVM Accuracy: {svm_acc}")

log_reg_roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
svm_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

log_reg_fpr, log_reg_tpr, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(log_reg_fpr, log_reg_tpr, label=f"Logistic Regression (AUC = {log_reg_roc_auc:.2f})")
plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Dashed line for random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
