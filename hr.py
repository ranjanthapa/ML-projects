import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = 'https://drive.google.com/uc?id=1g1nwk4k-h9FceEHKZc8ocfu_xp3xnZ8R'
data = pd.read_csv(url)

data['age'] = data['age'].fillna(data['age'].median())

print("\nRemaining missing values:")
print(data.isnull().sum())

data.dropna(inplace=True)
le_department = LabelEncoder()
le_salary = LabelEncoder()

data['Department'] = le_department.fit_transform(data['Department'])
data['salary'] = le_salary.fit_transform(data['salary'])

X = data.drop(columns=['left'])
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_reg_pred = log_reg.predict(X_test_scaled)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)


log_reg_acc = accuracy_score(y_test, log_reg_pred)

svm_acc = accuracy_score(y_test, svm_pred)

data.to_csv('cleaned_data.csv', index=False)
