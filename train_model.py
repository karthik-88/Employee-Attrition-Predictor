# train_model.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Step 1: Load the dataset ===
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(DATA_PATH)

# === Step 2: Drop irrelevant columns ===
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# === Step 3: Encode target variable (Attrition: Yes=1, No=0) ===
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# === Step 4: Label encode categorical columns ===
cat_cols = df.select_dtypes(include='object').columns.tolist()
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoder for use in app

# === Step 5: Split into features and label ===
X = df.drop('Attrition', axis=1)
y = df['Attrition']
feature_columns = X.columns.tolist()

# === Step 6: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 7: Train model (XGBoost Classifier) ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# === Step 8: Evaluate ===
y_pred = model.predict(X_test)
print("✅ Model trained.")
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# === Step 9: Save model and metadata ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(feature_columns, "models/columns.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("\n Model, columns, and encoders saved to 'models/' folder.")
