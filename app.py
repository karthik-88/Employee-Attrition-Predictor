# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and metadata
model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("👨‍💼 Employee Attrition Predictor")
st.write("Predict if an employee is likely to leave the company.")

# Input form
with st.form("attrition_form"):
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    distance_from_home = st.slider("Distance from Home (km)", 1, 30, 10)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)

    submitted = st.form_submit_button("Predict Attrition")

if submitted:
    input_dict = {
        "Age": age,
        "Gender": gender,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "YearsAtCompany": years_at_company,
        "DistanceFromHome": distance_from_home,
        "MonthlyIncome": monthly_income,
    }

    # Add all missing features with default zero
    full_input = {col: 0 for col in columns}
    for col in input_dict:
        if col in encoders:
            full_input[col] = encoders[col].transform([input_dict[col]])[0]
        else:
            full_input[col] = input_dict[col]

    # Convert to DataFrame
    input_df = pd.DataFrame([full_input])[columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"⚠️ Likely to leave the company. (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Unlikely to leave. (Confidence: {1 - probability:.2%})")
