# 🧠 Employee Attrition Predictor

Predict whether an employee is likely to leave the company using machine learning.  
Built using Python, Streamlit, Logistic Regression, and XGBoost.

---

## 📌 Purpose

To assist HR teams in predicting potential employee attrition and taking early action.

---

## 📊 Dataset

- Source: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- File used: `WA_Fn-UseC_-HR-Employee-Attrition.csv`

---

## ⚙️ Features Used

- Age, Job Role, Gender, Marital Status, OverTime  
- Years at Company, Monthly Income, Job Involvement, and more

---

## 🚀 Models Used

- Logistic Regression
- XGBoost (for better accuracy)

---

## 🧪 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/karthik-88/Employee-attrition-predictor.git
cd Employee-attrition-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # (use source venv/bin/activate on Linux/Mac)

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the app
streamlit run app.py
