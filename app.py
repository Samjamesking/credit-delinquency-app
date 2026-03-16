# ============================================
# CREDIT DELINQUENCY PREDICTION APP (Streamlit)
# ============================================

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("saved_model/model.pkl")

st.set_page_config(page_title="Credit Delinquency Predictor", page_icon="💳")
st.title("💳 Credit Delinquency Prediction App")
st.write("Enter customer details below to estimate the likelihood of delinquency.")

# -----------------------------
# Input fields
# -----------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
credit_utilization = st.slider("Credit Utilization (0–1)", 0.0, 1.0, 0.3)
missed_payments = st.number_input("Missed Payments (past 6 months)", 0, 6, 0)
loan_balance = st.number_input("Loan Balance", min_value=0, value=20000)
dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Retired", "Student"])
account_tenure = st.number_input("Account Tenure (years)", 0, 40, 5)
credit_card_type = st.selectbox("Credit Card Type", ["Standard", "Gold", "Platinum", "Business", "Student"])
location = st.selectbox("Location", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
total_missed = st.number_input("Total Missed Payments (6 months)", 0, 6, 0)
total_late = st.number_input("Total Late Payments (6 months)", 0, 6, 0)

# -----------------------------
# Prepare input for prediction
# -----------------------------
input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Credit_Score': [credit_score],
    'Credit_Utilization': [credit_utilization],
    'Missed_Payments': [missed_payments],
    'Loan_Balance': [loan_balance],
    'Debt_to_Income_Ratio': [dti_ratio],
    'Employment_Status': [employment_status],
    'Account_Tenure': [account_tenure],
    'Credit_Card_Type': [credit_card_type],
    'Location': [location],
    'Total_Missed': [total_missed],
    'Total_Late': [total_late]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Delinquency"):
    # Get the feature names the model was trained on
    expected_cols = model.named_steps['preprocessor'].feature_names_in_

    # Align input_data with expected columns
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0  # add any missing column with a neutral value

    # Reorder columns to match training order
    input_data = input_data[expected_cols]

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Delinquency\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Delinquency\nProbability: {probability:.2f}")



# -----------------------------
# Run instructions
# -----------------------------
st.markdown("---")
st.caption("To run this app locally, execute:  `streamlit run app.py`")