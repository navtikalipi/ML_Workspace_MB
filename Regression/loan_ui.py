import streamlit as st
import requests

# ---------------------------------------
# Page configuration
# ---------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center;'>🏦 Loan Approval Prediction</h1>
    <p style='text-align: center; color: grey;'>
    Predict loan approval using AI-powered credit assessment
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------
# Input form
# ---------------------------------------
with st.form("loan_form"):
    st.subheader("📄 Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=500)
        LoanAmount = st.number_input("Loan Amount", min_value=0, step=10)
        Credit_History = st.selectbox("Credit History", [1, 0])

    with col2:
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, step=500)
        Loan_Amount_Term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360])

    col3, col4 = st.columns(2)

    with col3:
        Married = st.selectbox("Marital Status", ["Yes", "No"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col4:
        Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
        Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("🔍 Predict Loan Status")

# ---------------------------------------
# API Call
# ---------------------------------------
if submitted:
    payload = {
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Married": Married,
        "Self_Employed": Self_Employed,
        "Education": Education,
        "Property_Area": Property_Area
    }

    with st.spinner("🔄 Analyzing loan eligibility..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()

                st.divider()
                st.subheader("📊 Prediction Result")

                # Status
                if result["loan_status"] == "Approved":
                    st.success("✅ **Loan Approved**")
                else:
                    st.error("❌ **Loan Rejected**")

                # Probability
                probability = result["approval_probability"]

                st.metric(
                    label="Approval Probability",
                    value=f"{probability * 100:.2f}%"
                )

                st.progress(probability)

            else:
                st.error(f"❌ API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("🚫 Cannot connect to prediction API (is FastAPI running?)")

        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")


import sqlite3
import pandas as pd
st.subheader("📜 Recent Predictions")

conn = sqlite3.connect("predictions.db")
history_df = pd.read_sql(
    "SELECT * FROM loan_predictions ORDER BY created_at DESC LIMIT 5",
    conn
)
conn.close()

st.dataframe(history_df)


# ---------------------------------------
# Footer
# ---------------------------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: grey;'>Powered by FastAPI + Streamlit</p>",
    unsafe_allow_html=True
)
