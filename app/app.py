import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
st.sidebar.title("🏦 Loan Default Predictor")

st.sidebar.markdown("""
### About

This application predicts the probability of loan default using an XGBoost machine learning model trained on the Home Credit Default Risk dataset.

**Model:** XGBoost

**Features Used:** 9

**Author:** Mohit Singh
""")
st.divider()

st.caption(
    "Built by Mohit Singh |  XGBoost + Streamlit"
)
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "deployment_model.pkl"

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🏦 Loan Default Prediction")
st.markdown(
    """
Predict the probability that a customer will default on a loan using an
XGBoost model trained on the Home Credit Default Risk dataset.
"""
)

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "Annual Income (₹)",
        min_value=0.0,
        value=150000.0,
        step=1000.0
    )

    credit = st.number_input(
        "Credit Amount (₹)",
        min_value=0.0,
        value=500000.0,
        step=1000.0
    )

    goods_price = st.number_input(
        "Goods Price (₹)",
        min_value=0.0,
        value=450000.0,
        step=1000.0
    )

    annuity = st.number_input(
        "Annuity (₹)",
        min_value=0.0,
        value=25000.0,
        step=100.0
    )

    age = st.number_input(
        "Age (Years)",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:

    employment_years = st.number_input(
        "Employment Years",
        min_value=0,
        value=5
    )

    

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Loan Default Risk", use_container_width=True):

    # ---------------- Validation ----------------

    if income <= 0:
        st.error("Annual Income must be greater than 0.")
        st.stop()

    if credit <= 0:
        st.error("Credit Amount must be greater than 0.")
        st.stop()

    if goods_price <= 0:
        st.error("Goods Price must be greater than 0.")
        st.stop()

    if annuity <= 0:
        st.error("Annuity must be greater than 0.")
        st.stop()

    if employment_years > age - 18:
        st.error(
            "Employment years cannot exceed working age."
        )
        st.stop()
    if credit > income * 10:
        st.warning(
            "⚠️ Credit amount is extremely high compared to annual income."
        )

    if annuity > income * 0.6:
        st.warning(
            "⚠️ Annual loan repayment appears high compared to annual income."
        )

    if goods_price < credit * 0.5:
        st.warning(
            "⚠️ Credit amount is much higher than the declared goods price."
        )
    # ---------------- Feature Engineering ----------------

    days_birth = -age * 365
    days_employed = -employment_years * 365

    credit_goods_ratio = credit / goods_price
    credit_income_ratio = credit / income
    employment_age_ratio = abs(days_employed) / abs(days_birth)

    input_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [income],
        "AMT_CREDIT": [credit],
        "AMT_GOODS_PRICE": [goods_price],
        "AMT_ANNUITY": [annuity],
        "DAYS_BIRTH": [days_birth],
        "DAYS_EMPLOYED": [days_employed],
        "CREDIT_INCOME_RATIO": [credit_income_ratio],
        "CREDIT_GOODS_RATIO": [credit_goods_ratio],
        "EMPLOYMENT_AGE_RATIO": [employment_age_ratio]
    })

        # ---------------- Prediction ----------------

    probability = model.predict_proba(input_df)[0][1]

    # ---------------- Risk Level ----------------

    if probability < 0.10:
        risk = "🟢 Low Risk"
        recommendation = "Eligible for normal processing."

    elif probability < 0.30:
        risk = "🟡 Medium Risk"
        recommendation = "Requires additional verification."

    else:
        risk = "🔴 High Risk"
        recommendation = "High default risk. Manual review recommended."

    # ---------------- Results ----------------

    st.divider()

    st.header("Prediction Result")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Default Probability",
            f"{probability:.2%}"
        )

    with c2:
        st.metric(
            "Risk Level",
            risk
        )

    st.info(f"📌 Recommendation: {recommendation}")

    st.subheader("Financial Ratios")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Credit / Income",
            f"{credit_income_ratio:.2f}"
        )

    with r2:
        st.metric(
            "Credit / Goods",
            f"{credit_goods_ratio:.2f}"
        )

    with r3:
        st.metric(
            "Employment / Age",
            f"{employment_age_ratio:.2f}"
        )

    with st.expander("View Model Input"):
        st.dataframe(input_df)