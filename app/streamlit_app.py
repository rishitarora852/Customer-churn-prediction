"""
Customer Churn Prediction — Streamlit App
==========================================
A retention decision-support tool for telecom customer service teams.

Run locally with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st


# ============================================================
# Page configuration — must be the first Streamlit call
# ============================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Header
# ============================================================
st.title("Customer Churn Prediction & Retention Strategy")
st.markdown(
    """
    Predict whether a telecom customer is likely to churn,
    explain *why* using SHAP values, and recommend a retention action
    based on their estimated Customer Lifetime Value (CLV).
    """
)
st.divider()


# ============================================================
# Sidebar — Customer Input Form (placeholder widgets)
# ============================================================
with st.sidebar:
    st.header("Customer Profile")
    st.caption("Enter the customer's details below to generate a churn prediction.")

    # Demographics
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Has a partner?", ["No", "Yes"])
    dependents = st.selectbox("Has dependents?", ["No", "Yes"])

    # Account info
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input(
        "Monthly charges (€)",
        min_value=0.0, max_value=200.0, value=70.0, step=0.5
    )

    predict_button = st.button("Predict churn risk", type="primary", use_container_width=True)


# ============================================================
# Main panel — Results (placeholder)
# ============================================================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Churn Prediction")
    if predict_button:
        st.info("Prediction logic will be wired up in Phase 3.")
        st.write(f"**Captured input — gender:** {gender}")
        st.write(f"**Captured input — tenure:** {tenure} months")
        st.write(f"**Captured input — contract:** {contract}")
        st.write(f"**Captured input — monthly charges:** €{monthly_charges:.2f}")
    else:
        st.caption("Enter customer details in the sidebar and click *Predict* to see the result.")

with col_right:
    st.subheader("Why this prediction?")
    st.caption("SHAP-based explanation will appear here once the model is wired in.")


# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "Built by Rishit Arora — MSc Business Analytics, University of Galway. "
    "Model: tuned XGBoost on IBM Telco Customer Churn dataset (7,043 customers, 37 features). "
    "[GitHub](https://github.com/rishitarora852/Customer-churn-prediction)"
)