"""
Customer Churn Prediction — Streamlit App
==========================================
A retention decision-support tool for telecom customer service teams.

Run locally with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shap


# ============================================================
# Page configuration — MUST be the first Streamlit call
# ============================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Paths
# ============================================================
APP_DIR = Path(__file__).parent
REPO_ROOT = APP_DIR.parent
MODELS_DIR = REPO_ROOT / "models"


# ============================================================
# Cached artefact loaders
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "xgb_churn_model.pkl")


@st.cache_resource
def load_feature_columns():
    return joblib.load(MODELS_DIR / "feature_columns.pkl")


@st.cache_resource
def load_clv_reference():
    return joblib.load(MODELS_DIR / "clv_reference.pkl")


@st.cache_resource
def load_scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")


model = load_model()
feature_columns = load_feature_columns()
clv_reference = load_clv_reference()
scaler = load_scaler()

@st.cache_resource
def load_explainer(_model):
    """Initialise SHAP TreeExplainer for the XGBoost model."""
    return shap.TreeExplainer(_model)


explainer = load_explainer(model)


# ============================================================
# Feature Transformation
# ============================================================
def transform_user_input(
    gender, senior, partner, dependents, tenure, contract,
    paperless, payment_method, monthly_charges,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    feature_columns
):
    """Convert sidebar inputs into a 37-column DataFrame matching training schema."""
    total_charges = monthly_charges * max(tenure, 1)
    senior_int = 1 if senior == "Yes" else 0

    service_addons = [
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies
    ]
    service_bundle_depth = sum(1 for s in service_addons if s == "Yes")

    price_sensitivity = monthly_charges / (tenure + 1)
    auto_payment = 1 if "automatic" in payment_method else 0
    charge_tenure_ratio = monthly_charges / max(tenure, 1)

    if tenure <= 3:
        tenure_phase = "Trial"
    elif tenure <= 12:
        tenure_phase = "Settling"
    elif tenure <= 24:
        tenure_phase = "Active"
    else:
        tenure_phase = "Renewal_Cliff"

    features = {
        "SeniorCitizen": senior_int,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "service_bundle_depth": service_bundle_depth,
        "price_sensitivity": price_sensitivity,
        "auto_payment": auto_payment,
        "charge_tenure_ratio": charge_tenure_ratio,
        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
        "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
        "tenure_phase_Renewal_Cliff": 1 if tenure_phase == "Renewal_Cliff" else 0,
        "tenure_phase_Settling": 1 if tenure_phase == "Settling" else 0,
        "tenure_phase_Trial": 1 if tenure_phase == "Trial" else 0,
    }

    df = pd.DataFrame([features])
    df = df[feature_columns]
    return df


# ============================================================
# Prediction & Business Logic Helpers
# ============================================================
def predict_churn(input_df, model, threshold=0.38):
    """Returns (probability, is_flagged_churner)."""
    proba = model.predict_proba(input_df)[0][1]
    is_churner = proba >= threshold
    return float(proba), bool(is_churner)


def calculate_clv(monthly_charges, tenure, contract, clv_reference):
    """Customer Lifetime Value using the notebook's formula."""
    avg_tenure_by_contract = clv_reference["avg_tenure_by_contract"]
    expected_total_tenure = avg_tenure_by_contract.get(contract, 21.0)
    remaining_tenure = max(expected_total_tenure - tenure, 1)
    return monthly_charges * remaining_tenure
    

def assign_segment(churn_probability, clv, clv_median=134.15):
    """Value × Risk segmentation."""
    high_value = clv >= clv_median
    high_risk = churn_probability >= 0.5

    if high_value and high_risk:
        return "Priority Save"
    elif high_value and not high_risk:
        return "Nurture"
    elif not high_value and high_risk:
        return "Low-Cost Nudge"
    else:
        return "Monitor"


def recommend_action(segment):
    """Map a segment to a retention recommendation."""
    actions = {
        "Priority Save": {
            "intervention": "Personal call + custom retention offer",
            "cost": 80,
            "rationale": "High future value AND high churn risk — worth significant investment to retain.",
            "tone": "error",
        },
        "Nurture": {
            "intervention": "Loyalty reward + early renewal reminder",
            "cost": 10,
            "rationale": "High future value but low risk — keep them engaged with low-cost touchpoints.",
            "tone": "success",
        },
        "Low-Cost Nudge": {
            "intervention": "Automated email + small incentive",
            "cost": 20,
            "rationale": "Lower future value but at risk — automate retention to protect margin.",
            "tone": "warning",
        },
        "Monitor": {
            "intervention": "No active intervention — standard service",
            "cost": 0,
            "rationale": "Low risk and lower future value — no proactive action needed.",
            "tone": "info",
        },
    }
    return actions[segment]


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
# Sidebar — Customer Input Form
# ============================================================
with st.sidebar:
    st.header("Customer Profile")
    st.caption("Enter the customer's details to generate a churn prediction.")

    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Has a partner?", ["No", "Yes"])
    dependents = st.selectbox("Has dependents?", ["No", "Yes"])

    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing?", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly charges (€)", 0.0, 200.0, 70.0, 0.5)

    st.subheader("Services")
    phone_service = st.selectbox("Phone service?", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple lines?", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online security?", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online backup?", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device protection?", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech support?", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies?", ["No", "Yes", "No internet service"])

    predict_button = st.button("Predict churn risk", type="primary", use_container_width=True)


# ============================================================
# Debug — must come AFTER the sidebar (REMOVE BEFORE DEPLOY)
# ============================================================
with st.expander("🔧 Debug — artefact load status & transformer test"):
    st.write(f"Model type: `{type(model).__name__}`")
    st.write(f"Number of expected features: **{len(feature_columns)}**")

    if predict_button:
        try:
            test_df = transform_user_input(
                gender, senior, partner, dependents, tenure, contract,
                paperless, payment_method, monthly_charges,
                phone_service, multiple_lines, internet_service,
                online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies,
                feature_columns
            )
            st.write(f"✅ Transformer succeeded. Shape: {test_df.shape}")
            st.write(f"Columns match expected: {list(test_df.columns) == feature_columns}")
        except Exception as e:
            st.error(f"❌ Transformer failed: {e}")


# ============================================================
# Main panel — Results
# ============================================================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Churn Prediction")

    if predict_button:
        input_df = transform_user_input(
            gender, senior, partner, dependents, tenure, contract,
            paperless, payment_method, monthly_charges,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies,
            feature_columns
        )

        churn_proba, is_churner = predict_churn(input_df, model, threshold=0.38)
        clv = calculate_clv(monthly_charges, tenure, contract, clv_reference)
        segment = assign_segment(churn_proba, clv)
        action = recommend_action(segment)

        st.metric(
            label="Churn Probability",
            value=f"{churn_proba * 100:.1f}%",
            delta=f"{'⚠️ Flagged at 0.38 threshold' if is_churner else '✓ Below threshold'}",
            delta_color="inverse" if is_churner else "normal"
        )
        st.progress(churn_proba)

        col_clv, col_seg = st.columns(2)
        with col_clv:
            st.metric("Customer Lifetime Value", f"€{clv:,.0f}")
        with col_seg:
            st.metric("Segment", segment)

        st.markdown("##### Recommended Retention Action")
        message = (
            f"**{action['intervention']}**  \n"
            f"*Estimated cost: €{action['cost']} per customer*  \n"
            f"_{action['rationale']}_"
        )
        if action["tone"] == "error":
            st.error(message)
        elif action["tone"] == "success":
            st.success(message)
        elif action["tone"] == "warning":
            st.warning(message)
        else:
            st.info(message)
    else:
        st.caption("Enter customer details in the sidebar and click *Predict churn risk* to see the result.")

with col_right:
    st.subheader("Why this prediction?")

    if predict_button:
        # Compute SHAP values for this single prediction
        shap_values = explainer.shap_values(input_df)
        expected_value = explainer.expected_value

        # Build a SHAP Explanation object for the waterfall plot
        single_explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns.tolist()
        )

        # Render waterfall plot
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(single_explanation, max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)

        st.caption(
            "Each bar shows how much one feature pushed this customer's predicted "
            "churn probability up (red, toward churn) or down (blue, toward retention). "
            "The model's baseline is the average prediction across all customers; "
            "individual features explain the deviation from that baseline."
        )
    else:
        st.caption("Click *Predict churn risk* to see the per-customer feature attribution.")


# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "Built by Rishit Arora — MSc Business Analytics, University of Galway. "
    "Model: tuned XGBoost on IBM Telco Customer Churn dataset (7,043 customers, 37 features). "
    "[GitHub](https://github.com/rishitarora852/Customer-churn-prediction)"
)