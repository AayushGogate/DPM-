"""
Diabetes Risk Prediction - Streamlit Web App
Trained on the Pima Indians Diabetes Dataset.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io  (Streamlit Community Cloud)
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

FEATURE_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


# ----------------------------------------------------------------------
# Load model artifacts (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('metrics.json') as f:
        metrics = json.load(f)
    with open('feature_stats.json') as f:
        feature_stats = json.load(f)
    return model, scaler, metrics, feature_stats


model, scaler, metrics, feature_stats = load_artifacts()

# ----------------------------------------------------------------------
# Sidebar - about / model info
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("About this tool")
    st.write(
        "This app predicts diabetes risk using a machine learning model "
        "trained on the **Pima Indians Diabetes Dataset** (768 patient records, "
        "National Institute of Diabetes and Digestive and Kidney Diseases)."
    )

    st.subheader("Model performance")
    st.metric("Model", metrics['model_name'])
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c1.metric("Precision", f"{metrics['precision']*100:.1f}%")
    c2.metric("Recall", f"{metrics['recall']*100:.1f}%")

    st.caption(
        f"Evaluated on a held-out test set of {metrics['n_test']} patients "
        f"(trained on {metrics['n_train']})."
    )

    st.divider()
    st.warning(
        "⚠️ **Not a medical device.** This tool is for educational/demo "
        "purposes only and must never replace professional medical advice, "
        "diagnosis, or treatment. Always consult a qualified healthcare "
        "provider."
    )

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter the patient's health indicators below. These are the same "
    "eight clinical measurements used in the Pima Indians Diabetes Study."
)

st.divider()

with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1, step=1,
            help="Number of times pregnant"
        )
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=110.0, step=1.0,
            help="Plasma glucose concentration, 2-hour oral glucose tolerance test"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0,
            help="Diastolic blood pressure"
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
            help="Triceps skin fold thickness"
        )

    with col2:
        insulin = st.number_input(
            "Insulin (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0, step=1.0,
            help="2-hour serum insulin"
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Body mass index = weight(kg) / height(m)^2"
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4, step=0.01,
            help="A function scoring likelihood of diabetes based on family history"
        )
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=33, step=1
        )

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=FEATURE_COLS)

    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0]
    prediction = model.predict(input_scaled)[0]
    risk_pct = proba[1] * 100

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"### ⚠️ Higher Diabetes Risk\nEstimated probability: **{risk_pct:.1f}%**")
    else:
        st.success(f"### ✅ Lower Diabetes Risk\nEstimated probability: **{risk_pct:.1f}%**")

    st.progress(min(int(risk_pct), 100))

    # Flag out-of-typical-range inputs for user awareness
    flags = []
    if glucose == 0 or blood_pressure == 0 or bmi == 0:
        flags.append("Some values are set to 0, which is not physiologically typical — "
                      "double check your entries for accuracy.")
    if glucose >= 126:
        flags.append("Glucose ≥126 mg/dL is in the diabetic range per clinical guidelines.")
    if bmi >= 30:
        flags.append("BMI ≥30 is classified as obese, a known diabetes risk factor.")

    if flags:
        with st.expander("ℹ️ Notes on your inputs"):
            for f in flags:
                st.write(f"- {f}")

    st.caption(
        "This prediction is a statistical estimate based on patterns in historical "
        "data, not a diagnosis. Please consult a doctor for medical evaluation."
    )

st.divider()
st.caption("Built with Python, scikit-learn & Streamlit · Pima Indians Diabetes Dataset")
