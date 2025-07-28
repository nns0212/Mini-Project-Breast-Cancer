import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("C:\\Users\\ASUS\\PycharmProjects\\BreastCancer\\.venv\\stacking_model.joblib")
scaler = joblib.load("C:\\Users\\ASUS\\PycharmProjects\\BreastCancer\\.venv\\scaler.joblib")
imputer = joblib.load("C:\\Users\\ASUS\\PycharmProjects\\BreastCancer\\.venv\\imputer.joblib")
selected_features = joblib.load("C:\\Users\\ASUS\\PycharmProjects\\BreastCancer\\.venv\\selected_features.joblib")

# App Title
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("This app uses a **stacked ensemble model** to predict whether a tumor is **malignant** or **benign** based on input features.")

# Sidebar for input
st.sidebar.header("📌 Input Tumor Features")
feature_inputs = {}

for feature in selected_features:
    feature_inputs[feature] = st.sidebar.number_input(f"{feature}", format="%.5f")

# Check if all inputs are filled
if any(v is None for v in feature_inputs.values()):
    st.warning("⚠️ Please enter all the required feature values to proceed.")
else:
    if st.button("Predict"):
        try:
            # Prepare input for prediction
            input_df = pd.DataFrame([feature_inputs])
            input_imputed = imputer.transform(input_df)
            input_scaled = scaler.transform(input_imputed)

            # Prediction and probability
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]

            # Output result
            if prediction == 1:
                st.subheader("🧬 The tumor is **Malignant**.")
            else:
                st.subheader("✅ The tumor is **Benign**.")

            st.write(f"🔵 **Probability of Benign**: `{probabilities[0]:.2%}`")
            st.write(f"🔴 **Probability of Malignant**: `{probabilities[1]:.2%}`")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
