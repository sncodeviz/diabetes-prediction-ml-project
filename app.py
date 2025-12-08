import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    return model

model = load_model()

# Try to get feature names from the model (works if trained on a DataFrame)
if hasattr(model, "feature_names_in_"):
    FEATURE_NAMES = list(model.feature_names_in_)
else:
    # Fallback if feature names are not stored
    # You can replace this with your actual column names later
    FEATURE_NAMES = [f"feature_{i}" for i in range(getattr(model, "n_features_in_", 10))]

st.title("🩺 Diabetes Risk Prediction Demo")
st.write("""
This interactive demo uses a machine learning model trained on health data to predict the **likelihood of diabetes**.

> **Disclaimer:** This app is for educational and demonstration purposes only and is **not** a medical diagnostic tool.
""")

st.sidebar.header("Input Features")

#Build user input section
user_input = {}

st.sidebar.write("Please fill in the values for each feature below:")

for name in FEATURE_NAMES:
    # Basic numeric input for each feature.
    # You can customize labels, ranges, and defaults later.
    user_input[name] = st.sidebar.number_input(
        label=name,
        value=0.0,
        step=0.1,
        format="%.2f"
    )

# Convert to DataFrame for the model (matches what it saw during training)
input_df = pd.DataFrame([user_input])

st.subheader("Your Input")
st.dataframe(input_df)

# ---------- Prediction ----------
if st.button("Predict Diabetes Risk"):
    try:
        prediction = model.predict(input_df)[0]

        # If model supports predict_proba (for probability)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]  # probability of class 1
        else:
            proba = None

        if prediction == 1:
            st.error("⚠️ The model predicts: **Diabetic (Higher Risk)**")
        else:
            st.success("✅ The model predicts: **Non-Diabetic (Lower Risk)**")

        if proba is not None:
            st.write(f"**Estimated probability of diabetes (class 1):** {proba:.2%}")

        st.caption("Again: this is a demo based on a machine learning model, not medical advice.")

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)
