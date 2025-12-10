import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# =========================
# 1. CONSTANTS & CONFIG
# =========================
st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")

# These MUST match how you trained your model in Colab
NUMERIC_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_FEATURES = ["gender", "smoking_history", "hypertension", "heart_disease"]
TARGET = "diabetes"


# =========================
# 2. LOAD DATA & MODEL
# =========================
@st.cache_resource
def load_data():
    """Load the dataset used for training/evaluation."""
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    return data


@st.cache_resource
def load_model():
    """
    Load the pre-trained model pipeline from disk.
    This should be the same pipeline you saved in Colab:
    joblib.dump(model, 'diabetes_model.pkl')
    """
    model = joblib.load("diabetes_model.pkl")
    return model


data = load_data()
model = load_model()
TOTAL_ROWS = len(data)

# For dropdowns
gender_options = sorted(data["gender"].dropna().unique())
smoking_options = sorted(data["smoking_history"].dropna().unique())


# =========================
# 3. EVALUATION FUNCTION
# =========================
@st.cache_data
def evaluate_model(n_eval_rows: int):
    """
    Evaluate the loaded model on a random subset of the dataset.

    NOTE: Since the model is already trained (in Colab),
    here we are only measuring how well it performs on a sampled
    subset of the available data. The model is NOT retrained here.
    """
    # Sample rows for evaluation
    n_eval_rows = min(n_eval_rows, TOTAL_ROWS)
    eval_data = data.sample(n=n_eval_rows, random_state=42)

    X_eval = eval_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_true = eval_data[TARGET]

    y_pred = model.predict(X_eval)

    accuracy = accuracy_score(y_true, y_pred)
    precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    report_text = classification_report(y_true, y_pred)

    metrics = {
        "eval_rows": n_eval_rows,
        "accuracy": accuracy,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "report": report_text,
    }

    return metrics


# =========================
# 4. APP TITLE & DESCRIPTION
# =========================
st.title("🩺 Diabetes Risk Prediction Demo")

st.write(
    """
This interactive app uses a **pre-trained machine learning model** to predict the
likelihood of diabetes based on basic health information.

> ⚠️ **Disclaimer:** This tool is for educational/demo purposes only and is **not** a medical diagnostic device.
"""
)


# =========================
# 5. SIDEBAR: EVAL SETTINGS
# =========================
st.sidebar.header("Model Evaluation Settings")

min_eval_rows = min(1000, TOTAL_ROWS)
if TOTAL_ROWS <= 1000:
    min_eval_rows = max(50, int(TOTAL_ROWS * 0.3))

n_eval_rows = st.sidebar.slider(
    "Number of rows used to evaluate the model",
    min_value=min_eval_rows,
    max_value=TOTAL_ROWS,
    value=min(5000, TOTAL_ROWS),
    step=max(1, TOTAL_ROWS // 20),
)

st.sidebar.caption(
    f"Dataset total rows: **{TOTAL_ROWS}**. "
    f"Currently evaluating on **{n_eval_rows}** rows."
)

# Get metrics for the current evaluation size
metrics = evaluate_model(n_eval_rows)


# =========================
# 6. SIDEBAR: USER INPUT
# =========================
st.sidebar.header("Patient Features")
st.sidebar.write("Fill in the values for each feature below:")

user_input = {}

# Default numeric values (medians from full dataset)
numeric_defaults = data[NUMERIC_FEATURES].median()

# Numeric inputs
for name in NUMERIC_FEATURES:
    user_input[name] = st.sidebar.number_input(
        label=name,
        value=float(numeric_defaults[name]),
        step=0.1,
    )

# Categorical inputs
user_input["gender"] = st.sidebar.selectbox("gender", options=gender_options)

user_input["smoking_history"] = st.sidebar.selectbox(
    "smoking_history",
    options=smoking_options,
)

htn_display = st.sidebar.selectbox("hypertension", ["0 (No)", "1 (Yes)"])
user_input["hypertension"] = 1 if "1" in htn_display else 0

hd_display = st.sidebar.selectbox("heart_disease", ["0 (No)", "1 (Yes)"])
user_input["heart_disease"] = 1 if "1" in hd_display else 0

# Create DataFrame in correct column order
input_df = pd.DataFrame([user_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

st.subheader("Your Input")
st.dataframe(input_df)


# =========================
# 7. MODEL PERFORMANCE
# =========================
st.subheader("📊 Model Performance (on Sampled Data)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Eval rows", f"{metrics['eval_rows']}")
c2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
c3.metric("Precision (class 1)", f"{metrics['precision_1']:.3f}")
c4.metric("Recall (class 1)", f"{metrics['recall_1']:.3f}")

st.caption(
    "Performance is measured on a random subset of the dataset. "
    "Move the slider in the sidebar to change how many rows are used for evaluation "
    "and see how the metrics change."
)

with st.expander("Show full classification report"):
    st.text(metrics["report"])


# =========================
# 8. PREDICTION
# =========================
st.subheader("🧮 Diabetes Risk Prediction")

if st.button("Predict Diabetes Risk"):
    try:
        pred = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
        else:
            proba = None

        if int(pred) == 1:
            st.error("⚠️ The model predicts: **Diabetic (Higher Risk)**")
        else:
            st.success("✅ The model predicts: **Non-Diabetic (Lower Risk)**")

        if proba is not None:
            st.write(f"**Estimated probability of diabetes (class 1):** {proba:.2%}")

        st.caption(
            "This prediction is generated by your pre-trained machine learning model. "
            "It is for educational purposes only and is not medical advice."
        )

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)
