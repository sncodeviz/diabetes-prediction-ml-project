import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # this is from imblearn, not sklearn


# ---------- 1. Load data + train model (cached) ----------
@st.cache_resource
def train_model():
    # TODO: change this to your actual CSV name in the repo
    data = pd.read_csv("diabetes_prediction_dataset.csv")

    # TODO: change "TARGET_COLUMN" to your actual target column, e.g. "diabetes_binary" or "Diabetic"
    target_col = "diabetes"

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split (optional, but matches your project structure)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TODO: adjust this pipeline to match what you used in your final Decision Tree model
    dt_pipeline = Pipeline(
        steps=[
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                random_state=42,
                # max_depth=..., criterion=..., etc if you used them
            )),
        ]
    )

    dt_pipeline.fit(X_train, y_train)

    feature_names = list(X.columns)

    return dt_pipeline, feature_names


model, FEATURE_NAMES = train_model()


# ---------- 2. Streamlit UI ----------
st.title("🩺 Diabetes Risk Prediction Demo")
st.write("""
This interactive app uses a machine learning model trained on health data to predict the **likelihood of diabetes**.

> ⚠️ **Disclaimer:** This tool is for educational/demo purposes only and is **not** a medical diagnostic device.
""")

st.sidebar.header("Input Features")

user_input = {}

st.sidebar.write("Fill in the values for each feature below:")

for name in FEATURE_NAMES:
    # Here we treat everything as numeric. This works well if your dataset is coded as numbers (0/1, etc.)
    user_input[name] = st.sidebar.number_input(
        label=name,
        value=0.0,
        step=0.1,
        format="%.2f",
    )

input_df = pd.DataFrame([user_input])

st.subheader("Your Input")
st.dataframe(input_df)


# ---------- 3. Prediction ----------
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

        st.caption("Again: this is a demo based on a machine learning model, not medical advice.")

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)
