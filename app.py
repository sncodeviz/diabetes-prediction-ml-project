import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------- 1. Load data ----------
@st.cache_resource
def load_data():
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    return data


# ---------- 2. Train model (cached) ----------
@st.cache_resource
def train_model(data: pd.DataFrame):
    # Match your notebook
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender', 'smoking_history', 'hypertension', 'heart_disease']
    target = 'diabetes'

    X = data[numeric_features + categorical_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # SMOTE + Decision Tree (your final model)
    smote = SMOTE(random_state=42)

    dt_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ))
    ])

    dt_pipeline.fit(X_train, y_train)

    gender_options = sorted(data['gender'].unique())
    smoking_options = sorted(data['smoking_history'].unique())

    return dt_pipeline, numeric_features, categorical_features, gender_options, smoking_options


data = load_data()
model, numeric_features, categorical_features, gender_options, smoking_options = train_model(data)


# ---------- 3. Streamlit UI ----------
st.title("🩺 Diabetes Risk Prediction Demo")
st.write("""
This interactive app uses a machine learning model trained on health data to predict the **likelihood of diabetes**.

> ⚠️ **Disclaimer:** This tool is for educational/demo purposes only and is **not** a medical diagnostic device.
""")

st.sidebar.header("Input Features")
st.sidebar.write("Fill in the values for each feature below:")

user_input = {}

# Default numeric values (medians from dataset)
numeric_defaults = data[numeric_features].median()

# Numeric inputs
for name in numeric_features:
    user_input[name] = st.sidebar.number_input(
        label=name,
        value=float(numeric_defaults[name]),
        step=0.1
    )

# Categorical inputs
user_input['gender'] = st.sidebar.selectbox(
    "gender",
    options=gender_options
)

user_input['smoking_history'] = st.sidebar.selectbox(
    "smoking_history",
    options=smoking_options
)

htn_display = st.sidebar.selectbox("hypertension", ["0 (No)", "1 (Yes)"])
user_input['hypertension'] = 1 if "1" in htn_display else 0

hd_display = st.sidebar.selectbox("heart_disease", ["0 (No)", "1 (Yes)"])
user_input['heart_disease'] = 1 if "1" in hd_display else 0

# Create DataFrame in correct column order
input_df = pd.DataFrame([user_input])[numeric_features + categorical_features]

st.subheader("Your Input")
st.dataframe(input_df)


# ---------- 4. Prediction ----------
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
