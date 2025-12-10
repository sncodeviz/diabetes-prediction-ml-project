import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ---------- 1. Load data ----------
@st.cache_resource
def load_data():
    # Make sure this CSV is in the same folder as app.py
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    return data


data = load_data()
TOTAL_ROWS = len(data)

# Define feature lists once (used for both model + UI)
NUMERIC_FEATURES = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
CATEGORICAL_FEATURES = ['gender', 'smoking_history', 'hypertension', 'heart_disease']
TARGET = 'diabetes'


# ---------- 2. Train model with adjustable row count ----------
@st.cache_resource
def train_model(n_rows: int | None = None):
    """
    Train a Decision Tree model on a subset of the data (n_rows).
    Returns the trained model and performance metrics.
    """
    # Decide how many rows to use
    if n_rows is not None and n_rows < len(data):
        data_used = data.sample(n=n_rows, random_state=42)
    else:
        data_used = data.copy()
        n_rows = len(data_used)

    X = data_used[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data_used[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y  # still okay as long as there are both classes
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
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # Decision Tree model (no SMOTE in deployed app)
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ))
    ])

    # Train the pipeline
    dt_pipeline.fit(X_train, y_train)

    # ---- Compute performance metrics on test set ----
    y_pred = dt_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    report_text = classification_report(y_test, y_pred)

    metrics = {
        "train_rows": n_rows,
        "test_rows": len(y_test),
        "accuracy": accuracy,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "report": report_text
    }

    # For UI dropdowns (use full dataset so all categories remain available)
    gender_options = sorted(data['gender'].dropna().unique())
    smoking_options = sorted(data['smoking_history'].dropna().unique())

    return dt_pipeline, gender_options, smoking_options, metrics


# ---------- 3. Streamlit UI ----------
st.title("🩺 Diabetes Risk Prediction Demo")

st.write("""
This interactive app uses a machine learning model trained on health data to predict the **likelihood of diabetes**.

> ⚠️ **Disclaimer:** This tool is for educational/demo purposes only and is **not** a medical diagnostic device.
""")

# --- Sidebar: training size slider ---
st.sidebar.header("Model Settings")

min_rows = min(2000, TOTAL_ROWS)  # fallback if dataset is small
if TOTAL_ROWS <= 2000:
    min_rows = int(TOTAL_ROWS * 0.5) if TOTAL_ROWS > 10 else TOTAL_ROWS

n_rows_selected = st.sidebar.slider(
    "Number of rows used for training",
    min_value=min_rows,
    max_value=TOTAL_ROWS,
    value=min(10000, TOTAL_ROWS),
    step=max(1, (TOTAL_ROWS // 20))  # dynamic step size
)

st.sidebar.caption(
    f"Dataset total rows: **{TOTAL_ROWS}**. "
    f"Currently training on **{n_rows_selected}** rows."
)

# Train model with chosen row count
model, gender_options, smoking_options, metrics = train_model(n_rows_selected)

# --- Sidebar: Input features ---
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
input_df = pd.DataFrame([user_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

st.subheader("Your Input")
st.dataframe(input_df)


# ---------- 4. Model Performance Section ----------
st.subheader("📊 Model Performance (Test Set)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training rows", f"{metrics['train_rows']}")
col2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col3.metric("Precision (class 1)", f"{metrics['precision_1']:.3f}")
col4.metric("Recall (class 1)", f"{metrics['recall_1']:.3f}")

st.caption(
    "Performance is measured on a held-out test set (20% of the selected rows). "
    "Try changing the training row slider in the sidebar to see how metrics change."
)

with st.expander("Show full classification report"):
    st.text(metrics["report"])


# ---------- 5. Prediction ----------
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

        st.caption("Again: this is a demo based on a machine learning model, not medical advice.")

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)
