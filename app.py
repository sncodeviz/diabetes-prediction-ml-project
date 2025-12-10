import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # for SMOTE + model


# =========================
# 1. CONFIG & CONSTANTS
# =========================
st.set_page_config(page_title="Diabetes Risk Prediction (Decision Tree + SMOTE)",
                   page_icon="🩺")

# These match your original notebook
numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history', 'hypertension', 'heart_disease']
target = 'diabetes'


# =========================
# 2. LOAD DATA
# =========================
@st.cache_resource
def load_data():
    # This should be the same dataset you used in the project
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    return df


df = load_data()
TOTAL_ROWS = len(df)


# =========================
# 3. TRAIN MODEL (Decision Tree + SMOTE, like your "Model 2")
# =========================
@st.cache_resource
def train_dt_smote_model(n_rows: int):
    """
    Train Decision Tree model (Model 2 in Phase 4) with SMOTE
    on a subset of the data (n_rows), using the same preprocessing
    pipeline. 
    """
    # --- Train/Test Split ---
    n_rows = min(n_rows, TOTAL_ROWS)
    data_used = df.sample(n=n_rows, random_state=42)

    X = data_used[numeric_features + categorical_features]
    y = data_used[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --- Preprocessing pipeline ---

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

    # --- SMOTE ---
    smote = SMOTE(random_state=42)

    # --- Model 2: Decision Tree ---
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

    # Fit model
    dt_pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_dt = dt_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_dt)
    precision_1 = precision_score(y_test, y_pred_dt, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_test, y_pred_dt, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_test, y_pred_dt, pos_label=1, zero_division=0)

    report_text = classification_report(y_test, y_pred_dt)

    metrics = {
        "train_rows": n_rows,
        "test_rows": len(y_test),
        "accuracy": accuracy,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "report": report_text
    }

    # Options for dropdowns (use full df so categories are complete)
    gender_options = sorted(df['gender'].dropna().unique())
    smoking_options = sorted(df['smoking_history'].dropna().unique())

    return dt_pipeline, gender_options, smoking_options, metrics


# =========================
# 4. UI: TITLE & DESCRIPTION
# =========================
st.title("🩺 Diabetes Risk Prediction – Decision Tree with SMOTE")

st.write("""
This app implements the **same Decision Tree model with SMOTE. notebook.

- Preprocessing: median imputation, most-frequent imputation, one-hot encoding, and standardization  
- Class imbalance handling: **SMOTE** on the training set  
- Final model: **DecisionTreeClassifier** (max_depth=6, min_samples_split=4, min_samples_leaf=2)

> ⚠️ **Disclaimer:** Educational/demo purposes only. This is **not** a medical diagnostic tool.
""")


# =========================
# 5. SIDEBAR: MODEL SETTINGS
# =========================
st.sidebar.header("Model Settings")

# Slider for number of rows BEFORE SMOTE (like changing dataset size)
min_rows = min(2000, TOTAL_ROWS)
if TOTAL_ROWS <= 2000:
    min_rows = int(TOTAL_ROWS * 0.5) if TOTAL_ROWS > 10 else TOTAL_ROWS

n_rows_selected = st.sidebar.slider(
    "Number of rows used for training (before SMOTE)",
    min_value=min_rows,
    max_value=TOTAL_ROWS,
    value=min(10000, TOTAL_ROWS),
    step=max(1, TOTAL_ROWS // 20),
)

st.sidebar.caption(
    f"Dataset total rows: **{TOTAL_ROWS}**. "
    f"Currently training on **{n_rows_selected}** rows (then applying SMOTE to the training split)."
)

# Train model with chosen row count
model, gender_options, smoking_options, metrics = train_dt_smote_model(n_rows_selected)


# =========================
# 6. SIDEBAR: PATIENT INPUTS
# =========================
st.sidebar.header("Patient Features")
st.sidebar.write("Fill in the values for each feature below:")

user_input = {}

# Default numeric values (medians from full dataset, like your preprocessing idea)
numeric_defaults = df[numeric_features].median()

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

# Create DataFrame in correct column order (like X in your code)
input_df = pd.DataFrame([user_input])[numeric_features + categorical_features]

st.subheader("Your Input")
st.dataframe(input_df)


# =========================
# 7. MODEL PERFORMANCE (ALIGNED WITH YOUR COMPARISON SUMMARY)
# =========================
st.subheader("📊 Model Performance – Decision Tree (with SMOTE)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training rows (before SMOTE)", f"{metrics['train_rows']}")
col2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col3.metric("Precision (class 1)", f"{metrics['precision_1']:.3f}")
col4.metric("Recall (class 1)", f"{metrics['recall_1']:.3f}")

st.caption(
    "Metrics are computed on a held-out 20% test set from the selected rows. "
    "SMOTE is applied on the training split to handle class imbalance in the diabetic class (1). "
    "Use the slider in the sidebar to see how using more or fewer rows affects performance."
)

with st.expander("Show full classification report (same style as notebook)"):
    st.text(metrics["report"])


# =========================
# 8. PREDICTION
# =========================
st.subheader("🧮 Diabetes Risk Prediction (Decision Tree + SMOTE)")

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
            "This uses your final Decision Tree model trained with SMOTE—"
            "the same pipeline as described in your Phase 4 model development."
        )

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)
