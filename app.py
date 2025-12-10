''' 
SMOTE from imbalanced-learn was used to address class imbalance.
In the deployed Streamlit app, due to environment limitations with imbalanced-learn, I implemented a custom SMOTE-like oversampling step using scikit-learn’s nearest neighbors and interpolation directly in the app.
'''

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
from sklearn.neighbors import NearestNeighbors


# =========================
# 1. CONFIG & CONSTANTS
# =========================
st.set_page_config(
    page_title="Diabetes Risk Prediction – Decision Tree with SMOTE-like Oversampling",
    page_icon="🩺",
)

# Match your original feature setup
numeric_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
categorical_features = ["gender", "smoking_history", "hypertension", "heart_disease"]
target = "diabetes"


# =========================
# 2. LOAD DATA
# =========================
@st.cache_resource
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    return df


df = load_data()
TOTAL_ROWS = len(df)


# =========================
# 3. CUSTOM SMOTE-LIKE OVERSAMPLING (no imblearn)
# =========================
def smote_oversample(X, y, minority_class=1, k=5, random_state=42):
    """
    Very simple SMOTE-like oversampling:
    - Works on a dense numeric feature matrix X (numpy array)
    - Adds synthetic samples for the minority class to balance it with the majority class
    - Uses k-NN among minority samples and interpolation between neighbors

    This is NOT as full-featured as imblearn.SMOTE but captures the same idea.
    """
    rng = np.random.RandomState(random_state)

    y = np.asarray(y)
    X = np.asarray(X)

    # Indices of minority and majority class
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]

    n_min = len(minority_idx)
    n_maj = len(majority_idx)

    # If already balanced or no minority, do nothing
    if n_min == 0 or n_min >= n_maj:
        return X, y

    n_synth = n_maj - n_min  # number of synthetic samples to create

    X_min = X[minority_idx]

    # k-NN among minority samples
    k = min(k, n_min)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_min)
    neighbors = nn.kneighbors(return_distance=False)  # shape (n_min, k)

    synthetic_samples = []

    for _ in range(n_synth):
        i = rng.randint(0, n_min)          # pick a random minority point
        xi = X_min[i]

        nn_index = rng.choice(neighbors[i])
        xnn = X_min[nn_index]

        lam = rng.rand()                   # random interpolation factor [0,1]
        synthetic = xi + lam * (xnn - xi)
        synthetic_samples.append(synthetic)

    X_syn = np.vstack(synthetic_samples)
    y_syn = np.full(n_synth, minority_class, dtype=y.dtype)

    X_out = np.vstack([X, X_syn])
    y_out = np.concatenate([y, y_syn])

    return X_out, y_out


# =========================
# 4. TRAIN MODEL (Decision Tree + custom SMOTE)
# =========================
@st.cache_resource
def train_dt_smote_like_model(n_rows: int):
    """
    Train a Decision Tree with a SMOTE-like oversampling step,
    using the same preprocessing as your original project.
    Everything happens here in app.py (no external model file).
    """
    # Sample subset for training + testing
    n_rows = min(n_rows, TOTAL_ROWS)
    data_used = df.sample(n=n_rows, random_state=42)

    X = data_used[numeric_features + categorical_features]
    y = data_used[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --- Preprocessing pipeline (same structure as notebook) ---

    # Numeric pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline (dense output so we can do SMOTE on the matrix)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit preprocessor on TRAIN only
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- Custom SMOTE-like oversampling on processed features ---
    X_train_bal, y_train_bal = smote_oversample(
        X_train_proc, y_train, minority_class=1, k=5, random_state=42
    )

    # --- Decision Tree classifier (your hyperparameters) ---
    clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )

    clf.fit(X_train_bal, y_train_bal)

    # Evaluate on test set (no oversampling on test)
    y_pred_dt = clf.predict(X_test_proc)

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
        "report": report_text,
    }

    # Dropdown options (use full df so categories are complete)
    gender_options = sorted(df["gender"].dropna().unique())
    smoking_options = sorted(df["smoking_history"].dropna().unique())

    return preprocessor, clf, gender_options, smoking_options, metrics


# =========================
# 5. UI: TITLE & DESCRIPTION
# =========================
st.title("🩺 Diabetes Risk Prediction – Decision Tree with SMOTE-like Oversampling")

st.write(
    """
This app trains a **Decision Tree model** on the diabetes dataset and uses a
**SMOTE-like oversampling step** implemented directly in this file (no `imbalanced-learn`).

- Features: age, BMI, HbA1c level, blood glucose level, gender, smoking history,
  hypertension, heart disease  
- Preprocessing: imputation, one-hot encoding, scaling  
- Class imbalance: oversampling of the diabetic class using a k-NN interpolation scheme  
- Final model: Decision Tree with your hyperparameters (`max_depth=6`, `min_samples_split=4`,
  `min_samples_leaf=2`).

> ⚠️ **Disclaimer:** Educational/demo purposes only. This is **not** a medical diagnostic tool.
"""
)


# =========================
# 6. SIDEBAR: MODEL SETTINGS
# =========================
st.sidebar.header("Model Settings")

min_rows = min(2000, TOTAL_ROWS)
if TOTAL_ROWS <= 2000:
    min_rows = int(TOTAL_ROWS * 0.5) if TOTAL_ROWS > 10 else TOTAL_ROWS

n_rows_selected = st.sidebar.slider(
    "Number of rows used for training (before oversampling)",
    min_value=min_rows,
    max_value=TOTAL_ROWS,
    value=min(10000, TOTAL_ROWS),
    step=max(1, TOTAL_ROWS // 20),
)

st.sidebar.caption(
    f"Dataset total rows: **{TOTAL_ROWS}**. "
    f"Currently training on **{n_rows_selected}** rows (then oversampling the minority class)."
)

# Train model + get metrics
preprocessor, clf, gender_options, smoking_options, metrics = train_dt_smote_like_model(
    n_rows_selected
)


# =========================
# 7. SIDEBAR: PATIENT INPUTS
# =========================
st.sidebar.header("Patient Features")
st.sidebar.write("Fill in the values for each feature below:")

user_input = {}
numeric_defaults = df[numeric_features].median()

for name in numeric_features:
    user_input[name] = st.sidebar.number_input(
        label=name,
        value=float(numeric_defaults[name]),
        step=0.1,
    )

user_input["gender"] = st.sidebar.selectbox("gender", options=gender_options)

user_input["smoking_history"] = st.sidebar.selectbox(
    "smoking_history", options=smoking_options
)

htn_display = st.sidebar.selectbox("hypertension", ["0 (No)", "1 (Yes)"])
user_input["hypertension"] = 1 if "1" in htn_display else 0

hd_display = st.sidebar.selectbox("heart_disease", ["0 (No)", "1 (Yes)"])
user_input["heart_disease"] = 1 if "1" in hd_display else 0

input_df = pd.DataFrame([user_input])[numeric_features + categorical_features]

st.subheader("Your Input")
st.dataframe(input_df)


# =========================
# 8. MODEL PERFORMANCE
# =========================
st.subheader("📊 Model Performance – Decision Tree with Oversampling")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training rows (before oversampling)", f"{metrics['train_rows']}")
c2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
c3.metric("Precision (class 1)", f"{metrics['precision_1']:.3f}")
c4.metric("Recall (class 1)", f"{metrics['recall_1']:.3f}")

st.caption(
    "Metrics are computed on a 20% held-out test set from the selected rows. "
    "Oversampling is applied only on the training split. "
    "Use the slider to see how training size affects performance."
)

with st.expander("Show full classification report"):
    st.text(metrics["report"])


# =========================
# 9. PREDICTION
# =========================
st.subheader("🧮 Diabetes Risk Prediction")

if st.button("Predict Diabetes Risk"):
    try:
        # Apply same preprocessing as training
        X_input_proc = preprocessor.transform(input_df)
        pred = clf.predict(X_input_proc)[0]

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_input_proc)[0][1]
        else:
            proba = None

        if int(pred) == 1:
            st.error("⚠️ The model predicts: **Diabetic (Higher Risk)**")
        else:
            st.success("✅ The model predicts: **Non-Diabetic (Lower Risk)**")

        if proba is not None:
            st.write(f"**Estimated probability of diabetes (class 1):** {proba:.2%}")

        st.caption(
            "This prediction uses a Decision Tree trained with a SMOTE-like oversampling "
            "step implemented directly in this app (no external model files or imblearn)."
        )

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)
