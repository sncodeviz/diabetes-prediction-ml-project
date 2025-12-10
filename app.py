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
    page_title="Diabetes Risk Prediction – Decision Tree with Oversampling",
    page_icon="🩺",
)

# Match your project features
NUMERIC_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_FEATURES = ["gender", "smoking_history", "hypertension", "heart_disease"]
TARGET = "diabetes"


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
# 3. SMOTE-LIKE OVERSAMPLING (custom, no imblearn)
# =========================
def smote_like_oversample(X, y, minority_class=1, k=5, random_state=42):
    """
    Very simple SMOTE-like oversampling implemented with NumPy + scikit-learn.

    - X: 2D numpy array (numeric features)
    - y: 1D array-like labels
    - Creates synthetic samples for the minority_class until
      it's balanced with the majority.
    """
    rng = np.random.RandomState(random_state)

    y = np.asarray(y)
    X = np.asarray(X)

    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]

    n_min = len(minority_idx)
    n_maj = len(majority_idx)

    if n_min == 0 or n_min >= n_maj:
        # Already balanced or no minority class
        return X, y

    n_synth = n_maj - n_min

    X_min = X[minority_idx]

    # k-NN among minority samples
    k = min(k, n_min)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_min)
    neighbors = nn.kneighbors(return_distance=False)  # shape: (n_min, k)

    synthetic_samples = []

    for _ in range(n_synth):
        i = rng.randint(0, n_min)  # pick random minority sample
        xi = X_min[i]

        nn_index = rng.choice(neighbors[i])
        xnn = X_min[nn_index]

        lam = rng.rand()          # interpolation factor in [0, 1]
        synthetic = xi + lam * (xnn - xi)
        synthetic_samples.append(synthetic)

    X_syn = np.vstack(synthetic_samples)
    y_syn = np.full(n_synth, minority_class, dtype=y.dtype)

    X_out = np.vstack([X, X_syn])
    y_out = np.concatenate([y, y_syn])

    return X_out, y_out


# =========================
# 4. TRAIN MODEL (Decision Tree + custom oversampling)
# =========================
@st.cache_resource
def train_model(n_rows: int):
    """
    Train a Decision Tree on a subset of the data (n_rows),
    using:
      - preprocessing (impute + one-hot + scale)
      - custom SMOTE-like oversampling on the training set
    Returns: fitted preprocessor, classifier, dropdown options, metrics dict
    """
    n_rows = min(n_rows, TOTAL_ROWS)
    data_used = df.sample(n=n_rows, random_state=42)

    X = data_used[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data_used[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --- Preprocessing ---

    # Numeric pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline (dense output so we can oversample on the full matrix)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Fit preprocessor on TRAIN only
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- Custom oversampling (SMOTE-like) on processed training data ---
    X_train_bal, y_train_bal = smote_like_oversample(
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
    y_pred = clf.predict(X_test_proc)

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
        "report": report_text,
    }

    gender_options = sorted(df["gender"].dropna().unique())
    smoking_options = sorted(df["smoking_history"].dropna().unique())

    return preprocessor, clf, gender_options, smoking_options, metrics


# =========================
# 5. UI: TITLE & DESCRIPTION
# =========================
st.title("🩺 Diabetes Risk Prediction – Decision Tree with Oversampling")

st.write(
    """
This app trains a **Decision Tree model** on the diabetes dataset and uses a
**SMOTE-like oversampling** step (implemented directly here) to handle class imbalance.

- Features: age, BMI, HbA1c level, blood glucose level, gender, smoking history,
  hypertension, heart disease  
- Preprocessing: imputation, one-hot encoding, scaling  
- Class imbalance: minority diabetic class is oversampled using a k-NN interpolation scheme  
- Final model: Decision Tree with your project hyperparameters:
  `max_depth=6`, `min_samples_split=4`, `min_samples_leaf=2`.

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

preprocessor, clf, gender_options, smoking_options, metrics = train_model(
    n_rows_selected
)


# =========================
# 7. SIDEBAR: PATIENT INPUTS
# =========================
st.sidebar.header("Patient Features")
st.sidebar.write("Fill in the values for each feature below:")

user_input = {}
numeric_defaults = df[NUMERIC_FEATURES].median()

for name in NUMERIC_FEATURES:
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

input_df = pd.DataFrame([user_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

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
        # Apply same preprocessing as during training
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
