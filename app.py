import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import streamlit as st


# -----------------------------
# Config
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "knn_regression_dataset.csv")
TARGET_COL = "target"


# -----------------------------
# Data / preprocessing
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def build_preprocess_pipeline(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    return preprocess


@st.cache_resource(show_spinner=False)
def train_model(metric: str, n_neighbors: int):
    df = load_data()
    preprocess = build_preprocess_pipeline(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        metric=metric,
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return pipe, {"mse": mse, "r2": r2}


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="KNN Regression", layout="wide")

st.title("KNN Regression")


with st.sidebar:
    st.header("Model Settings")

    metric = st.selectbox(
        "Distance Metric",
        options=[
            "euclidean",
            "manhattan",
            "minkowski",
            "hamming",
        ],
    )

    n_neighbors = st.slider(
        "K (Number of Neighbors)",
        min_value=1,
        max_value=25,
        value=5,
        step=1,
    )

    st.caption("Train/Test Split: 80/20")


df = load_data()

feature_cols = [c for c in df.columns if c != TARGET_COL]

pipe, metrics = train_model(
    metric=metric,
    n_neighbors=n_neighbors,
)


# -----------------------------
# Metrics
# -----------------------------
col1, col2 = st.columns(2)

col1.metric("MSE", float(metrics["mse"]))
col2.metric("R² Score", float(metrics["r2"]))


# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Input Features")

X_one = {}

for col in feature_cols:

    if df[col].dtype.kind in "if":

        cmin = float(np.nanmin(df[col].values.astype(float)))
        cmax = float(np.nanmax(df[col].values.astype(float)))
        default = float(np.nanmedian(df[col].values.astype(float)))

        X_one[col] = st.number_input(
            f"{col}",
            value=default,
            min_value=cmin,
            max_value=cmax,
        )

    else:

        options = sorted(df[col].dropna().astype(str).unique().tolist())

        X_one[col] = st.selectbox(
            f"{col}",
            options=options,
        )


# -----------------------------
# Prediction
# -----------------------------
predict_btn = st.button("Predict Target")

if predict_btn:

    X_df = pd.DataFrame([X_one], columns=feature_cols)

    y_hat = pipe.predict(X_df)[0]

    st.success(f"Predicted Target: {y_hat:,.3f}")

    st.caption(
        "KNN Regression predicts using the average value of nearest neighbors."
    )




