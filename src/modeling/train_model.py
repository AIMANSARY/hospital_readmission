"""
Model Training Module
Trains a baseline Logistic Regression model for hospital readmission prediction
and saves evaluation metrics as a CSV table.
"""

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
import joblib


def train_model():
    """
    Train a baseline Logistic Regression model and save evaluation metrics.
    This function is designed to be called by the Airflow DAG.
    """

    print("Starting model training step...")

    # --------------------------------------------------
    # Define paths
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]

    data_path = project_root / "data" / "processed" / "hospital_readmissions_features.csv"
    metrics_path = project_root / "tables" / "baseline_model_metrics.csv"
    model_path = project_root / "src" / "modeling" / "models" / "baseline_model.pkl"

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_csv(data_path)

    target_col = "readmitted"

    # ✅ FIX: Encode target variable (no/yes → 0/1)
    df[target_col] = df[target_col].map({"no": 0, "yes": 1})

    # Drop rows where target is missing after mapping
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # --------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # --------------------------------------------------
    # Handle missing values
    # --------------------------------------------------
    X = X.dropna(axis=1, how="all")

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # --------------------------------------------------
    # Train-test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Train baseline model
    # --------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # Metrics table (REQUIRED FORMAT)
    # --------------------------------------------------
    metrics_df = pd.DataFrame({
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ],
        "value": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0),
            roc_auc_score(y_test, y_prob),
        ]
    })

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print("Model training completed successfully.")
    print(metrics_df)
