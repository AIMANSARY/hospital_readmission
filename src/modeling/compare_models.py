"""
Model Comparison Module - FIXED VERSION
Compares Logistic Regression (FAST) and Random Forest
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.impute import SimpleImputer

def compare_models():
    print("🚀 Starting FAST model comparison...")

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "hospital_readmissions_features.csv"
    tables_path = project_root / "tables"
    figures_path = project_root / "figures"

    tables_path.mkdir(exist_ok=True)
    figures_path.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"📊 Data shape: {df.shape}")

    # Encode target
    df["readmitted"] = df["readmitted"].map({"no": 0, "yes": 1})
    df = df.dropna(subset=["readmitted"])

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"].astype(int)

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.dropna(axis=1, how="all")
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # SCALE FEATURES (CRITICAL for Logistic Regression speed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")

    # FAST Logistic Regression
    lr = LogisticRegression(
        solver="liblinear",  # FASTER than lbfgs[web:114][web:128]
        max_iter=200,        # Quick convergence
        random_state=42
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,    # Reduced for speed
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    # Metrics table
    metrics_df = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "accuracy": accuracy_score(y_test, lr_pred),
            "precision": precision_score(y_test, lr_pred, zero_division=0),
            "recall": recall_score(y_test, lr_pred, zero_division=0),
            "f1_score": f1_score(y_test, lr_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, lr_prob),
        },
        {
            "model": "Random Forest",
            "accuracy": accuracy_score(y_test, rf_pred),
            "precision": precision_score(y_test, rf_pred, zero_division=0),
            "recall": recall_score(y_test, rf_pred, zero_division=0),
            "f1_score": f1_score(y_test, rf_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, rf_prob),
        },
    ])

    # Save results
    metrics_df.to_csv(tables_path / "model_comparison_metrics.csv", index=False)
    
    # ROC Curve
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, lr_prob):.3f})")
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC={roc_auc_score(y_test
