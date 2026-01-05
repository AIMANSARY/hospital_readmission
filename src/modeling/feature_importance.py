"""
Feature Importance Module
Extracts feature importance from Random Forest
to identify key clinical drivers of readmission.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def generate_feature_importance():
    print("Starting feature importance analysis...")

    project_root = Path(__file__).resolve().parents[2]

    data_path = project_root / "data" / "processed" / "hospital_readmissions_features.csv"
    tables_path = project_root / "tables"
    figures_path = project_root / "figures"

    tables_path.mkdir(exist_ok=True)
    figures_path.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Encode target
    df["readmitted"] = df["readmitted"].map({"no": 0, "yes": 1})
    df = df.dropna(subset=["readmitted"])

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"].astype(int)

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.dropna(axis=1, how="all")
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # Save table
    importance_df.to_csv(
        tables_path / "feature_importance_rf.csv",
        index=False
    )

    # Plot top 10 features
    top_features = importance_df.head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.title("Top 10 Clinical Features Influencing Readmission")
    plt.tight_layout()
    plt.savefig(figures_path / "feature_importance_rf.png")
    plt.close()

    print("Feature importance analysis completed.")
    print(top_features)
