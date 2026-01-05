"""
Feature Engineering Module
This module creates model-ready features from cleaned hospital readmission data.
"""

import pandas as pd
from pathlib import Path


def engineer_features():
    """
    Perform basic feature engineering and save processed features.
    This function is designed to be called by the Airflow DAG.
    """

    print("Starting feature engineering step...")

    # --------------------------------------------------
    # Define paths
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "cleaned" / "hospital_readmissions_cleaned.csv"
    output_path = project_root / "data" / "processed" / "hospital_readmissions_features.csv"

    # --------------------------------------------------
    # Load cleaned data
    # --------------------------------------------------
    df = pd.read_csv(input_path)

    # --------------------------------------------------
    # FIX: Convert age column to numeric
    # --------------------------------------------------
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[0, 30, 50, 70, 120],
            labels=["young", "adult", "senior", "elder"]
        )

    # --------------------------------------------------
    # Save processed features
    # --------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Feature engineering completed.")
    print(f"Saved features to: {output_path}")
