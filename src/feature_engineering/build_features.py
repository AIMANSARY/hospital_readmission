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

    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "cleaned" / "hospital_readmissions_cleaned.csv"
    output_path = project_root / "data" / "processed" / "hospital_readmissions_features.csv"

    # Load cleaned data
    df = pd.read_csv(input_path)

    # ------------------------
    # Basic feature engineering
    # ------------------------

    # Example: Age bucket
    if "age" in df.columns:
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[0, 30, 50, 70, 100],
            label
