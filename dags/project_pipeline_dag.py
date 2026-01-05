"""
Hospital Readmission Prediction Pipeline
Airflow DAG for Part II – Technical Submission
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# --------------------------------------------------
# Import pipeline functions
# --------------------------------------------------

from src.data_ingestion.ingest_data import ingest_data
from src.data_cleaning.clean_data import clean_data
from src.feature_engineering.build_features import engineer_features
from src.modeling.train_model import train_model
from src.evaluation.generate_outputs import generate_outputs

# --------------------------------------------------
# Default DAG arguments
# --------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

# --------------------------------------------------
# DAG definition
# --------------------------------------------------

with DAG(
    dag_id="project_pipeline",
    default_args=default_args,
    description="End-to-end Hospital Readmission Prediction Pipeline",
    schedule_interval=None,   # Manual trigger (acceptable for academic submission)
    catchup=False,
    tags=["data-engineering", "ml", "healthcare"],
) as dag:

    # ------------------------
    # Task 1: Data Ingestion
    # ------------------------
    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    # ------------------------
    # Task 2: Data Cleaning
    # ------------------------
    clean_task = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )

    # ------------------------
    # Task 3: Feature Engineering
    # ------------------------
    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=engineer_features,
    )

    # ------------------------
    # Task 4: Model Training
    # ------------------------
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    # ------------------------
    # Task 5: Generate Outputs
    # ------------------------
    generate_outputs_task = PythonOperator(
        task_id="generate_outputs",
        python_callable=generate_outputs,
    )

    # --------------------------------------------------
    # Task dependencies (pipeline order)
    # --------------------------------------------------
    ingest_task >> clean_task >> feature_engineering_task >> train_model_task >> generate_outputs_task