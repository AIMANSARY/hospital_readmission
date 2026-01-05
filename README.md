Hospital Readmission Prediction
Project Overview

Hospital readmissions are costly and often preventable.
This project builds a machine learning pipeline to predict whether a patient is likely to be readmitted to the hospital, helping healthcare providers take proactive action and improve patient outcomes.

The project implements an end-to-end machine learning workflow including:

Data ingestion

Data cleaning & sampling

Feature engineering

Model training & evaluation

Reproducible project structure

Objective

To predict hospital readmission (Yes / No) using patient and admission-related data, with a focus on:

Clean data pipelines

Reproducible experiments

Practical ML engineering practices

Tech Stack

Python 3

Pandas, NumPy – data processing

Scikit-learn – machine learning

VS Code – development

Git & GitHub – version control

Dataset link

Research Questions Addressed

 Project Structure

 ```text
hospital_readmission/
│
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Cleaned / transformed data (ignored in Git)
│
├── src/
│   ├── data_ingestion/   # Data loading & sampling
│   ├── data_cleaning/    # Cleaning & preprocessing
│   ├── feature_engineering/
│   ├── modeling/         # Model training scripts
│   ├── evaluation/       # Evaluation & outputs
│
├── notebooks/            # Experiments & analysis
├── figures/              # Visualizations
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

