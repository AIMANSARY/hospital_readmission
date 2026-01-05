Hospital Readmission Prediction using Feature-Store-Driven ML Pipeline

Project Overview
Hospital readmissions within 30 days are a major challenge for healthcare systems, impacting patient outcomes and operational costs.
This project implements an end-to-end machine learning pipeline to predict hospital readmissions using a feature-store-based architecture, multiple models, and systematic evaluation.

The pipeline is designed to be modular, reproducible, and extensible, following best practices in data engineering and applied machine learnin

Objectives

1. Build a feature-store-driven data pipeline for hospital readmission prediction

2. Compare multiple machine learning models on the same feature set

3. Identify the most influential clinical features driving readmission risk

4. Analyze model trade-offs relevant to real-world healthcare decision-making


Research Questions Addressed
RQ1: How effectively can a feature-store-based pipeline improve hospital readmission prediction?
Answered by comparing model performance using a shared engineered feature set.

RQ2: Which clinical features contribute most strongly to readmission risk?
Answered using Random Forest feature importance analysis.

RQ3:How does model choice affect the trade-off between precision and recall in hospital readmission prediction?
Answered through metric comparison and a precision–recall visualization.

Project Structure
 ```text
hospital_readmission/
│
├── data/
│   ├── raw/                 # Original dataset
│   ├── cleaned/             # Cleaned data
│   └── processed/           # Feature-store-ready dataset
│
├── src/
│   ├── feature_engineering/
│   │   └── build_features.py
│   ├── modeling/
│   │   ├── train_model.py
│   │   ├── compare_models.py
│   │   └── feature_importance.py
│   └── evaluation/
│       └── rq3_precision_recall_plot.py
│
├── dags/
│   └── project_pipeline_dag.py
│
├── tables/
│   ├── baseline_model_metrics.csv
│   ├── model_comparison_metrics.csv
│   └── feature_importance_rf.csv
│
├── figures/
│   ├── model_comparison_roc.png
│   ├── feature_importance_rf.png
│   └── precision_recall_comparison.png
│
├── README.md
└── requirements.txt

```

