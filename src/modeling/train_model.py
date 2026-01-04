import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("Hospital Readmission Predictor - TRAINING")

# --- Resolve project root safely ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hospital_readmissions_cleaned.csv"

print("Loading processed data from:")
print(DATA_PATH)
print("File exists:", DATA_PATH.exists())

df = pd.read_csv(DATA_PATH)
