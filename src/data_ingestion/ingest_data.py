import pandas as pd
from pathlib import Path

print("Starting data ingestion...")

# Exact path to raw data
raw_path = r"C:\Users\aimaa\OneDrive\Desktop\hospital_readmission\data\raw\hospital_readmissions.csv"

# Create processed folder
processed_dir = Path(r"C:\Users\aimaa\OneDrive\Desktop\hospital_readmission\data\processed")
processed_dir.mkdir(parents=True, exist_ok=True)

print(f" Loading {raw_path}...")
df = pd.read_csv(raw_path)

print(f" Raw shape: {df.shape}")
print(f" Columns: {list(df.columns)}")

# Basic cleaning
df = df.dropna()          # Drop missing values
df = df.drop_duplicates() # Remove duplicates

print(f"Cleaned shape: {df.shape}")

# Save cleaned data
processed_path = processed_dir / "hospital_readmissions_cleaned.csv"
df.to_csv(processed_path, index=False)

print(f"Cleaned data saved to: {processed_path}")
print("Data ingestion completed successfully!")
