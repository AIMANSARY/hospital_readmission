"""
RQ3 Precision–Recall Comparison Figure
Creates a bar chart comparing precision and recall
for Logistic Regression and Random Forest models.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_precision_recall():
    print("Generating RQ3 precision–recall comparison figure...")

    project_root = Path(__file__).resolve().parents[2]

    metrics_path = project_root / "tables" / "model_comparison_metrics.csv"
    figures_path = project_root / "figures"
    figures_path.mkdir(exist_ok=True)

    # Load model comparison metrics
    df = pd.read_csv(metrics_path)

    models = df["model"]
    precision = df["precision"]
    recall = df["recall"]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x, precision, width, label="Precision")
    plt.bar([i + width for i in x], recall, width, label="Recall")

    plt.xticks([i + width / 2 for i in x], models, rotation=10)
    plt.ylabel("Score")
    plt.title("Precision vs Recall by Model (RQ3)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figures_path / "precision_recall_comparison.png")
    plt.close()

    print("RQ3 figure saved: figures/precision_recall_comparison.png")
