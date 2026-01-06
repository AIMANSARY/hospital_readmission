import pandas as pd
from pathlib import Path

def export_tables_to_xlsx():
    project_root = Path(__file__).resolve().parents[2]

    tables_src = project_root / "tables"
    tables_dst = project_root / "Figures_and_Tables" / "Tables"
    tables_dst.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # RQ1 – Model comparison
    # -------------------------
    pd.read_csv(
        tables_src / "model_comparison_metrics.csv"
    ).to_excel(
        tables_dst / "RQ1_Table1.xlsx",
        index=False
    )

    # -------------------------
    # RQ2 – Feature importance
    # -------------------------
    pd.read_csv(
        tables_src / "feature_importance_rf.csv"
    ).to_excel(
        tables_dst / "RQ2_Table1.xlsx",
        index=False
    )

    # -------------------------
    # RQ3 – Precision / Recall trade-off
    # (Derived from model comparison)
    # -------------------------
    pd.read_csv(
        tables_src / "model_comparison_metrics.csv"
    ).to_excel(
        tables_dst / "RQ3_Table1.xlsx",
        index=False
    )

    print("All tables exported as XLSX with correct RQ naming.")

if __name__ == "__main__":
    export_tables_to_xlsx()

