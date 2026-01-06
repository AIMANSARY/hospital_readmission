import matplotlib.pyplot as plt
from pathlib import Path

def export_figures_to_pdf():
    project_root = Path(__file__).resolve().parents[2]

    figures_src = project_root / "figures"
    figures_dst = project_root / "Figures_and_Tables" / "Figures"
    figures_dst.mkdir(parents=True, exist_ok=True)

    mapping = {
        "model_comparison_roc.png": "RQ1_Fig1.pdf",
        "feature_importance_rf.png": "RQ2_Fig1.pdf",
        "precision_recall_comparison.png": "RQ3_Fig1.pdf",
    }

    for src, dst in mapping.items():
        img = plt.imread(figures_src / src)
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(figures_dst / dst, format="pdf", bbox_inches="tight")
        plt.close()

    print("All figures exported as PDF with correct names.")

if __name__ == "__main__":
    export_figures_to_pdf()
