from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .pipeline import DEFAULT_CLASSIFIERS, DEFAULT_POSES, run_background_comparison
except ImportError:
    from pipeline import DEFAULT_CLASSIFIERS, DEFAULT_POSES, run_background_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def _safe_slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def save_confusion_matrix_figure(
    confusion_matrix_data,
    labels: list[str],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(confusion_matrix_data, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    threshold = confusion_matrix_data.max() / 2 if confusion_matrix_data.size else 0
    for row_idx in range(confusion_matrix_data.shape[0]):
        for col_idx in range(confusion_matrix_data.shape[1]):
            color = "white" if confusion_matrix_data[row_idx, col_idx] > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                str(confusion_matrix_data[row_idx, col_idx]),
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_comparison_chart(summary_df: pd.DataFrame, save_path: Path) -> None:
    pivot = summary_df.pivot(index="Experience", columns="Classifier", values="Accuracy (%)")
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0, color=["#2563eb", "#0f766e", "#ea580c"])
    ax.set_title("Comparaison de precision HOG par experience et classifieur")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Experience")
    ax.legend(title="Classifier")

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    fig = ax.get_figure()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, max(2.8, 0.55 * len(summary_df) + 1.8)))
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.round(2).values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.12, 1.4)
    ax.set_title("Resultats comparatifs HOG + SVM/KNN", fontsize=14, pad=16)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("YOGA AI - Comparaison HOG avec SVM et KNN")

    outputs = run_background_comparison(
        data_root=DATA_ROOT,
        poses=DEFAULT_POSES,
        classifiers=DEFAULT_CLASSIFIERS,
    )

    summary = outputs.get("summary", pd.DataFrame())
    if summary.empty:
        print("Aucune experience n'a pu etre executee. Verifie le dataset dans data/raw et data/raw_sans_fond.")
        return

    print("\nResultats:")
    print(summary.round(2).to_string(index=False))

    improvement_by_classifier = outputs.get("improvement_by_classifier", {})
    if improvement_by_classifier:
        print("\nGain sans fond par classifieur:")
        for classifier, gain in improvement_by_classifier.items():
            print(f"- {classifier}: {gain:+.2f}%")

    for model_name, payload in outputs.get("experiments", {}).items():
        save_path = PLOTS_DIR / f"confusion_{_safe_slug(model_name)}.png"
        save_confusion_matrix_figure(
            confusion_matrix_data=payload["confusion_matrix"],
            labels=payload["labels"],
            title=f"Matrice de confusion - {model_name}",
            save_path=save_path,
        )
        print(f"Matrice sauvegardee: {save_path}")

    comparison_chart_path = PLOTS_DIR / "comparaison_fond.png"
    save_comparison_chart(summary, comparison_chart_path)
    print(f"Graphique comparatif sauvegarde: {comparison_chart_path}")

    summary_table_path = PLOTS_DIR / "comparaison_modeles_table.png"
    save_summary_table(summary, summary_table_path)
    print(f"Tableau comparatif sauvegarde: {summary_table_path}")


if __name__ == "__main__":
    main()
