"""
Before-and-after charts for SMOTE, saved to docs/smote_images/.

Standalone: it imports no other project script. All data comes from the files in
dataset/smote/. SMOTE_TRAIN.csv stores the original rows first and
SMOTE_REPORT.csv says how many there are. Only how_smote_works.png is a
schematic with made-up points.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "smote"
IMAGES = ROOT / "docs" / "smote_images"

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a0a0a0"
STYLE = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
         "axes.titleweight": "bold", "legend.frameon": False}


def class_balance(report):
    """Share of each class in the training set, before and after."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for axis, stage, title in zip(axes, ("before", "after"), ("Before SMOTE", "After SMOTE")):
        row = report.loc[stage]
        synthetic = int(row["synthetic_rows"])
        parts = [(int(row["non_fraud"]), BLUE, "legitimate"),
                 (int(row["fraud"]) - synthetic, ORANGE, "original fraud"),
                 (synthetic, AQUA, "synthetic fraud")]
        parts = [part for part in parts if part[0]]
        total = int(row["rows"])
        axis.pie([count for count, _, _ in parts], colors=[colour for _, colour, _ in parts],
                 startangle=90, counterclock=False,
                 wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
                 labels=[f"{count:,}\n{count / total:.1%}" for count, _, _ in parts])
        axis.text(0, 0, f"{total:,}\nrows", ha="center", va="center", fontsize=12)
        axis.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)]
    figure.legend(handles, ["legitimate", "original fraud", "synthetic fraud"], ncols=3,
                  loc="outside lower center")
    return figure


def how_smote_works():
    """Three steps on made-up points."""
    start = np.array([0.0, 0.0])
    near = np.array([[1.6, 1.1], [-1.3, 0.9], [-0.9, -1.2], [1.2, -0.9], [0.2, 1.7]])
    far = np.array([[3.1, 0.4], [-3.0, -0.3], [2.4, -2.1], [-2.5, 2.0]])
    end, new = near[0], near[0] * 0.55

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    titles = ("1. Pick a fraud row", "2. Find its 5 nearest fraud neighbours",
              "3. Create a new row on the line to one of them")
    for step, (axis, title) in enumerate(zip(axes, titles), start=1):
        axis.scatter(*np.vstack([near, far]).T, s=70, c=ORANGE, alpha=0.4)
        if step == 2:
            for other in near:
                axis.plot(*np.c_[start, other], color=GREY, linestyle="--", linewidth=1)
            axis.scatter(*near.T, s=80, c=ORANGE)
        if step == 3:
            axis.plot(*np.c_[start, end], color="#555555", linewidth=1.5)
            axis.scatter(*end, s=80, c=ORANGE, zorder=3)
            axis.scatter(*new, s=220, c=AQUA, marker="*", zorder=4)
            axis.annotate("new synthetic row", new, xytext=(-125, 24), textcoords="offset points")
        axis.scatter(*start, s=130, c=ORANGE, edgecolors="black", linewidths=1.5, zorder=3)
        axis.set_title(title, fontsize=11)
        axis.set(xticks=[], yticks=[], xlim=(-3.6, 3.8), ylim=(-2.7, 2.6))
    return figure


def fraud_share_map(table, original_rows):
    """Share of fraud in each area of the first two components, before and after."""
    extent = (table["PC1"].min(), table["PC1"].max(), table["PC2"].min(), table["PC2"].max())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True,
                                layout="constrained")
    stages = ((table.iloc[:original_rows], "Before SMOTE"), (table, "After SMOTE"))
    for axis, (part, title) in zip(axes, stages):
        cells = axis.hexbin(part["PC1"], part["PC2"], C=100 * part["is_fraud"],
                            reduce_C_function=np.mean, gridsize=35, extent=extent,
                            cmap="Oranges", vmin=0, vmax=100, mincnt=1,
                            edgecolors="#bbbbbb", linewidths=0.3)
        axis.set_title(title)
        axis.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    figure.colorbar(cells, ax=axes, label="fraud rows in the area (%)")
    return figure


def results_change(results):
    """Test-set errors for each training set."""
    errors = (("false_negatives", "Frauds missed"), ("false_positives", "False alarms"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
    models = list(results["model"].unique())
    versions = (("No oversampling", GREY, "o"), ("SMOTE 0.5", ORANGE, "s"), ("SMOTE 1.0", AQUA, "D"))
    for axis, (column, title) in zip(axes, errors):
        for row, model in enumerate(models):
            values = results[results["model"] == model].set_index("training_data")[column]
            axis.plot([values.min(), values.max()], [row, row], color="#d0d0d0", linewidth=3,
                      zorder=1)
            for (version, colour, marker), shift in zip(versions, (12, -20, 12)):
                axis.scatter(values[version], row, s=110, c=colour, marker=marker, zorder=2,
                             label=version if row == 0 else None)
                axis.annotate(f"{values[version]}", (values[version], row), ha="center",
                              xytext=(0, shift), textcoords="offset points", fontsize=10)
        axis.set_title(f"{title} (lower is better)")
        axis.set_xlim(0, results[column].max() * 1.12)
        axis.set_xlabel("test rows")
    axes[0].set_yticks(range(len(models)), models)
    axes[0].set_ylim(len(models) - 0.4, -0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncols=3, loc="outside lower center")
    return figure


def main():
    table = pd.read_csv(DATA / "SMOTE_TRAIN.csv")
    report = pd.read_csv(DATA / "SMOTE_REPORT.csv").set_index("stage")
    results = pd.read_csv(DATA / "SMOTE_EVALUATION.csv")
    if len(table) != report.loc["after", "rows"]:
        raise ValueError("SMOTE_TRAIN.csv and SMOTE_REPORT.csv do not match.")

    IMAGES.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        charts = {"class_balance": class_balance(report),
                  "how_smote_works": how_smote_works(),
                  "fraud_share_map": fraud_share_map(table, int(report.loc["before", "rows"])),
                  "results_change": results_change(results)}
        for name, figure in charts.items():
            figure.savefig(IMAGES / f"{name}.png", dpi=160, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved docs/smote_images/{name}.png")


if __name__ == "__main__":
    main()