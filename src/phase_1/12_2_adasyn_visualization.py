"""
Figures and model comparison for the ADASYN presentation.

Trains the same two classifiers on the training set without oversampling and
with ADASYN (ratio 0.5 and 1.0), evaluates them on the untouched test set (11B)
and writes 12C_ADASYN_EVALUATION.csv and the figures in docs/adasyn_images/.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "dataset" / "phase_1"
IMAGES = ROOT / "docs" / "adasyn_images"
TARGET = "is_fraud"

spec = importlib.util.spec_from_file_location("adasyn_oversampling", HERE / "12_1_adasyn_oversampling.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "savefig.dpi": 160, "savefig.bbox": "tight",
                     "legend.frameon": False})


def save(figure, name):
    figure.savefig(IMAGES / name)
    plt.close(figure)


def plot_idea(seed=7):
    """Small made-up 2D example showing r_i and where the new points go."""
    rng = np.random.default_rng(seed)
    majority = rng.normal((0, 0), (1.0, 1.0), (220, 2))
    minority = np.vstack([rng.normal((3.4, 0.6), 0.55, (22, 2)), rng.normal((1.5, -0.2), 0.45, (8, 2))])
    X = np.vstack([majority, minority])
    y = np.r_[np.zeros(len(majority)), np.ones(len(minority))]
    synthetic, _, r, _ = core.adasyn(X, y, ratio=0.5, k=5, seed=seed)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis in axes:
        axis.scatter(*majority.T, s=18, c=BLUE, alpha=0.45, label="majority (legitimate)")
    axes[0].scatter(*minority.T, s=40 + 220 * r, c=ORANGE, edgecolors="white",
                    label="minority (fraud), size = difficulty")
    for point, value in zip(minority, r):
        if value >= 0.8:
            axes[0].annotate(f"{value:.1f}", point, xytext=(6, 5), textcoords="offset points", fontsize=8)
    axes[0].set_title("1. Score each fraud point: r = share of majority neighbours")
    axes[1].scatter(*synthetic.T, s=26, c=AQUA, marker="x", label="synthetic fraud")
    axes[1].scatter(*minority.T, s=40, c=ORANGE, edgecolors="white", label="original fraud")
    axes[1].set_title("2. More synthetic points where r is high")
    for axis in axes:
        axis.legend(loc="upper left", fontsize=9)
        axis.set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    save(figure, "1_how_adasyn_works.png")


def plot_counts(train, sets):
    names = list(sets)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    legit = int((train[TARGET] == 0).sum())
    original = int(train[TARGET].sum())
    added = [len(sets[n]) - len(train) for n in names]
    x = np.arange(len(names))
    bars = [axis.bar(x - 0.2, [legit] * len(names), 0.38, color=BLUE, label="legitimate"),
            axis.bar(x + 0.2, [original] * len(names), 0.38, color=ORANGE, label="original fraud"),
            axis.bar(x + 0.2, added, 0.38, bottom=original, color=AQUA, label="synthetic fraud",
                     edgecolor="white", linewidth=1.5)]
    axis.bar_label(bars[0], labels=[f"{legit:,}"] * len(names), padding=2, fontsize=10)
    for i, extra in enumerate(added):
        axis.text(x[i] + 0.2, original + extra + 250, f"{original + extra:,}", ha="center", fontsize=10)
    axis.set_xticks(x, names)
    axis.set_ylim(0, legit * 1.22)
    axis.set_ylabel("training rows")
    axis.set_title("Class balance in the training set")
    axis.legend(ncols=3, loc="upper center")
    save(figure, "2_class_balance.png")


def plot_scatter(train, balanced, seed=42):
    new = balanced.iloc[len(train):]
    rng = np.random.default_rng(seed)

    def pick(frame, share=0.12):                 # same share of every group
        return frame.iloc[rng.choice(len(frame), int(len(frame) * share), replace=False)]

    groups = [(pick(train[train[TARGET] == 0]), BLUE, "o", "legitimate"),
              (pick(train[train[TARGET] == 1]), ORANGE, "o", "original fraud"),
              (pick(new), AQUA, "x", "synthetic fraud")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, shown, title in ((axes[0], groups[:2], "Before ADASYN"), (axes[1], groups, "After ADASYN (1:1)")):
        for frame, colour, marker, label in shown:
            axis.scatter(frame["PC1"], frame["PC2"], s=9, c=colour, marker=marker,
                         alpha=0.5, linewidths=0.8, label=label)
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.legend(loc="best", markerscale=2, fontsize=9)
    axes[0].set_ylabel("PC2")
    figure.text(0.5, -0.02, "Random 12% of each group, first 2 of 14 components", ha="center", fontsize=9)
    save(figure, "3_pca_before_after.png")


def evaluate(train, sets, test):
    features = [c for c in train.columns if c != TARGET]
    models = {"Logistic regression": lambda: LogisticRegression(max_iter=2000),
              "Random forest": lambda: RandomForestClassifier(
                  n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)}
    rows = []
    for model_name, build in models.items():
        for set_name, frame in sets.items():
            model = build().fit(frame[features], frame[TARGET])
            predicted = model.predict(test[features])
            score = model.predict_proba(test[features])[:, 1]
            tn, fp, fn, tp = confusion_matrix(test[TARGET], predicted).ravel()
            rows.append({"model": model_name, "training_data": set_name,
                         "precision": precision_score(test[TARGET], predicted),
                         "recall": recall_score(test[TARGET], predicted),
                         "f1": f1_score(test[TARGET], predicted),
                         "pr_auc": average_precision_score(test[TARGET], score),
                         "true_positives": tp, "false_negatives": fn,
                         "false_positives": fp, "true_negatives": tn})
    return pd.DataFrame(rows).round(4)


def plot_metrics(results):
    metrics = {"precision": "Precision", "recall": "Recall", "f1": "F1", "pr_auc": "PR-AUC"}
    models = results["model"].unique()
    figure, axes = plt.subplots(len(models), 1, figsize=(10, 7), sharex=True)
    for axis, model in zip(axes, models):
        part = results[results["model"] == model]
        x = np.arange(len(metrics))
        for i, ((_, row), colour) in enumerate(zip(part.iterrows(), (BLUE, ORANGE, AQUA))):
            bars = axis.bar(x + (i - 1) * 0.27, [row[m] for m in metrics], 0.25,
                            color=colour, label=row["training_data"])
            axis.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        axis.set_title(model)
        axis.set_ylim(0, 1.15)
        axis.set_xticks(x, list(metrics.values()))
    axes[0].legend(ncols=3, loc="upper center", fontsize=9)
    figure.suptitle("Results on the untouched test set", fontweight="bold")
    figure.tight_layout()
    save(figure, "4_model_metrics.png")


def run():
    IMAGES.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "11A_PCA_TRAIN.csv")
    test = pd.read_csv(DATA / "11B_PCA_TEST.csv")
    features = [c for c in train.columns if c != TARGET]
    sets = {"No oversampling": train}
    for ratio, name in ((0.5, "ADASYN 0.5"), (1.0, "ADASYN 1.0")):
        synthetic, label, _, _ = core.adasyn(train[features].to_numpy(), train[TARGET].to_numpy(), ratio)
        added = pd.DataFrame(synthetic, columns=features).assign(**{TARGET: int(label)})
        sets[name] = pd.concat([train, added], ignore_index=True)

    plot_idea()
    plot_counts(train, sets)
    plot_scatter(train, sets["ADASYN 1.0"])
    results = evaluate(train, sets, test)
    results.to_csv(DATA / "12C_ADASYN_EVALUATION.csv", index=False)
    plot_metrics(results)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run()
