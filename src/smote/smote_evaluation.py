"""
Model comparison for SMOTE.

Trains two classifiers on the training set without oversampling and with SMOTE
(ratio 0.5 and 1.0), evaluates them on the untouched test set (11B) and writes
dataset/smote/SMOTE_EVALUATION.csv.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "dataset" / "phase_1"
SMOTE_DIR = HERE.parents[1] / "dataset" / "smote"
TARGET = "is_fraud"

spec = importlib.util.spec_from_file_location("smote_oversampling", HERE / "smote_oversampling.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def evaluate(sets, test):
    features = [c for c in test.columns if c != TARGET]
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


def run():
    train = pd.read_csv(DATA / "11A_PCA_TRAIN.csv")
    test = pd.read_csv(DATA / "11B_PCA_TEST.csv")
    features = [c for c in train.columns if c != TARGET]
    sets = {"No oversampling": train}
    for ratio in (0.5, 1.0):
        synthetic, label, _ = core.smote(train[features].to_numpy(), train[TARGET].to_numpy(), ratio)
        added = pd.DataFrame(synthetic, columns=features).assign(**{TARGET: int(label)})
        sets[f"SMOTE {ratio}"] = pd.concat([train, added], ignore_index=True)

    results = evaluate(sets, test)
    SMOTE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(SMOTE_DIR / "SMOTE_EVALUATION.csv", index=False)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run()