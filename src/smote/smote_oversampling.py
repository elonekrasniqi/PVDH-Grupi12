"""
SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al. 2002)
oversampling of the PCA training set (11A). The test set (11B) is never read or
changed.

  1. G = ratio * majority - minority          synthetic rows to create
  2. every minority row gets G // minority rows; the remainder goes to randomly
     chosen rows, one each
  3. new = x_i + lambda * (x_zi - x_i)        x_zi: one of the k nearest minority
                                              neighbours, lambda ~ U(0, 1)

Outputs: SMOTE_TRAIN.csv (original rows first, synthetic rows after them)
and SMOTE_REPORT.csv (class counts before and after), both in dataset/smote/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "phase_1"
SMOTE_DIR = ROOT / "dataset" / "smote"
DEFAULT_INPUT = DATA / "11A_PCA_TRAIN.csv"
DEFAULT_OUTPUT = SMOTE_DIR / "SMOTE_TRAIN.csv"
DEFAULT_REPORT = SMOTE_DIR / "SMOTE_REPORT.csv"

TARGET = "is_fraud"


def smote(X: np.ndarray, y: np.ndarray, ratio: float = 1.0, k: int = 5, seed: int = 42):
    """Returns (synthetic rows, minority label, synthetic rows per minority row)."""
    labels, sizes = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("SMOTE here needs exactly two classes.")
    if not 0 < ratio <= 1:
        raise ValueError("ratio (minority / majority after sampling) must be in (0, 1].")
    label = labels[np.argmin(sizes)]
    minority = y == label
    n_min, n_maj = int(minority.sum()), int((~minority).sum())
    if k < 1 or n_min <= k:
        raise ValueError("k must be at least 1 and smaller than the minority class size.")
    total = int(round(ratio * n_maj)) - n_min
    if total <= 0:
        raise ValueError("The data already has the requested class ratio.")

    rng = np.random.default_rng(seed)
    g = np.full(n_min, total // n_min)
    g[rng.choice(n_min, total % n_min, replace=False)] += 1

    rows = X[minority]
    model = NearestNeighbors(n_neighbors=k + 1).fit(rows)
    neighbours = model.kneighbors(rows, return_distance=False)
    # a row is not its own neighbour
    neighbours = np.array([found[found != i][:k] for i, found in enumerate(neighbours)])

    base = np.repeat(np.arange(n_min), g)
    partner = neighbours[base, rng.integers(0, k, size=total)]
    step = rng.random((total, 1))
    synthetic = rows[base] + step * (rows[partner] - rows[base])
    return synthetic, label, g


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT,
        ratio: float = 1.0, k: int = 5, seed: int = 42):
    input_path, output_path, report_path = map(Path, (input_path, output_path, report_path))
    if "test" in input_path.name.lower():
        raise ValueError("Never oversample the test set.")
    if input_path.resolve() in (output_path.resolve(), report_path.resolve()):
        raise ValueError("Output paths must not overwrite the input dataset.")
    data = pd.read_csv(input_path)
    if TARGET not in data.columns:
        raise ValueError(f"Missing required column: {TARGET}")
    features = [c for c in data.columns if c != TARGET]
    X = data[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Features contain missing or non-finite values.")
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be 0/1.")

    synthetic, label, _ = smote(X, data[TARGET].to_numpy(), ratio, k, seed)
    added = pd.DataFrame(synthetic, columns=features)
    added[TARGET] = int(label)
    result = pd.concat([data, added[data.columns]], ignore_index=True)

    def counts(stage, frame, new):
        fraud = int(frame[TARGET].sum())
        return {"stage": stage, "rows": len(frame), "non_fraud": len(frame) - fraud,
                "fraud": fraud, "synthetic_rows": new,
                "fraud_share_pct": round(fraud / len(frame) * 100, 2),
                "ratio": ratio, "k_neighbors": k, "seed": seed}

    report = pd.DataFrame([counts("before", data, 0), counts("after", result, len(added))])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)

    print("SMOTE OVERSAMPLING COMPLETED")
    print(report.drop(columns=["ratio", "k_neighbors", "seed"]).to_string(index=False))
    return result, report


def main():
    parser = argparse.ArgumentParser(description="SMOTE oversampling of the training set")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="fraud / non-fraud after sampling (default 1.0)")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output, args.report, args.ratio, args.neighbors, args.seed)


if __name__ == "__main__":
    main()