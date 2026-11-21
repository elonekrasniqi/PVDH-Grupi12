
"""Phase II Task E: Isolation Forest and Local Outlier Factor on Phase I 8A.

Run from repository root:
python src/phase_2/2_1_multivariate_outlier_detection.py
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "dataset/phase_1/8A_TRANSFORMED_NORMALIZED_DATASET.csv"
OUTPUT = ROOT / "dataset/phase_2"

FEATURES = [
    "amount_usd_log_z",
    "customer_avg_amount_usd_log_z",
    "velocity_total_amount_log_z",
    "customer_txn_count_z",
    "velocity_num_transactions_z",
    "velocity_max_single_amount_z",
    "velocity_unique_merchants_z",
    "velocity_unique_countries_z",
    "customer_away_share_minmax",
    "customer_night_share_minmax",
    "customers_per_device_minmax",
    "hour_sin",
    "hour_cos",
]


def detect(df: pd.DataFrame, contamination: float = 0.05, neighbors: int = 20):

    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    if len(df) < 4:
        raise ValueError("At least 4 rows required")

    missing = sorted(set(FEATURES) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required scaled columns in 8A: {missing}")

    x = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(
            "Selected features contain missing or non-finite values; inspect 8A first"
        )

    # Standardize features for distance-based detection.
    x_scaled = StandardScaler().fit_transform(x)

    # Isolation Forest
    forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    if_labels = forest.fit_predict(x_scaled)

    # Local Outlier Factor
    lof = LocalOutlierFactor(
        n_neighbors=min(neighbors, len(df) - 1),
        contamination=contamination,
        n_jobs=-1,
    )

    lof_labels = lof.fit_predict(x_scaled)

    # Save detection results
    out = pd.DataFrame({
        "source_row": np.arange(len(df)),
        "isolation_forest_outlier": (if_labels == -1).astype(int),
        "isolation_forest_anomaly_score": -forest.score_samples(x_scaled),
        "lof_outlier": (lof_labels == -1).astype(int),
        "lof_anomaly_score": -lof.negative_outlier_factor_,
    })

    # Preserve identifiers and fraud labels for later analysis.
    for identifier in ("transaction_id", "customer_id", "is_fraud"):
        if identifier in df.columns:
            out.insert(1, identifier, df[identifier].to_numpy())

    out["both_methods_outlier"] = (
        out["isolation_forest_outlier"] & out["lof_outlier"]
    ).astype(int)

    report = {
        "input_rows": len(df),
        "features": FEATURES,
        "contamination_assumption": contamination,
        "lof_neighbors": min(neighbors, len(df) - 1),
        "isolation_forest_outliers": int(out.isolation_forest_outlier.sum()),
        "lof_outliers": int(out.lof_outlier.sum()),
        "both_methods_outliers": int(out.both_methods_outlier.sum()),
        "note": (
            "Outlier flags are unsupervised anomaly indicators, "
            "not fraud predictions."
        ),
    }

    # Evaluate the detected anomalies against the fraud label.
    if "is_fraud" in out.columns:
        labels = pd.to_numeric(out["is_fraud"], errors="coerce")

        if labels.notna().all() and labels.isin([0, 1]).all():
            for method in ("isolation_forest", "lof"):
                mask = out[f"{method}_outlier"].eq(1)

                report[f"{method}_fraud_count_among_outliers"] = int(
                    labels[mask].sum()
                )

                report[f"{method}_fraud_rate_among_outliers"] = (
                    float(labels[mask].mean()) if mask.any() else None
                )

    return out, report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--neighbors", type=int, default=20)

    args = parser.parse_args()

    if args.neighbors < 2:
        parser.error("--neighbors must be at least 2")

    if not args.input.exists():
        parser.error(f"File not found: {args.input}")

    # Verify that Git LFS downloaded the actual CSV.
    with args.input.open("r", encoding="utf-8") as f:
        if f.readline().startswith("version https://git-lfs.github.com/spec"):
            parser.error("8A is a Git LFS pointer. Run: git lfs pull")

    df = pd.read_csv(args.input, low_memory=False)

    results, report = detect(
        df,
        contamination=args.contamination,
        neighbors=args.neighbors,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        args.output_dir / "2E_MULTIVARIATE_OUTLIER_RESULTS.csv",
        index=False,
    )

    (args.output_dir / "2E_MULTIVARIATE_OUTLIER_REPORT.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
