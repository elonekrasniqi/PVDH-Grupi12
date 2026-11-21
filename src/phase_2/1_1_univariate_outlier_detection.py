"""
Univariate outlier detection: Z-score and IQR

Outputs:
  1A_UNIVARIATE_OUTLIERS_DATASET.csv  - the 8A data + outlier flag columns
  1B_UNIVARIATE_OUTLIER_REPORT.csv    - bounds and number of outliers per feature and method
  1C_UNIVARIATE_OUTLIER_PLOTS.png     - one boxplot per feature
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "dataset" / "phase_1" / "8A_TRANSFORMED_NORMALIZED_DATASET.csv"
OUTPUT_FOLDER = ROOT / "dataset" / "phase_2"

FEATURES = [
    "amount_usd",
    "customer_avg_amount_usd",
    "customer_txn_count",
    "customer_away_share",
    "customer_night_share",
    "customers_per_device",
    "velocity_num_transactions",
    "velocity_total_amount",
    "velocity_unique_merchants",
    "velocity_unique_countries",
    "velocity_max_single_amount",
]

# strongly skewed features: their log z-score was already computed in Phase I (8A)
LOG_Z_COLUMNS = {
    "amount_usd": "amount_usd_log_z",
    "customer_avg_amount_usd": "customer_avg_amount_usd_log_z",
    "velocity_total_amount": "velocity_total_amount_log_z",
}

Z_THRESHOLD = 3
IQR_FACTOR = 1.5
IQR_EXTREME_FACTOR = 3


def zscore_outliers(values):
    mean = values.mean()
    std = values.std(ddof=0)
    lower = mean - Z_THRESHOLD * std
    upper = mean + Z_THRESHOLD * std
    outliers = ((values - mean) / std).abs() > Z_THRESHOLD
    return outliers, lower, upper


def zscore_log_outliers(values, log_z):
    """Flags from the existing *_log_z column; bounds shown on the original scale."""
    outliers = log_z.abs() > Z_THRESHOLD
    log_values = np.log1p(values)
    lower = np.expm1(log_values.mean() - Z_THRESHOLD * log_values.std(ddof=0))
    upper = np.expm1(log_values.mean() + Z_THRESHOLD * log_values.std(ddof=0))
    return outliers, lower, upper


def iqr_outliers(values, factor=IQR_FACTOR):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    if iqr == 0:
        # more than half of the rows have the same value -> IQR cannot be used
        return pd.Series(False, index=values.index), lower, upper
    outliers = (values < lower) | (values > upper)
    return outliers, lower, upper


def detect(df):
    df = df.copy()
    report = []

    for feature in FEATURES:
        values = df[feature]
        methods = [("Z-score", "_z_outlier", zscore_outliers(values))]
        if feature in LOG_Z_COLUMNS:
            methods.append(("Z-score (log)", "_zlog_outlier",
                            zscore_log_outliers(values, df[LOG_Z_COLUMNS[feature]])))
        methods.append(("IQR", "_iqr_outlier", iqr_outliers(values, IQR_FACTOR)))
        methods.append(("IQR (extreme)", "_iqr_extreme_outlier", iqr_outliers(values, IQR_EXTREME_FACTOR)))

        for method, suffix, (outliers, lower, upper) in methods:
            df[feature + suffix] = outliers.astype(int)
            report.append({
                "feature": feature,
                "method": method,
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "outliers": int(outliers.sum()),
                "outliers_pct": round(100 * outliers.mean(), 3),
                "note": "IQR = 0, not used" if method.startswith("IQR") and lower == upper else "",
            })

    # primary methods only: Z-score and IQR (1.5)
    primary_columns = [f + suffix for f in FEATURES for suffix in ("_z_outlier", "_iqr_outlier")]
    df["outlier_count"] = df[primary_columns].sum(axis=1)
    df["is_univariate_outlier"] = (df["outlier_count"] > 0).astype(int)

    # all four methods: Z-score, Z-score (log), IQR (1.5), IQR (extreme)
    all_columns = [c for c in df.columns if c.endswith("_outlier") and c != "is_univariate_outlier"]
    df["outlier_count_all_methods"] = df[all_columns].sum(axis=1)
    df["is_univariate_outlier_all_methods"] = (df["outlier_count_all_methods"] > 0).astype(int)

    return df, pd.DataFrame(report)


def plot(df, path):
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(10, 2 * len(FEATURES)))
    for ax, feature in zip(axes, FEATURES):
        ax.boxplot(df[feature], vert=False, flierprops={"marker": "x", "markeredgecolor": "red"})
        title = feature
        if feature in LOG_Z_COLUMNS:
            ax.set_xscale("symlog", linthresh=1)
            title += " (log scale)"
        if df[feature].quantile(0.75) == df[feature].quantile(0.25):
            title += " (IQR = 0, IQR not used)"
        ax.set_title(title)
        ax.set_yticks([])
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def main(input_path=INPUT, output_folder=OUTPUT_FOLDER):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    result, report = detect(df)

    result.to_csv(output_folder / "1A_UNIVARIATE_OUTLIERS_DATASET.csv", index=False)
    report.to_csv(output_folder / "1B_UNIVARIATE_OUTLIER_REPORT.csv", index=False)
    plot(result, output_folder / "1C_UNIVARIATE_OUTLIER_PLOTS.png")

    print("UNIVARIATE OUTLIER DETECTION (Z-score, IQR)")
    print(f"Rows: {len(result):,}")
    print(f"Rows with at least one outlier (Z-score, IQR): {result['is_univariate_outlier'].sum():,}")
    print(f"Rows with at least one outlier (all 4 methods): "
          f"{result['is_univariate_outlier_all_methods'].sum():,}\n")
    print(report.to_string(index=False))
    return result, report


if __name__ == "__main__":
    main()