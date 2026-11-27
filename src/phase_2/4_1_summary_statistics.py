"""
Summary statistics (data exploration, one attribute at a time)

Input:
  3A_OUTLIER_TREATED_DATASET.csv  - 8A data + confirmed outlier flags + *_capped columns

Statistics
  numeric attributes      - location (mean, median, mode, trimmed mean, percentiles),
                            spread (range, variance, std, IQR, AAD, MAD, CV),
                            shape (skewness, excess kurtosis)
  categorical attributes  - frequency, percentage, mode, fraud rate per value

Outputs:
  4A_SUMMARY_STATISTICS.csv           - one row per numeric attribute
  4B_SUMMARY_STATISTICS_BY_CLASS.csv  - the same statistics for fraud and non-fraud
  4C_CATEGORICAL_FREQUENCIES.csv      - one row per value of every categorical attribute
                                        (spellings that differ only in letter case are merged)
  4D_TREATMENT_COMPARISON.csv         - original vs capped values (after outlier treatment)
  4E_NUMERIC_DISTRIBUTIONS.png        - histogram with mean and median per numeric attribute
  4F_CATEGORICAL_FREQUENCIES.png      - frequency and fraud rate of the main categories
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHASE_2 = ROOT / "dataset" / "phase_2"
INPUT = PHASE_2 / "3A_OUTLIER_TREATED_DATASET.csv"
OUTPUT_FOLDER = PHASE_2

TARGET = "is_fraud"
TRIM = 0.10   # share cut from EACH end for the trimmed mean

NUMERIC = [
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

# transaction_hour is cyclic (23 is next to 0), so it is counted, not averaged
CATEGORICAL = [
    TARGET, "merchant_category", "merchant_type", "currency", "country", "city_size",
    "card_type", "card_brand", "device", "channel", "card_present", "high_risk_merchant",
    "distance_from_home", "weekend_transaction", "is_shared_device", "transaction_hour",
    "transaction_time_period", "transaction_amount_category", "transaction_frequency_category",
    "customer_activity_level", "customer_spending_level",
]

PLOT_CATEGORIES = ["merchant_category", "country", "card_type", "channel", "device",
                   "transaction_hour"]

NOTES = {
    "velocity_total_amount": "sum in local currencies: not comparable between currencies",
    "velocity_max_single_amount": "local currencies: not comparable between currencies",
    "customer_avg_amount_usd": "customer level: repeated for every transaction of a customer",
    "customer_txn_count": "customer level: repeated for every transaction of a customer",
    "customer_away_share": "customer level: repeated for every transaction of a customer",
    "customer_night_share": "customer level: repeated for every transaction of a customer",
}
LOG_PLOTS = ("amount_usd", "customer_avg_amount_usd", "velocity_total_amount",
             "velocity_max_single_amount")


def load(path):
    path = Path(path)
    with open(path, "rb") as file:
        if file.read(40).startswith(b"version https://git-lfs"):
            raise ValueError(f"{path.name} is a Git LFS pointer, not the data. Run: git lfs pull")
    return pd.read_csv(path, low_memory=False)


def fraud_flag(column):
    values = column.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    if values.isna().any():
        raise ValueError(f"{TARGET} has values other than True/False or 1/0")
    return values.astype(int)


def trimmed_mean(values, share=TRIM):
    ordered = np.sort(values.to_numpy())
    cut = int(len(ordered) * share)
    return ordered[cut:len(ordered) - cut].mean()


def describe_numeric(column):
    values = pd.to_numeric(column, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError(f"{column.name} has no numeric values")
    q = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    mean, median, std = values.mean(), q[0.5], values.std()
    modes = values.mode()
    return {
        "count": len(values),
        "missing": int(len(column) - len(values)),
        "unique": int(values.nunique()),
        "mean": mean,
        "median": median,
        "mode": modes.iloc[0],
        "mode_frequency": int((values == modes.iloc[0]).sum()),
        "trimmed_mean_10": trimmed_mean(values),
        "min": values.min(),
        "p01": q[0.01], "p05": q[0.05], "q1": q[0.25], "q3": q[0.75],
        "p95": q[0.95], "p99": q[0.99],
        "max": values.max(),
        "range": values.max() - values.min(),
        "variance": values.var(),
        "std": std,
        "iqr": q[0.75] - q[0.25],
        "aad": (values - mean).abs().mean(),        # average absolute deviation
        "mad": (values - median).abs().median(),    # median absolute deviation
        "cv": std / mean if mean != 0 else np.nan,
        "skewness": values.skew(),
        "kurtosis": values.kurt(),
    }


def numeric_summary(df):
    rows = [{"feature": f, **describe_numeric(df[f]), "note": NOTES.get(f, "")}
            for f in NUMERIC if f in df.columns]
    return pd.DataFrame(rows).round(4)


def class_summary(df):
    fraud = fraud_flag(df[TARGET])
    rows = []
    for feature in [f for f in NUMERIC if f in df.columns]:
        for label, name in ((0, "non-fraud"), (1, "fraud")):
            part = df.loc[fraud == label, feature]
            if part.notna().any():
                rows.append({"feature": feature, "class": name, **describe_numeric(part)})
    return pd.DataFrame(rows).round(4)


def merge_case_variants(values):
    """'BRAZIL', 'brazil', 'Brazil' -> one value, written like its most common spelling."""
    text = values.astype(str).str.strip()
    key = text.str.casefold()
    common = text.groupby(key).agg(lambda spellings: spellings.value_counts().index[0])
    return key.map(common)


def categorical_frequencies(df):
    fraud = fraud_flag(df[TARGET])
    tables = []
    for feature in [f for f in CATEGORICAL if f in df.columns]:
        values = df[feature].astype(object).where(df[feature].notna(), "Missing")
        if values.map(lambda value: isinstance(value, str)).any():
            values = merge_case_variants(values)
        table = fraud.groupby(values).agg(count="size", fraud_count="sum")
        table.index.name = "value"
        table = table.reset_index()
        table.insert(0, "feature", feature)
        table["percentage"] = 100 * table["count"] / len(df)
        table["fraud_rate"] = table["fraud_count"] / table["count"]
        table["is_mode"] = (table["count"] == table["count"].max()).astype(int)
        tables.append(table.sort_values("count", ascending=False, kind="stable"))
    return pd.concat(tables, ignore_index=True).round(4)


def treatment_comparison(df):
    rows = []
    for feature in NUMERIC:
        capped = feature + "_capped"
        if feature not in df.columns or capped not in df.columns:
            continue
        before, after = describe_numeric(df[feature]), describe_numeric(df[capped])
        changed = int((~np.isclose(df[feature], df[capped])).sum())
        for statistic in ("mean", "median", "trimmed_mean_10", "std", "iqr", "mad",
                          "max", "skewness", "kurtosis"):
            rows.append({"feature": feature, "statistic": statistic,
                         "original": before[statistic], "capped": after[statistic],
                         "difference": after[statistic] - before[statistic],
                         "rows_changed": changed})
    columns = ["feature", "statistic", "original", "capped", "difference", "rows_changed"]
    return pd.DataFrame(rows, columns=columns).round(4)


def plot_numeric(df, path):
    features = [f for f in NUMERIC if f in df.columns]
    rows = (len(features) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        mean, median = values.mean(), values.median()
        title = feature
        if feature in LOG_PLOTS and (values >= 0).all():
            values, mean, median = np.log10(values + 1), np.log10(mean + 1), np.log10(median + 1)
            title += "  [log10(x + 1)]"
        ax.hist(values, bins=40, color="#4C72B0")
        ax.axvline(mean, color="red", linestyle="--", label="mean")
        ax.axvline(median, color="black", label="median")
        ax.set_title(title)
        ax.set_ylabel("rows")
        ax.legend()
    for ax in axes.flat[len(features):]:
        ax.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_categorical(frequencies, overall_fraud_rate, path):
    features = [f for f in PLOT_CATEGORIES if f in set(frequencies["feature"])]
    if not features:
        return
    fig, axes = plt.subplots(len(features), 2, figsize=(14, 4 * len(features)), squeeze=False)
    for (left, right), feature in zip(axes, features):
        table = frequencies[frequencies["feature"] == feature]
        if feature == "transaction_hour":
            table = table.sort_values("value")
        labels = table["value"].astype(str)
        left.bar(labels, table["count"], color="#4C72B0")
        left.set_title(f"{feature}: frequency")
        right.bar(labels, 100 * table["fraud_rate"], color="#C44E52")
        right.axhline(100 * overall_fraud_rate, color="black", linestyle="--", label="all rows")
        right.set_title(f"{feature}: fraud rate (%)")
        right.legend()
        for ax in (left, right):
            ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(input_path=INPUT, output_folder=OUTPUT_FOLDER):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = load(input_path)
    if TARGET not in df.columns:
        raise ValueError(f"Input is missing the column {TARGET}")
    missing = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]

    summary = numeric_summary(df)
    by_class = class_summary(df)
    frequencies = categorical_frequencies(df)
    comparison = treatment_comparison(df)

    summary.to_csv(output_folder / "4A_SUMMARY_STATISTICS.csv", index=False)
    by_class.to_csv(output_folder / "4B_SUMMARY_STATISTICS_BY_CLASS.csv", index=False)
    frequencies.to_csv(output_folder / "4C_CATEGORICAL_FREQUENCIES.csv", index=False)
    comparison.to_csv(output_folder / "4D_TREATMENT_COMPARISON.csv", index=False)
    plot_numeric(df, output_folder / "4E_NUMERIC_DISTRIBUTIONS.png")
    fraud_rate = fraud_flag(df[TARGET]).mean()
    plot_categorical(frequencies, fraud_rate, output_folder / "4F_CATEGORICAL_FREQUENCIES.png")

    print("SUMMARY STATISTICS")
    print(f"Rows: {len(df):,} | fraud rate: {fraud_rate:.2%}")
    if missing:
        print(f"Columns not found (skipped): {', '.join(missing)}")
    print()
    print(summary[["feature", "mean", "median", "mode", "std", "iqr", "skewness"]].to_string(index=False))
    print("\nMost frequent value per categorical attribute")
    modes = frequencies[frequencies["is_mode"] == 1]
    print(modes[["feature", "value", "count", "percentage", "fraud_rate"]].to_string(index=False))
    return summary, by_class, frequencies, comparison


if __name__ == "__main__":
    main()