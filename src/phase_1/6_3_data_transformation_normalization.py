"""
Data transformation and normalization of the discretization & binarization
output (6B).

Transformation
  - velocity_last_hour text  -> 5 numeric velocity_* columns
  - transaction_hour         -> hour_sin / hour_cos (cyclic: 23h is next to 0h)
  - Low/Medium/High levels   -> ordinal codes 0 / 1 / 2
  - transaction_time_period  -> one-hot time_period_* columns
  - strongly skewed money values -> log1p (log(1 + x))
  - invalid (<= 0) amount_usd and empty values -> median, with a flag column

Normalization
  - z-score  (mean 0, std 1) for the log-transformed and other unbounded numbers
  - min-max  ([0, 1])        for bounded numbers and the ordinal codes

All 6B columns are kept; the new columns are added at the end. The scaling
parameters (mean/std, min/max) are written to the report so the same scaling
can be repeated on new data.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "phase_1"
DEFAULT_INPUT = DATA / "6B_DISCRETIZED_BINARIZED_DATASET.csv"
DEFAULT_OUTPUT = DATA / "6D_TRANSFORMED_NORMALIZED_DATASET.csv"
DEFAULT_REPORT = DATA / "6E_TRANSFORMATION_NORMALIZATION_REPORT.csv"

TARGET = "is_fraud"
UNKNOWN = "Unknown"

VELOCITY_KEYS = ("num_transactions", "total_amount", "unique_merchants",
                 "unique_countries", "max_single_amount")

# discretized columns from 6B (ordered levels -> 0, 1, 2)
ORDINAL_COLUMNS = ("transaction_amount_category", "transaction_frequency_category",
                   "customer_activity_level", "customer_spending_level")
LEVEL_CODES = {"Low": 0, "Medium": 1, "High": 2}
TIME_PERIODS = ("Night", "Morning", "Afternoon", "Evening")

# strongly right-skewed (skew > 1 on 6B: 36.1, 20.5, 5.2), non-negative: log1p, then z-score
LOG_Z = ("amount_usd", "customer_avg_amount_usd", "velocity_total_amount")
# unbounded but only mildly skewed (|skew| < 1) or left-skewed: z-score only
# (log1p would push them to a strong left skew, e.g. customer_txn_count 0.84 -> -4.35)
Z_ONLY = ("customer_txn_count", "velocity_num_transactions", "velocity_max_single_amount",
          "velocity_unique_merchants", "velocity_unique_countries")
# bounded values: min-max to [0, 1]
MIN_MAX = ("customer_away_share", "customer_night_share", "customers_per_device",
           *[f"{c}_code" for c in ORDINAL_COLUMNS])

POSITIVE_ONLY = ("amount_usd",)   # <= 0 is the converted -999 placeholder, not a real amount


def to_number(column: pd.Series) -> pd.Series:
    return pd.to_numeric(column, errors="coerce")


def skew(values: pd.Series):
    value = values.skew()
    return round(float(value), 4) if pd.notna(value) else ""


# --- transformation -----------------------------------------------------------

def parse_velocity(text: str) -> dict:
    """velocity_last_hour text -> dict; works with 'single' and "double" quotes."""
    try:
        values = ast.literal_eval(str(text).strip())
        return {key: float(values[key]) for key in VELOCITY_KEYS}
    except (ValueError, SyntaxError, KeyError, TypeError, AttributeError):
        return {key: float("nan") for key in VELOCITY_KEYS}


def add_velocity_columns(result: pd.DataFrame) -> None:
    parsed = pd.DataFrame(list(result["velocity_last_hour"].map(parse_velocity)), index=result.index)
    for key in VELOCITY_KEYS:
        result[f"velocity_{key}"] = parsed[key]


def add_cyclic_hour(result: pd.DataFrame) -> None:
    hour = to_number(result["transaction_hour"])
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24).round(6)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24).round(6)


def add_ordinal_codes(result: pd.DataFrame) -> None:
    for column in ORDINAL_COLUMNS:
        result[f"{column}_code"] = result[column].astype(str).str.strip().map(LEVEL_CODES)  # Unknown -> NaN


def add_time_period_one_hot(result: pd.DataFrame) -> None:
    period = result["transaction_time_period"].astype(str).str.strip()
    for name in TIME_PERIODS:
        result[f"time_period_{name.lower()}"] = (period == name).astype(int)


def impute(result: pd.DataFrame, columns) -> dict:
    """Invalid/empty values -> median of the valid ones. Returns {column: (rows imputed, median)}."""
    info = {}
    for column in columns:
        values = to_number(result[column])
        if column in POSITIVE_ONLY:
            invalid = values.isna() | (values <= 0)
            result[f"{column}_invalid"] = invalid.astype(int)
            values = values.mask(invalid)
        median = values.median()
        if pd.isna(median):
            raise ValueError(f"{column} has no valid values to compute a median")
        info[column] = (int(values.isna().sum()), float(median))
        result[column] = values.fillna(median)
    return info


def transform(data: pd.DataFrame):
    result = data.copy()
    add_velocity_columns(result)
    add_cyclic_hour(result)
    add_ordinal_codes(result)
    add_time_period_one_hot(result)
    imputed = impute(result, [*LOG_Z, *Z_ONLY, *MIN_MAX])
    return result, imputed


# --- normalization ------------------------------------------------------------

def z_score(values: pd.Series):
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if std == 0:
        return values * 0.0, mean, 1.0        # constant column -> all 0
    return (values - mean) / std, mean, std


def min_max(values: pd.Series):
    low, high = float(values.min()), float(values.max())
    if high == low:
        return values * 0.0, low, 1.0
    return (values - low) / (high - low), low, high - low


def normalize(result: pd.DataFrame, imputed: dict):
    rows = []

    def record(source, column, transformation, normalization, before, center, scale):
        after = result[column]
        rows.append({
            "feature": column, "source": source,
            "transformation": transformation, "normalization": normalization,
            "imputed_rows": imputed[source][0], "imputed_with_median": round(imputed[source][1], 4),
            "skew_before": skew(before), "skew_after": skew(after),
            "center": round(center, 6), "scale": round(scale, 6),
            "min_after": round(float(after.min()), 4), "max_after": round(float(after.max()), 4),
            "mean_after": round(float(after.mean()), 4), "std_after": round(float(after.std(ddof=0)), 4),
        })

    for source in LOG_Z:
        before = result[source].astype(float)
        column = f"{source}_log_z"
        result[column], center, scale = z_score(np.log1p(before))
        record(source, column, "log1p", "z-score", before, center, scale)

    for source in Z_ONLY:
        before = result[source].astype(float)
        column = f"{source}_z"
        result[column], center, scale = z_score(before)
        record(source, column, "none", "z-score", before, center, scale)

    for source in MIN_MAX:
        before = result[source].astype(float)
        column = f"{source}_minmax"
        result[column], center, scale = min_max(before)
        transformation = "ordinal encoding (Low=0, Medium=1, High=2)" if source.endswith("_code") else "none"
        record(source, column, transformation, "min-max [0, 1]", before, center, scale)

    for column in [c for c in result.columns if c.endswith(("_log_z", "_z", "_minmax"))]:
        result[column] = result[column].round(6)
    return result, pd.DataFrame(rows)


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT):
    data = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    required = {TARGET, "velocity_last_hour", "transaction_hour", "transaction_time_period",
                "amount_usd", "customer_avg_amount_usd", "customer_txn_count",
                "customer_away_share", "customer_night_share", "customers_per_device", *ORDINAL_COLUMNS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    result, imputed = transform(data)
    result, report = normalize(result, imputed)
    if len(result) != len(data):
        raise ValueError("Row count changed during transformation/normalization.")
    new_columns = [c for c in result.columns if c not in data.columns]
    if result[new_columns].isna().any().any():
        raise ValueError("Transformed columns still contain missing values.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)

    print("DATA TRANSFORMATION & NORMALIZATION COMPLETED")
    print(f"Rows: {len(result):,} | new columns: {len(new_columns)}\n")
    print(report[["feature", "transformation", "normalization", "imputed_rows",
                  "skew_before", "skew_after", "min_after", "max_after"]].to_string(index=False))
    return result, report


def main():
    parser = argparse.ArgumentParser(description="Phase I - data transformation & normalization")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run(args.input, args.output, args.report)


if __name__ == "__main__":
    main()
