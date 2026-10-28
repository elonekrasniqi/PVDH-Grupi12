"""
Discretization (5 grouped columns) and binarization (0/1 columns) of the
feature creation output (6A).
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "phase_1"
DEFAULT_INPUT = DATA / "6A_FEATURE_CREATION_DATASET.csv"
DEFAULT_OUTPUT = DATA / "6B_DISCRETIZED_BINARIZED_DATASET.csv"
DEFAULT_REPORT = DATA / "6C_DISCRETIZATION_BINARIZATION_REPORT.csv"

TARGET = "is_fraud"
BOOLEAN_COLUMNS = ("is_fraud", "card_present", "high_risk_merchant", "weekend_transaction")
SHARED_DEVICE_MIN = 2
CARD_BRANDS = ("visa", "mastercard", "amex", "discover")
LEAKAGE_MIN_ROWS = 100  # same rule as the feature relationship analysis
UNKNOWN = "Unknown"
LEVELS = ["Low", "Medium", "High"]

# new column: source column (equal-frequency tertiles: Low / Medium / High)
EQUAL_FREQUENCY = {
    "transaction_amount_category": "amount_usd",
    "transaction_frequency_category": "velocity_txn_count",   # read from velocity_last_hour
    "customer_activity_level": "customer_txn_count",
    "customer_spending_level": "customer_avg_amount_usd",
}
TIME_PERIODS = ([-0.5, 5.5, 11.5, 17.5, 23.5], ["Night", "Morning", "Afternoon", "Evening"])
BINARY_FLAGS = ["is_shared_device", *[f"card_brand_{b}" for b in CARD_BRANDS]]


def to_number(column: pd.Series) -> pd.Series:
    return pd.to_numeric(column, errors="coerce")


# --- discretization -----------------------------------------------------------

def velocity_txn_count(text: str) -> float:
    """num_transactions from the velocity_last_hour text; works with 'single' and "double" quotes."""
    try:
        return float(ast.literal_eval(text.strip())["num_transactions"])
    except (ValueError, SyntaxError, KeyError, TypeError, AttributeError):
        return float("nan")


def equal_frequency(values: pd.Series, labels: list[str]):
    """Splits values into len(labels) groups with (about) the same number of rows."""
    valid = values.where(values > 0)            # 0 / negative amounts are not real values
    groups, edges = pd.qcut(valid, len(labels), labels=labels, retbins=True, duplicates="drop")
    return groups.cat.add_categories(UNKNOWN).fillna(UNKNOWN), list(edges)


def discretize(data: pd.DataFrame):
    result = data.copy()
    sources = {"velocity_txn_count": result["velocity_last_hour"].map(velocity_txn_count)}
    limits = {}
    for column, source in EQUAL_FREQUENCY.items():
        values = sources[source] if source in sources else to_number(result[source])
        result[column], limits[column] = equal_frequency(values, LEVELS)

    edges, labels = TIME_PERIODS
    hour = pd.cut(to_number(result["transaction_hour"]), bins=edges, labels=labels)
    result["transaction_time_period"] = hour.cat.add_categories(UNKNOWN).fillna(UNKNOWN)
    limits["transaction_time_period"] = edges
    return result, limits


# --- binarization -------------------------------------------------------------

def binarize_boolean(column: pd.Series) -> pd.Series:
    values = column.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    if values.isna().any():
        raise ValueError(f"{column.name} has values other than True/False")
    return values.astype(int)


def binarize(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in BOOLEAN_COLUMNS:
        result[column] = binarize_boolean(result[column])

    result["is_shared_device"] = (to_number(result["customers_per_device"]) >= SHARED_DEVICE_MIN).astype(int)

    brand = result["card_brand"].astype(str).str.strip().str.lower()
    for name in CARD_BRANDS:
        result[f"card_brand_{name}"] = (brand == name).astype(int)
    return result


# --- report -------------------------------------------------------------------

def leakage_warning(rows: int, fraud_rate) -> str:
    if fraud_rate != "" and rows >= LEAKAGE_MIN_ROWS and fraud_rate in (0.0, 1.0):
        return "possible target leakage: group is always or never fraud"
    return ""


def group_rows(data: pd.DataFrame, column: str, method: str, edges=None) -> list[dict]:
    groups = data[column]
    categories = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(groups.unique())
    rows = []
    for i, group in enumerate(categories):
        in_group = groups == group
        if group == UNKNOWN and not in_group.any():
            continue                                    # only list "unknown" when it occurs
        has_limits = edges is not None and group != UNKNOWN and i < len(edges) - 1
        rows.append({
            "feature": column,
            "method": method,
            "group": group,
            "lower": round(edges[i], 2) if has_limits else "",
            "upper": round(edges[i + 1], 2) if has_limits else "",
            "rows": int(in_group.sum()),
            "fraud_rate": round(data.loc[in_group, TARGET].mean(), 4) if in_group.any() else "",
        })
    return rows


def build_report(data: pd.DataFrame, limits: dict) -> pd.DataFrame:
    rows = []
    for column in EQUAL_FREQUENCY:
        rows += group_rows(data, column, "discretization: equal-frequency (tertiles)", limits[column])
    rows += group_rows(data, "transaction_time_period", "discretization: fixed ranges",
                       limits["transaction_time_period"])
    for column in BINARY_FLAGS:
        rows += group_rows(data, column, "binarization")
    report = pd.DataFrame(rows)
    report["warning"] = [leakage_warning(r, f) for r, f in zip(report["rows"], report["fraud_rate"])]
    return report


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT):
    data = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    required = {*BOOLEAN_COLUMNS, "transaction_hour", "card_brand", "customers_per_device",
                "velocity_last_hour", "amount_usd", "customer_txn_count", "customer_avg_amount_usd"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    result, limits = discretize(data)
    result = binarize(result)
    if len(result) != len(data):
        raise ValueError("Row count changed during discretization/binarization.")
    report = build_report(result, limits)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)

    print("DISCRETIZATION & BINARIZATION COMPLETED")
    print(f"Rows: {len(result):,}\n")
    print(report[["feature", "group", "lower", "upper", "rows", "fraud_rate", "warning"]].to_string(index=False))
    return result, report


def main():
    parser = argparse.ArgumentParser(description="Phase I - discretization & binarization")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run(args.input, args.output, args.report)


if __name__ == "__main__":
    main()