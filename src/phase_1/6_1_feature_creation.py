"""
Feature creation: adds 6 new features to the stratified sample (5B).

Classic feature creation techniques. The features describe the transaction and
the customer; they do not score how unusual a value is (outliers are handled in
Phase 2). is_fraud is never used to build a feature.

Construction by aggregation (customer profile, from the full dataset 4F):
  customer_txn_count        - number of transactions of the customer
  customer_avg_amount_usd   - the customer's average amount_usd (amounts <= 0, e.g. -999, ignored)
  customer_away_share       - share of the customer's transactions made far from home (distance_from_home = 1)
  customer_night_share      - share of the customer's transactions made between 00:00 and 05:59

Construction by aggregation (device, from the full dataset 4F):
  customers_per_device      - number of different customers who used this device_fingerprint

Extraction:
  card_brand                - card network from the first digit of card_number
                              (3 = amex, 4 = visa, 5 = mastercard, 6 = discover)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "phase_1"
DEFAULT_INPUT = DATA / "5B_STRATIFIED_SAMPLE.csv"
DEFAULT_HISTORY = DATA / "4A_INTEGRATED_DATASET.csv"
DEFAULT_OUTPUT = DATA / "6A_FEATURE_CREATION_DATASET.csv"
DEFAULT_REPORT = DATA / "6B_FEATURE_CREATION_REPORT.csv"

TARGET = "is_fraud"
KEY = ["transaction_id", "customer_id", "timestamp"]
CARD_BRANDS = {"3": "amex", "4": "visa", "5": "mastercard", "6": "discover"}
NIGHT_HOURS = range(0, 6)
FEW_VALUES = 10          # features with at most this many values get a fraud rate per value
LEAKAGE_MIN_ROWS = 100   # same rule as the feature relationship analysis

NEW_FEATURES = {
    "customer_txn_count": "Number of transactions of the customer",
    "customer_avg_amount_usd": "Average amount_usd of the customer",
    "customer_away_share": "Share of the customer's transactions made far from home",
    "customer_night_share": "Share of the customer's transactions made between 00:00 and 05:59",
    "customers_per_device": "Different customers who used this device_fingerprint",
    "card_brand": "Card network from the first digit of card_number",
}
CATEGORICAL = {"card_brand"}
HISTORY_COLUMNS = KEY + ["amount_usd", "distance_from_home", "transaction_hour", "device_fingerprint"]


def read_csv(path: Path, columns=None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)


def clean(column: pd.Series) -> pd.Series:
    return column.astype(str).str.strip().str.upper()


# --- extraction ---------------------------------------------------------------

def card_brand(card_number: pd.Series) -> pd.Series:
    first_digit = card_number.astype(str).str.replace(r"\D", "", regex=True).str[:1]
    return first_digit.map(CARD_BRANDS).fillna("other")


# --- construction: customer profile --------------------------------------------

def customer_profile(history: pd.DataFrame) -> pd.DataFrame:
    history = history.drop_duplicates(KEY)
    customer = history["customer_id"]
    amount = pd.to_numeric(history["amount_usd"], errors="coerce")
    distance = pd.to_numeric(history["distance_from_home"], errors="coerce")
    hour = pd.to_numeric(history["transaction_hour"], errors="coerce")

    away = (distance == 1).astype(float).where(distance.notna())   # unknown distance: not counted
    night = hour.isin(NIGHT_HOURS).astype(float).where(hour.notna())
    return pd.DataFrame({
        "customer_txn_count": customer.value_counts(),
        "customer_avg_amount_usd": amount.where(amount > 0).groupby(customer).mean().round(2),
        "customer_away_share": away.groupby(customer).mean().round(4),
        "customer_night_share": night.groupby(customer).mean().round(4),
    })


# --- construction: device -------------------------------------------------------

def customers_per_device(history: pd.DataFrame) -> pd.Series:
    history = history.drop_duplicates(KEY)
    return history.groupby(history["device_fingerprint"].str.strip())["customer_id"].nunique()


def create_features(data: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    result = data.merge(customer_profile(history), left_on="customer_id", right_index=True, how="left")
    per_device = customers_per_device(history)
    result["customers_per_device"] = result["device_fingerprint"].str.strip().map(per_device).astype("Int64")
    result["card_brand"] = card_brand(result["card_number"])
    return result[list(data.columns) + list(NEW_FEATURES)]


def category_row(feature: str, description: str, values: pd.Series, fraud: pd.Series) -> dict:
    """For a category: fraud rate per value; AUC of using that rate as the score."""
    rate = fraud.groupby(values).mean()
    score = values.map(rate)
    auc = roc_auc_score(fraud, score) if values.nunique() > 1 and fraud.nunique() > 1 else 0.5
    return {
        "feature": feature,
        "description": description,
        "rows_with_value": int(values.notna().sum()),
        "mean_fraud": "", "mean_non_fraud": "",
        "auc": round(auc, 4),
        "detects_fraud": ("constant: drop" if values.nunique() <= 1 else
                          "strong" if auc >= 0.7 else "moderate" if auc >= 0.6 else "weak"),
    }


def rates_and_warning(values: pd.Series, fraud: pd.Series) -> tuple[str, str]:
    """Fraud rate per value (only for features with few values) and a leakage warning."""
    if values.nunique() > FEW_VALUES:
        return "", ""
    groups = fraud.groupby(values).agg(["mean", "size"])
    rates = "; ".join(f"{k:g}={r['mean']:.3f} (n={int(r['size'])})" if isinstance(k, float) else
                      f"{k}={r['mean']:.3f} (n={int(r['size'])})" for k, r in groups.iterrows())
    leaking = groups[(groups["size"] >= LEAKAGE_MIN_ROWS) & groups["mean"].isin([0.0, 1.0])]
    warning = "possible target leakage: a group is always or never fraud" if len(leaking) else ""
    return rates, warning


def build_report(data: pd.DataFrame) -> pd.DataFrame:
    fraud = data[TARGET].str.strip().str.lower().isin({"true", "1"}).astype(int)
    rows = []
    for feature, description in NEW_FEATURES.items():
        if feature in CATEGORICAL:
            rows.append(category_row(feature, description, data[feature], fraud))
            continue
        values = pd.to_numeric(data[feature], errors="coerce").astype(float)
        valid = values.notna()
        varies = values[valid].nunique() > 1 and fraud[valid].nunique() > 1
        auc = roc_auc_score(fraud[valid], values[valid]) if varies else 0.5
        auc = max(auc, 1 - auc)  # direction does not matter for detection
        rows.append({
            "feature": feature,
            "description": description,
            "rows_with_value": int(valid.sum()),
            "mean_fraud": round(values[valid & (fraud == 1)].mean(), 4),
            "mean_non_fraud": round(values[valid & (fraud == 0)].mean(), 4),
            "auc": round(auc, 4),
            "detects_fraud": ("constant: drop" if not varies else
                              "strong" if auc >= 0.7 else "moderate" if auc >= 0.6 else "weak"),
        })
    report = pd.DataFrame(rows)
    rates, warnings = zip(*(rates_and_warning(data[f] if f in CATEGORICAL else
                                              pd.to_numeric(data[f], errors="coerce"), fraud)
                            for f in NEW_FEATURES))
    report["fraud_rate_by_value"], report["warning"] = rates, warnings
    return report


def run(input_path=DEFAULT_INPUT, history_path=DEFAULT_HISTORY,
        output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT):
    data = read_csv(Path(input_path))
    missing = (set(HISTORY_COLUMNS) | {TARGET, "card_number"}) - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    history = read_csv(Path(history_path), HISTORY_COLUMNS)
    result = create_features(data, history)
    if len(result) != len(data):
        raise ValueError("Row count changed during feature creation.")

    report = build_report(result)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)

    print("FEATURE CREATION COMPLETED")
    print(f"Rows: {len(result):,} | New features: {len(NEW_FEATURES)}")
    print("AUC: 0.5 = guessing, 1.0 = perfect detection\n")
    print(report[["feature", "rows_with_value", "mean_fraud", "mean_non_fraud",
                  "auc", "detects_fraud", "warning"]].to_string(index=False))
    for _, row in report[report["fraud_rate_by_value"] != ""].iterrows():
        print(f"\n{row['feature']} - fraud rate per value: {row['fraud_rate_by_value']}")
    return result, report


def main():
    parser = argparse.ArgumentParser(description="Phase I - feature creation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run(args.input, args.history, args.output, args.report)


if __name__ == "__main__":
    main()