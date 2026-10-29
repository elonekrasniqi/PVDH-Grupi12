"""
Feature relationship analysis on the transformed & normalized dataset (6D).

Every 6D column gets a row in the report:
  - identifiers and raw columns that were replaced by a transformed version are
    listed with the reason they are not candidates;
  - every candidate feature is measured against the target is_fraud:
      numeric / binary  -> point-biserial correlation
      categorical       -> Cramer's V
      all               -> mutual information (also catches non-linear links)
  - candidates are checked for possible target leakage (a value/category with
    >= 100 rows that is always or never fraud), constant values and high
    cardinality;
  - relevant candidates are compared with each other (Spearman for numeric /
    binary, Cramer's V for categorical); of a pair with |relationship| >= 0.85
    the feature with the lower mutual information is marked redundant;
  - complete one-hot groups (all levels kept) drop their weakest level so the
    columns do not always sum to 1 (dummy-variable trap).

Outputs: 6F_FEATURE_RELATIONSHIP_REPORT.csv (one row per column) and
6F_FEATURE_CORRELATION_PAIRS.csv (candidate pairs with |relationship| >= 0.5).
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "phase_1"
DEFAULT_INPUT = DATA / "6D_TRANSFORMED_NORMALIZED_DATASET.csv"
DEFAULT_REPORT = DATA / "6F_FEATURE_RELATIONSHIP_REPORT.csv"
DEFAULT_PAIRS = DATA / "6F_FEATURE_CORRELATION_PAIRS.csv"

TARGET = "is_fraud"
SEED = 42

# --- candidate features (already transformed / normalized in 6D) -----------------
NUMERIC_FEATURES = (
    "amount_usd_log_z", "customer_avg_amount_usd_log_z", "velocity_total_amount_log_z",
    "customer_txn_count_z", "velocity_num_transactions_z", "velocity_max_single_amount_z",
    "velocity_unique_merchants_z", "velocity_unique_countries_z",
    "customer_away_share_minmax", "customer_night_share_minmax", "customers_per_device_minmax",
    "transaction_amount_category_code_minmax", "transaction_frequency_category_code_minmax",
    "customer_activity_level_code_minmax", "customer_spending_level_code_minmax",
    "hour_sin", "hour_cos",
)
BINARY_FEATURES = (
    "distance_from_home", "card_present", "high_risk_merchant", "weekend_transaction",
    "currency_imputed", "amount_usd_invalid", "is_shared_device",
    "card_brand_visa", "card_brand_mastercard", "card_brand_amex", "card_brand_discover",
    "time_period_night", "time_period_morning", "time_period_afternoon", "time_period_evening",
)
# country before currency_resolved: on a tie the first one is kept (country is more granular)
CATEGORICAL_FEATURES = (
    "country", "currency_resolved", "city", "city_size", "merchant_category",
    "merchant_type", "merchant", "card_type", "device", "channel",
)
ONE_HOT_GROUPS = {
    "card_brand": ("card_brand_visa", "card_brand_mastercard", "card_brand_amex", "card_brand_discover"),
    "transaction_time_period": ("time_period_night", "time_period_morning",
                                "time_period_afternoon", "time_period_evening"),
}

# --- columns that are not candidates ------------------------------------------
IDENTIFIERS = ("transaction_id", "customer_id", "card_number", "device_fingerprint",
               "ip_address", "timestamp", "exchange_rate_date")
SUPERSEDED = {
    "amount": "amount_usd_log_z (raw amount in mixed currencies)",
    "currency": "currency_resolved",
    "exchange_rate": "country / currency_resolved (encodes the currency only)",
    "amount_usd": "amount_usd_log_z",
    "customer_avg_amount_usd": "customer_avg_amount_usd_log_z",
    "customer_txn_count": "customer_txn_count_z",
    "customer_away_share": "customer_away_share_minmax",
    "customer_night_share": "customer_night_share_minmax",
    "customers_per_device": "customers_per_device_minmax",
    "transaction_hour": "hour_sin / hour_cos",
    "velocity_last_hour": "velocity_* columns",
    "velocity_num_transactions": "velocity_num_transactions_z",
    "velocity_total_amount": "velocity_total_amount_log_z",
    "velocity_unique_merchants": "velocity_unique_merchants_z",
    "velocity_unique_countries": "velocity_unique_countries_z",
    "velocity_max_single_amount": "velocity_max_single_amount_z",
    "card_brand": "card_brand_* one-hot columns",
    "transaction_time_period": "time_period_* one-hot columns",
    **{f"{c}": f"{c}_code_minmax" for c in ("transaction_amount_category", "transaction_frequency_category",
                                             "customer_activity_level", "customer_spending_level")},
    **{f"{c}_code": f"{c}_code_minmax" for c in ("transaction_amount_category", "transaction_frequency_category",
                                                  "customer_activity_level", "customer_spending_level")},
}

LEAKAGE_MIN_ROWS = 100
NUMERIC_ASSOCIATION_THRESHOLD = 0.05      # |point-biserial|
CATEGORICAL_ASSOCIATION_THRESHOLD = 0.04  # Cramer's V
MI_THRESHOLD = 0.01                       # mutual information (nats)
REDUNDANCY_THRESHOLD = 0.85
PAIR_REPORT_THRESHOLD = 0.5
MAX_CATEGORICAL_LEVELS = 50

REPORT_COLUMNS = ("feature", "feature_type", "metric", "association", "mutual_information",
                  "valid_rows", "levels", "top_categories", "potential_leakage",
                  "redundant_with", "recommendation", "notes")


# --- measures -----------------------------------------------------------------

def clean_category(column: pd.Series) -> pd.Series:
    """Same spelling for the same value ('GOLD CREDIT' == 'Gold Credit')."""
    return column.astype(str).str.strip().str.casefold()


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    total = table.sum()
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / total
    chi_squared = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi_squared / (total * (min(table.shape) - 1))))


def point_biserial(values: pd.Series, target: pd.Series):
    if values.nunique() < 2:
        return None
    return float(np.corrcoef(values, target)[0, 1])


def leakage_levels(values: pd.Series, target: pd.Series) -> str:
    groups = target.groupby(values).agg(["size", "mean"])
    leaking = groups[(groups["size"] >= LEAKAGE_MIN_ROWS) & groups["mean"].isin([0.0, 1.0])]
    return "; ".join(str(level) for level in sorted(leaking.index, key=str))


def top_categories(values: pd.Series, target: pd.Series, count=3) -> str:
    groups = target.groupby(values).agg(["size", "mean"]).sort_values("size", ascending=False)
    return "; ".join(f"{level}:fraud_rate={row['mean']:.3f},n={int(row['size'])}"
                     for level, row in groups.head(count).iterrows())


def mutual_information(data: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    columns = [*NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES]
    matrix = pd.DataFrame({
        column: (data[column].astype("category").cat.codes if column in CATEGORICAL_FEATURES
                 else data[column].astype(float))
        for column in columns
    })
    discrete = [column not in NUMERIC_FEATURES for column in columns]
    scores = mutual_info_classif(matrix, target, discrete_features=discrete, random_state=SEED)
    return dict(zip(columns, scores))


def pair_relationship(data: pd.DataFrame, a: str, b: str):
    """Spearman for numeric/binary pairs, Cramer's V for categorical pairs, None for mixed."""
    a_categorical, b_categorical = a in CATEGORICAL_FEATURES, b in CATEGORICAL_FEATURES
    if a_categorical and b_categorical:
        return "cramers_v", cramers_v(data[a], data[b])
    if not a_categorical and not b_categorical:
        value = data[a].corr(data[b], method="spearman")
        return "spearman", (0.0 if pd.isna(value) else float(value))
    return None, None


# --- analysis -----------------------------------------------------------------

def load(input_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path, low_memory=False)
    required = {TARGET, *NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    target = pd.to_numeric(data[TARGET], errors="coerce")
    if not target.isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be 0/1 (run the transformation step first).")
    data[TARGET] = target.astype(int)
    for column in CATEGORICAL_FEATURES:
        data[column] = clean_category(data[column])
    for column in (*NUMERIC_FEATURES, *BINARY_FEATURES):
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
        if data[column].isna().any():
            raise ValueError(f"{column} has missing values.")
    return data


def measure_candidates(data: pd.DataFrame) -> dict[str, dict]:
    target = data[TARGET]
    mi = mutual_information(data, target)
    rows = {}
    for column in (*NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES):
        values = data[column]
        categorical = column in CATEGORICAL_FEATURES
        kind = "categorical" if categorical else ("binary" if column in BINARY_FEATURES else "numeric")
        association = cramers_v(values, target) if categorical else point_biserial(values, target)
        rows[column] = {
            "feature": column, "feature_type": kind,
            "metric": "cramers_v" if categorical else "point_biserial",
            "association": association, "mutual_information": float(mi[column]),
            "valid_rows": int(values.notna().sum()), "levels": int(values.nunique()),
            "top_categories": top_categories(values, target) if kind != "numeric" else "",
            "potential_leakage": leakage_levels(values, target) if kind != "numeric" else "",
            "redundant_with": "", "notes": "",
        }
    return rows


def relevance(row: dict) -> str:
    """First recommendation pass: everything except redundancy."""
    if row["potential_leakage"]:
        return "exclude_possible_target_leakage"
    if row["levels"] < 2:
        return "exclude_constant"
    if row["feature_type"] == "categorical" and row["levels"] > MAX_CATEGORICAL_LEVELS:
        return "exclude_high_cardinality"
    threshold = (CATEGORICAL_ASSOCIATION_THRESHOLD if row["feature_type"] == "categorical"
                 else NUMERIC_ASSOCIATION_THRESHOLD)
    strong = row["association"] is not None and abs(row["association"]) >= threshold
    if strong or row["mutual_information"] >= MI_THRESHOLD:
        return "retain"
    return "exclude_low_association"


def mark_redundancy(data: pd.DataFrame, rows: dict[str, dict]) -> list[dict]:
    order = list(rows)       # stable order for ties
    relevant = sorted((c for c in order if rows[c]["recommendation"] == "retain"),
                      key=lambda c: (-rows[c]["mutual_information"], order.index(c)))
    pairs, kept = [], []
    for a, b in combinations(order, 2):
        if rows[a]["recommendation"] in ("exclude_constant",) or rows[b]["recommendation"] in ("exclude_constant",):
            continue
        metric, value = pair_relationship(data, a, b)
        if metric and abs(value) >= PAIR_REPORT_THRESHOLD:
            pairs.append({"feature_a": a, "feature_b": b, "metric": metric,
                          "relationship": round(value, 6), "redundant": abs(value) >= REDUNDANCY_THRESHOLD,
                          "dropped": ""})
    strong = {(p["feature_a"], p["feature_b"]): p for p in pairs if p["redundant"]}
    for column in relevant:
        partner = next((k for k in kept if (k, column) in strong or (column, k) in strong), None)
        if partner is None:
            kept.append(column)
            continue
        pair = strong.get((partner, column)) or strong[(column, partner)]
        pair["dropped"] = column
        rows[column]["recommendation"] = "exclude_redundant"
        rows[column]["redundant_with"] = partner
        rows[column]["notes"] = (f"{pair['metric']}={pair['relationship']:.3f} with {partner}, "
                                 f"which has higher mutual information.")
    for pair in strong.values():
        if not pair["dropped"]:
            pair["dropped"] = "-"    # at least one of the two was already excluded for another reason
    return pairs


def mark_dummy_trap(rows: dict[str, dict]) -> None:
    for group, columns in ONE_HOT_GROUPS.items():
        if all(rows[c]["recommendation"] == "retain" for c in columns):
            weakest = min(columns, key=lambda c: rows[c]["mutual_information"])
            rows[weakest]["recommendation"] = "exclude_dummy_trap"
            rows[weakest]["notes"] = (f"All {group} levels were kept; the weakest level is dropped "
                                      "because the one-hot columns always sum to 1.")


def non_candidate_rows(columns) -> list[dict]:
    rows = []
    for column in columns:
        if column in IDENTIFIERS:
            recommendation, notes = "exclude_identifier", "Identifier / timestamp, not a predictive feature."
        elif column in SUPERSEDED:
            recommendation, notes = "exclude_superseded", f"Replaced by {SUPERSEDED[column]}."
        else:
            recommendation, notes = "exclude_not_analyzed", "Not listed as a candidate feature."
        rows.append({"feature": column, "feature_type": "", "metric": "", "association": "",
                     "mutual_information": "", "valid_rows": "", "levels": "", "top_categories": "",
                     "potential_leakage": "", "redundant_with": "",
                     "recommendation": recommendation, "notes": notes})
    return rows


def analyze(input_path=DEFAULT_INPUT, report_path=DEFAULT_REPORT, pairs_path=DEFAULT_PAIRS):
    data = load(Path(input_path))
    rows = measure_candidates(data)
    for row in rows.values():
        row["recommendation"] = relevance(row)
    pairs = mark_redundancy(data, rows)
    mark_dummy_trap(rows)

    for row in rows.values():
        if row["association"] is not None:
            row["association"] = round(row["association"], 6)
        row["mutual_information"] = round(row["mutual_information"], 6)
        if not row["notes"] and row["recommendation"] == "exclude_possible_target_leakage":
            row["notes"] = "These values/categories are always or never fraud (>= 100 rows)."
    others = [c for c in data.columns if c not in rows and c != TARGET]
    report = pd.DataFrame([*rows.values(), *non_candidate_rows(others)], columns=REPORT_COLUMNS)
    pairs = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "metric", "relationship", "redundant", "dropped"])
    pairs = pairs.reindex(pairs["relationship"].abs().sort_values(ascending=False).index)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    return len(data), report, pairs


def main():
    parser = argparse.ArgumentParser(description="Phase I - feature relationship analysis (on 6D)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    args = parser.parse_args()
    rows, report, pairs = analyze(args.input, args.report, args.pairs)

    candidates = report[report["metric"] != ""].copy()
    candidates = candidates.sort_values("mutual_information", ascending=False)
    print("FEATURE RELATIONSHIP ANALYSIS COMPLETED")
    print(f"Rows analyzed: {rows:,} | candidates: {len(candidates)} | "
          f"retained: {(report['recommendation'] == 'retain').sum()}\n")
    print(candidates[["feature", "metric", "association", "mutual_information", "recommendation"]]
          .to_string(index=False))
    print("\nRedundant pairs (|relationship| >= 0.85):")
    print(pairs[pairs["redundant"]].to_string(index=False))


if __name__ == "__main__":
    main()
