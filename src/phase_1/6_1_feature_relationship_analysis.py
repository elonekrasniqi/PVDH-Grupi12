from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "dataset" / "phase_1" / "5B_STRATIFIED_SAMPLE.csv"
DEFAULT_OUTPUT = ROOT / "dataset" / "phase_1" / "6A_FEATURE_RELATIONSHIP_REPORT.csv"

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}
TARGET_COLUMN = "is_fraud"
NUMERIC_FEATURES = (
    "amount_usd",
    "distance_from_home",
    "transaction_hour",
    "exchange_rate",
)
CATEGORICAL_FEATURES = (
    "country",
    "currency_resolved",
    "merchant_category",
    "merchant_type",
    "merchant",
    "city",
    "city_size",
    "card_type",
    "card_present",
    "device",
    "channel",
    "high_risk_merchant",
    "weekend_transaction",
    "currency_imputed",
)
POTENTIAL_LEAKAGE_MIN_COUNT = 100
CATEGORICAL_ASSOCIATION_THRESHOLD = 0.04
NUMERIC_ASSOCIATION_THRESHOLD = 0.05
MAX_CATEGORICAL_LEVELS = 50
REDUNDANT_FEATURES = {
    "currency_resolved": "Overlaps strongly with country; country is more granular.",
    "exchange_rate": "Primarily encodes country/currency, already represented by country.",
}
REPORT_COLUMNS = (
    "feature",
    "feature_type",
    "metric",
    "association",
    "valid_rows",
    "levels",
    "top_categories",
    "potential_leakage",
    "recommendation",
    "notes",
)


def parse_target(value: str | None) -> int | None:
    normalized = (value or "").strip().casefold()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    return None


def parse_number(value: str | None) -> float | None:
    if value is None or value.strip().casefold() in NULL_TOKENS:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def clean_category(value: str | None) -> str | None:
    normalized = (value or "").strip()
    if normalized.casefold() in NULL_TOKENS:
        return None
    normalized = normalized.casefold()
    return {"yes": "true", "no": "false"}.get(normalized, normalized)


def cramers_v(categories: dict[str, list[int]]) -> float:
    total = sum(sum(counts) for counts in categories.values())
    levels = len(categories)
    if total == 0 or levels < 2:
        return 0.0

    target_totals = [
        sum(counts[target] for counts in categories.values())
        for target in (0, 1)
    ]
    chi_squared = 0.0
    for counts in categories.values():
        category_total = sum(counts)
        for target in (0, 1):
            expected = category_total * target_totals[target] / total
            if expected:
                chi_squared += (counts[target] - expected) ** 2 / expected

    return math.sqrt(chi_squared / (total * min(levels - 1, 1)))


def point_biserial(
    non_fraud: tuple[int, float, float],
    fraud: tuple[int, float, float],
) -> float | None:
    count_0, sum_0, square_sum_0 = non_fraud
    count_1, sum_1, square_sum_1 = fraud
    count = count_0 + count_1
    if count == 0 or count_0 == 0 or count_1 == 0:
        return None

    mean_x = (sum_0 + sum_1) / count
    mean_y = count_1 / count
    variance_x = (square_sum_0 + square_sum_1) / count - mean_x**2
    variance_y = mean_y * (1 - mean_y)
    covariance = sum_1 / count - mean_x * mean_y
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def build_feature_row(
    feature: str,
    feature_type: str,
    metric: str,
    association: float | None,
    valid_rows: int,
    levels: int,
    top_categories: str = "",
    leakage: str = "",
    notes: str = "",
) -> dict[str, str]:
    if leakage:
        recommendation = "exclude_possible_target_leakage"
    elif levels < 2:
        recommendation = "exclude_constant"
    elif feature in REDUNDANT_FEATURES:
        recommendation = "exclude_redundant"
        notes = REDUNDANT_FEATURES[feature]
    elif feature_type == "categorical" and levels > MAX_CATEGORICAL_LEVELS:
        recommendation = "exclude_high_cardinality"
        notes = (
            f"{levels} levels exceed the candidate limit of "
            f"{MAX_CATEGORICAL_LEVELS}; guard against sparse-category overfitting."
        )
    elif association is not None and abs(association) >= (
        NUMERIC_ASSOCIATION_THRESHOLD
        if feature_type == "numeric"
        else CATEGORICAL_ASSOCIATION_THRESHOLD
    ):
        recommendation = "retain_candidate"
    else:
        recommendation = "low_univariate_association"

    return {
        "feature": feature,
        "feature_type": feature_type,
        "metric": metric,
        "association": "" if association is None else f"{association:.6f}",
        "valid_rows": str(valid_rows),
        "levels": str(levels),
        "top_categories": top_categories,
        "potential_leakage": leakage,
        "recommendation": recommendation,
        "notes": notes,
    }


def analyze(input_path: Path, output_path: Path) -> tuple[int, list[dict[str, str]]]:
    numeric_stats = {
        feature: {0: [0, 0.0, 0.0], 1: [0, 0.0, 0.0]}
        for feature in NUMERIC_FEATURES
    }
    numeric_levels = {feature: set() for feature in NUMERIC_FEATURES}
    category_stats: dict[str, dict[str, list[int]]] = {
        feature: defaultdict(lambda: [0, 0])
        for feature in CATEGORICAL_FEATURES
    }
    target_counts: Counter[int] = Counter()
    rows_read = 0

    with input_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Input dataset has no header: {input_path}")
        required = {TARGET_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
        missing_columns = required.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Input dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            rows_read += 1
            target = parse_target(row.get(TARGET_COLUMN))
            if target is None:
                raise ValueError(
                    f"Missing or invalid {TARGET_COLUMN} at row {row_number}."
                )
            target_counts[target] += 1

            for feature, by_target in numeric_stats.items():
                value = parse_number(row.get(feature))
                if value is None:
                    continue
                numeric_levels[feature].add(value)
                stats = by_target[target]
                stats[0] += 1
                stats[1] += value
                stats[2] += value * value

            for feature, categories in category_stats.items():
                category = clean_category(row.get(feature))
                if category is None:
                    continue
                counts = categories[category]
                counts[target] += 1

    if rows_read == 0:
        raise ValueError(f"Input dataset is empty: {input_path}")

    results: list[dict[str, str]] = []
    for feature, by_target in numeric_stats.items():
        valid_rows = sum(stats[0] for stats in by_target.values())
        levels = len(numeric_levels[feature])
        association = point_biserial(tuple(by_target[0]), tuple(by_target[1]))
        results.append(
            build_feature_row(
                feature,
                "numeric",
                "point_biserial",
                association,
                valid_rows,
                levels,
                notes=(
                    "Absolute association threshold for candidate retention: "
                    f"{NUMERIC_ASSOCIATION_THRESHOLD:.2f}."
                ),
            )
        )

    for feature, categories in category_stats.items():
        valid_rows = sum(sum(counts) for counts in categories.values())
        levels = len(categories)
        association = cramers_v(categories)
        ranked = sorted(
            categories.items(),
            key=lambda item: (-sum(item[1]), item[0]),
        )
        top_categories = "; ".join(
            f"{category}:fraud_rate={counts[1] / sum(counts):.3f},n={sum(counts)}"
            for category, counts in ranked[:3]
            if sum(counts)
        )
        leakage_categories = [
            category
            for category, counts in categories.items()
            if sum(counts) >= POTENTIAL_LEAKAGE_MIN_COUNT
            and counts[1] in {0, sum(counts)}
        ]
        leakage = "; ".join(sorted(leakage_categories))
        notes = (
            "Univariate association is descriptive, not a causal or model-performance measure. "
            f"Categorical candidate threshold: {CATEGORICAL_ASSOCIATION_THRESHOLD:.2f}."
        )
        if levels < 2:
            notes = "Constant feature in this sample."
        results.append(
            build_feature_row(
                feature,
                "categorical",
                "cramers_v",
                association,
                valid_rows,
                levels,
                top_categories,
                leakage,
                notes,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    return rows_read, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze candidate feature relationships with the fraud target."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows_read, results = analyze(args.input, args.output)
    fraud_feature_rows = [
        row for row in results if row["potential_leakage"]
    ]
    print(f"Rows analyzed: {rows_read:,}")
    print(f"Fraud prevalence: output is stratified by fraud and country.")
    print(f"Feature report: {args.output}")
    print("Association ranking:")
    for row in sorted(
        results,
        key=lambda item: abs(float(item["association"] or 0.0)),
        reverse=True,
    ):
        association = row["association"] or "n/a"
        print(
            f"  {row['feature']}: {row['metric']}={association}; "
            f"{row['recommendation']}"
        )
    if fraud_feature_rows:
        print("Potential target leakage found:")
        for row in fraud_feature_rows:
            print(f"  {row['feature']}: {row['potential_leakage']}")


if __name__ == "__main__":
    main()
