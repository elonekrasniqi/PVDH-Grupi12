from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "dataset" / "phase_1" / "1F_FORMAT_STANDARDIZED_DATASET.csv"
DEFAULT_OUTPUT = ROOT / "dataset" / "phase_1" / "2B_MISSING_VALUES_HANDLED_DATASET.csv"
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}
NUMERIC_COLUMNS = ("amount", "distance_from_home", "transaction_hour")
CATEGORICAL_COLUMNS = ("currency", "merchant_category", "card_type", "channel")
TARGET_COLUMN = "is_fraud"


def is_missing(value: str | None) -> bool:
    return (value or "").strip().casefold() in NULL_TOKENS


def parse_finite(value: str | None) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def currency_group(value: str | None) -> str:
    if is_missing(value):
        return "unknown"
    return (value or "").strip().casefold()


def target_is_valid(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"true", "false", "0", "1"}


def read_profile(
    input_path: Path,
) -> tuple[list[str], dict[str, int], dict[str, list[float]], dict[str, list[float]], int, int]:
    missing_counts: dict[str, int] = {}
    amounts_by_currency: dict[str, list[float]] = defaultdict(list)
    numeric_values = {column: [] for column in NUMERIC_COLUMNS[1:]}
    rows_read = 0
    rows_without_target = 0

    with input_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if columns is None:
            raise ValueError(f"Input dataset has no header: {input_path}")
        required = {
            TARGET_COLUMN,
            *NUMERIC_COLUMNS,
            *CATEGORICAL_COLUMNS,
        }
        missing_columns = required.difference(columns)
        if missing_columns:
            raise ValueError(
                f"Input is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        missing_counts = dict.fromkeys(columns, 0)
        for row in reader:
            rows_read += 1
            for column in columns:
                if is_missing(row.get(column)):
                    missing_counts[column] += 1

            label = row.get(TARGET_COLUMN)
            if is_missing(label):
                rows_without_target += 1
                continue
            if not target_is_valid(label):
                raise ValueError(
                    f"Unexpected {TARGET_COLUMN} value on input row {rows_read + 1}: {label!r}"
                )

            currency = currency_group(row.get("currency"))
            amount = parse_finite(row.get("amount"))
            if amount is not None:
                amounts_by_currency[currency].append(amount)
            for column in NUMERIC_COLUMNS[1:]:
                value = parse_finite(row.get(column))
                if value is not None:
                    numeric_values[column].append(value)

    return (
        columns,
        missing_counts,
        amounts_by_currency,
        numeric_values,
        rows_read,
        rows_without_target,
    )


def merge_amount_values(
    amounts_by_currency: dict[str, list[float]],
) -> Iterable[float]:
    for values in amounts_by_currency.values():
        yield from values


def handle_missing_values(
    input_path: Path, output_path: Path
) -> tuple[dict[str, int], int, int, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must not overwrite the source dataset.")

    (
        columns,
        missing_counts,
        amounts_by_currency,
        numeric_values,
        rows_read,
        rows_without_target,
    ) = read_profile(input_path)
    if rows_read == 0:
        raise ValueError(f"Input dataset is empty: {input_path}")
    if not any(amounts_by_currency.values()):
        raise ValueError("No valid amounts are available for median imputation.")
    medians_by_currency = {
        currency: statistics.median(values)
        for currency, values in amounts_by_currency.items()
    }
    global_amount_median = statistics.median(
        merge_amount_values(amounts_by_currency)
    )
    medians = {
        column: statistics.median(values)
        for column, values in numeric_values.items()
        if values
    }
    missing_numeric_medians = set(numeric_values).difference(medians)
    if missing_numeric_medians:
        raise ValueError(
            "No valid values are available for median imputation in: "
            + ", ".join(sorted(missing_numeric_medians))
        )

    rows_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open("r", newline="", encoding="utf-8") as source,
        output_path.open("w", newline="", encoding="utf-8") as destination,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=2):
            if is_missing(row.get(TARGET_COLUMN)):
                continue

            currency = currency_group(row.get("currency"))
            for column in CATEGORICAL_COLUMNS:
                if is_missing(row.get(column)):
                    row[column] = "Unknown"

            amount = parse_finite(row.get("amount"))
            if amount is None:
                amount = medians_by_currency.get(currency, global_amount_median)
            row["amount"] = str(amount)

            for column, median in medians.items():
                value = parse_finite(row.get(column))
                if value is None:
                    row[column] = str(median)
                else:
                    row[column] = str(value)

            for column in NUMERIC_COLUMNS:
                if parse_finite(row.get(column)) is None:
                    raise ValueError(
                        f"Unable to impute {column} on input row {row_number}."
                    )

            writer.writerow(row)
            rows_written += 1

    return missing_counts, rows_read, rows_written, rows_without_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Handle missing values in the standardized dataset and write a "
            "separate phase 1 intermediate CSV."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    (
        missing_counts,
        rows_read,
        rows_written,
        rows_without_target,
    ) = handle_missing_values(args.input, args.output)
    print(f"Rows read: {rows_read:,}")
    print(f"Missing values identified: {sum(missing_counts.values()):,}")
    for column, count in sorted(
        missing_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        if count:
            print(f"  {column}: {count:,}")
    print(f"Rows written: {rows_written:,}")
    print(f"Rows excluded for missing target: {rows_without_target:,}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("Strategy: amount median by currency (global median fallback);")
    print("  distance_from_home and transaction_hour medians;")
    print("  categorical missing values represented as 'Unknown'.")
    print("Rows with missing is_fraud are excluded; identifiers are not imputed.")


if __name__ == "__main__":
    main()
