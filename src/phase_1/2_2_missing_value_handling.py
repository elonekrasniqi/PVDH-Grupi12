"""Missing-value handling
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "phase_1" / "1B_FORMAT_STANDARDIZED_DATASET.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "phase_1" / "2B_MISSING_VALUES_HANDLED_DATASET.csv"

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}
TARGET_COLUMN = "is_fraud"
AMOUNT_COLUMN = "amount"
HOUR_COLUMN = "transaction_hour"
MEDIAN_COLUMNS = ("distance_from_home", HOUR_COLUMN)
NUMERIC_COLUMNS = (AMOUNT_COLUMN, *MEDIAN_COLUMNS)
CATEGORICAL_COLUMNS = ("currency", "merchant_category", "card_type", "channel")
UNKNOWN = "Unknown"

# Same mapping as COUNTRY_CURRENCY in 4_1_dataset_integration.py.
COUNTRY_CURRENCY = {
    "AUSTRALIA": "AUD",
    "BRAZIL": "BRL",
    "CANADA": "CAD",
    "FRANCE": "EUR",
    "GERMANY": "EUR",
    "JAPAN": "JPY",
    "MEXICO": "MXN",
    "NIGERIA": "NGN",
    "RUSSIA": "RUB",
    "SINGAPORE": "SGD",
    "UK": "GBP",
    "USA": "USD",
}

# column -> (strategy, reason); used by this script and by the 2A report (2_1).
STRATEGIES = {
    TARGET_COLUMN: (
        "remove row",
        "The target cannot be imputed; a guessed label would bias the fraud rate.",
    ),
    AMOUNT_COLUMN: (
        "median of the same currency (currency inferred from country if missing); "
        "global median fallback; amount <= 0 (-999 placeholder) treated as missing",
        "Amounts are in local currency and right-skewed: a per-currency median is "
        "robust to extreme values and keeps the currency scale.",
    ),
    HOUR_COLUMN: (
        "hour from timestamp; median fallback",
        "The hour is fully determined by the timestamp, so it can be recovered exactly.",
    ),
    "distance_from_home": (
        "median",
        "Few values missing; the median keeps the column distribution almost unchanged.",
    ),
    **{
        column: (
            f"category '{UNKNOWN}'",
            "Nominal attribute: an explicit 'Unknown' level keeps the row and does not "
            "invent a category." + (" Currency is resolved from country in 4_1."
                                    if column == "currency" else ""),
        )
        for column in CATEGORICAL_COLUMNS
    },
}
NOT_IMPUTED = (
    "not imputed",
    "Identifier or column without a defined imputation rule; left empty.",
)


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


def is_placeholder_amount(value: str | None) -> bool:
    """A number that is not a valid amount (<= 0, e.g. the -999 placeholder)."""
    number = parse_finite(value)
    return number is not None and number <= 0


def parse_amount(value: str | None) -> float | None:
    number = parse_finite(value)
    return number if number is not None and number > 0 else None


def is_missing_value(column: str, value: str | None) -> bool:
    if column == AMOUNT_COLUMN:
        return parse_amount(value) is None
    return is_missing(value)


def resolve_currency(row: dict) -> str | None:
    """Currency of the row, inferred from country when currency is missing."""
    currency = (row.get("currency") or "").strip().upper()
    if not is_missing(currency) and currency != UNKNOWN.upper():
        return currency
    return COUNTRY_CURRENCY.get((row.get("country") or "").strip().upper())


def hour_from_timestamp(value: str | None) -> int | None:
    if is_missing(value):
        return None
    try:
        return datetime.fromisoformat(value.strip()).hour
    except ValueError:
        return None


def target_is_valid(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"true", "false", "0", "1"}


def format_number(value: float) -> str:
    value = round(float(value), 4)          # medians of two values can give 265.78999999999996
    return str(int(value)) if value.is_integer() else str(value)


def read_profile(input_path: Path):
    """First pass: missing counts and the values needed for the medians."""
    amounts_by_currency: dict[str | None, list[float]] = defaultdict(list)
    median_values = {column: [] for column in MEDIAN_COLUMNS}
    rows_read = 0
    rows_without_target = 0

    with input_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if columns is None:
            raise ValueError(f"Input dataset has no header: {input_path}")
        required = {TARGET_COLUMN, "country", "timestamp", *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS}
        missing_columns = required.difference(columns)
        if missing_columns:
            raise ValueError(
                f"Input is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            rows_read += 1
            label = row.get(TARGET_COLUMN)
            if is_missing(label):
                rows_without_target += 1
                continue
            if not target_is_valid(label):
                raise ValueError(
                    f"Unexpected {TARGET_COLUMN} value on input row {rows_read + 1}: {label!r}"
                )
            amount = parse_amount(row.get(AMOUNT_COLUMN))
            if amount is not None:
                amounts_by_currency[resolve_currency(row)].append(amount)
            for column in MEDIAN_COLUMNS:
                value = parse_finite(row.get(column))
                if value is not None:
                    median_values[column].append(value)

    return list(columns), amounts_by_currency, median_values, rows_read, rows_without_target


def handle_missing_values(input_path: Path, output_path: Path):
    """Returns (rows_read, rows_written, rows_without_target, imputed counts per column)."""
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Output path must not overwrite the source dataset.")

    columns, amounts_by_currency, median_values, rows_read, rows_without_target = (
        read_profile(Path(input_path))
    )
    if rows_read == 0:
        raise ValueError(f"Input dataset is empty: {input_path}")
    all_amounts = [a for values in amounts_by_currency.values() for a in values]
    if not all_amounts:
        raise ValueError("No valid amounts are available for median imputation.")
    medians_by_currency = {
        currency: statistics.median(values)
        for currency, values in amounts_by_currency.items()
        if currency is not None
    }
    global_amount_median = statistics.median(all_amounts)
    medians = {}
    for column, values in median_values.items():
        if not values:
            raise ValueError(f"No valid values are available for median imputation in: {column}")
        medians[column] = statistics.median(values)

    imputed = defaultdict(int)
    rows_written = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with (
        Path(input_path).open("r", newline="", encoding="utf-8") as source,
        Path(output_path).open("w", newline="", encoding="utf-8") as destination,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()

        for row in reader:
            if is_missing(row.get(TARGET_COLUMN)):
                continue

            if parse_amount(row.get(AMOUNT_COLUMN)) is None:
                currency = resolve_currency(row)
                row[AMOUNT_COLUMN] = format_number(
                    medians_by_currency.get(currency, global_amount_median)
                )
                imputed[AMOUNT_COLUMN] += 1

            if parse_finite(row.get(HOUR_COLUMN)) is None:
                hour = hour_from_timestamp(row.get("timestamp"))
                row[HOUR_COLUMN] = str(hour if hour is not None else round(medians[HOUR_COLUMN]))
                imputed[HOUR_COLUMN] += 1

            for column in MEDIAN_COLUMNS:
                if column != HOUR_COLUMN and parse_finite(row.get(column)) is None:
                    row[column] = format_number(medians[column])
                    imputed[column] += 1

            for column in CATEGORICAL_COLUMNS:
                if is_missing(row.get(column)):
                    row[column] = UNKNOWN
                    imputed[column] += 1

            writer.writerow(row)
            rows_written += 1

    return rows_read, rows_written, rows_without_target, dict(imputed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Handle missing values in the standardized dataset (writes 2B)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows_read, rows_written, rows_without_target, imputed = handle_missing_values(
        args.input, args.output
    )
    print("MISSING VALUE HANDLING COMPLETED")
    print(f"Rows read: {rows_read:,}")
    print(f"Rows removed (missing {TARGET_COLUMN}): {rows_without_target:,}")
    print(f"Rows written: {rows_written:,}")
    print("Values imputed per column:")
    for column, count in sorted(imputed.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {column}: {count:,}  -> {STRATEGIES[column][0]}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()