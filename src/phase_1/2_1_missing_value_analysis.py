from __future__ import annotations

import csv
from pathlib import Path

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "dataset"
    / "phase_1"
    / "1F_FORMAT_STANDARDIZED_DATASET.csv"
)
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}


def analyze_missing_values(path: Path) -> tuple[int, dict[str, int]]:
    missing_by_column: dict[str, int] = {}
    row_count = 0

    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, [])
        missing_by_column = dict.fromkeys(header, 0)

        for row in reader:
            row_count += 1
            for index, value in enumerate(row[:len(header)]):
                if (value or "").strip().lower() in NULL_TOKENS:
                    missing_by_column[header[index]] += 1

    return row_count, missing_by_column


def print_summary() -> None:
    row_count, missing_by_column = analyze_missing_values(DATASET_PATH)
    columns_with_missing = {
        name: count for name, count in missing_by_column.items() if count > 0
    }

    print(f"Dataset: {DATASET_PATH.name}")
    print(f"Rows scanned: {row_count}")
    print(f"Missing values: {sum(missing_by_column.values())}")

    if not columns_with_missing:
        print("No missing values detected.")
        return

    print("Columns with missing values:")
    for name, count in sorted(
        columns_with_missing.items(), key=lambda item: (-item[1], item[0])
    ):
        print(f"  {name}: {count}")

    print("Handling strategy used by missing_value_handling.py:")
    print("  - Exclude rows without a valid is_fraud target.")
    print("  - Impute amount with the median for its currency; use global median if unknown.")
    print("  - Impute distance_from_home and transaction_hour with their medians.")
    print("  - Represent missing currency, merchant_category, card_type, and channel as 'Unknown'.")
    print("  - Treat blank values and '?' as missing; preserve the source dataset.")


if __name__ == "__main__":
    print_summary()
