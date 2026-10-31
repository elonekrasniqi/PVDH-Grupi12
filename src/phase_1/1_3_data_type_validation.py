"""Data type definition and validation for Phase I preprocessing.

Reads the original fraud dataset and generates a column-level report.
The input dataset is never modified.
"""

import csv
import math
import re

from collections import Counter
from datetime import datetime
from pathlib import Path


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "dataset" / "synthetic_fraud_data.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "phase_1"

OUTPUT_FILE = (
    OUTPUT_DIR / "1C_DATA_TYPE_REPORT.csv"
)


# ------------------------------------------------------------
# 2. EXPECTED DATA TYPES
# ------------------------------------------------------------

EXPECTED_TYPES = {
    "transaction_id": "identifier",
    "customer_id": "identifier",
    "card_number": "string",
    "timestamp": "datetime",
    "merchant_category": "category",
    "merchant_type": "category",
    "merchant": "string",
    "amount": "float",
    "currency": "category",
    "country": "category",
    "city": "category",
    "city_size": "category",
    "card_type": "category",
    "card_present": "boolean",
    "device": "category",
    "channel": "category",
    "device_fingerprint": "string",
    "ip_address": "string",
    "distance_from_home": "float",
    "high_risk_merchant": "boolean",
    "transaction_hour": "integer_0_23",
    "weekend_transaction": "boolean",
    "velocity_last_hour": "string",
    "is_fraud": "boolean",
}

ID_PREFIXES = {
    "transaction_id": "TX_",
    "customer_id": "CUST_",
}

BOOLEAN_VALUES = {
    "true", "false",
    "yes", "no",
    "1", "0",
}

STANDARD_BOOLEAN_VALUES = {
    "True", "False",
}

MISSING_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "na",
    "n/a",
    "?",
}

REPORT_COLUMNS = [
    "column",
    "expected_type",
    "total_rows",
    "non_missing",
    "missing_count",
    "valid_count",
    "invalid_count",
    "format_issue_count",
    "unclassified_count",
    "invalid_examples",
    "format_examples",
]


# ------------------------------------------------------------
# 3. MISSING VALUE VALIDATION
# ------------------------------------------------------------

def is_missing(value):
    """Check whether a value is missing."""

    if value is None:
        return True

    return str(value).strip().lower() in MISSING_VALUES


# ------------------------------------------------------------
# 4. NUMERIC VALIDATION
# ------------------------------------------------------------

def validate_float(value):
    """Accept only finite numeric values."""

    try:
        number = float(value)
        return math.isfinite(number)

    except (ValueError, TypeError, OverflowError):
        return False


def validate_integer_0_23(value):
    """Accept whole-number hours from 0 to 23.

    Examples:
        15   -> valid
        15.0 -> valid
        5.0  -> valid
        15.5 -> invalid
        24   -> invalid
    """

    try:
        number = float(value)

        return (
            math.isfinite(number)
            and number.is_integer()
            and 0 <= number <= 23
        )

    except (ValueError, TypeError, OverflowError):
        return False


# ------------------------------------------------------------
# 5. BOOLEAN VALIDATION
# ------------------------------------------------------------

def validate_boolean(value):
    """Accept interpretable boolean representations."""

    return str(value).strip().lower() in BOOLEAN_VALUES


# ------------------------------------------------------------
# 6. IDENTIFIER VALIDATION
# ------------------------------------------------------------

def validate_identifier(value, column):
    """Validate identifier type and detect formatting differences."""

    text = str(value).strip()
    prefix = ID_PREFIXES.get(column)

    if not text or prefix is None:
        return "invalid"

    # Standard format: CUST_12345 or TX_12345
    if re.fullmatch(
        rf"{re.escape(prefix)}[A-Za-z0-9_-]+",
        text
    ):
        return "valid"

    # Alternative prefix capitalization or separator
    normalized = text.upper().replace("-", "_")

    if re.fullmatch(
        rf"{re.escape(prefix)}[A-Za-z0-9_-]+",
        normalized
    ):
        return "format_issue"

    # Numeric-only identifiers
    if text.isdigit():
        return "format_issue"

    return "invalid"

# ------------------------------------------------------------
# 7. DATETIME VALIDATION
# ------------------------------------------------------------

DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%Y-%m-%d",
    "%d/%m/%Y",
]


def validate_datetime(value):
    """Recognize valid dates in standard and alternative formats.

    Returns:
        valid        - ISO-compatible timestamp
        format_issue - recognized alternative date format
        invalid      - unrecognized or impossible date
    """

    text = str(value).strip()

    try:
        datetime.fromisoformat(text)

        if re.match(r"^\d{4}-\d{2}-\d{2}", text):
            return "valid"

    except ValueError:
        pass

    for date_format in DATETIME_FORMATS:

        try:
            datetime.strptime(text, date_format)

            if date_format.startswith("%d/%m/%Y"):
                return "format_issue"

            return "valid"

        except ValueError:
            continue

    return "invalid"


# ------------------------------------------------------------
# 8. GENERAL TYPE VALIDATION
# ------------------------------------------------------------

def validate_value(column, value, expected_type):
    """Return valid, invalid, format_issue or unclassified."""

    if expected_type == "float":

        return (
            "valid"
            if validate_float(value)
            else "invalid"
        )

    if expected_type == "integer_0_23":

        if not validate_integer_0_23(value):
            return "invalid"

        number = float(value)

        if str(value).strip() != str(int(number)):
            return "format_issue"

        return "valid"

    if expected_type == "boolean":

        if not validate_boolean(value):
            return "invalid"

        if str(value).strip() not in STANDARD_BOOLEAN_VALUES:
            return "format_issue"

        return "valid"

    if expected_type == "datetime":

        return validate_datetime(value)

    if expected_type == "identifier":

        return validate_identifier(value, column)

    if expected_type in {"string", "category"}:

        return "valid"

    return "unclassified"


# ------------------------------------------------------------
# 9. CREATE EMPTY COLUMN STATISTICS
# ------------------------------------------------------------

def create_column_stats(column, expected_type):

    return {
        "column": column,
        "expected_type": expected_type,
        "total_rows": 0,
        "non_missing": 0,
        "missing_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "format_issue_count": 0,
        "unclassified_count": 0,
        "invalid_examples": Counter(),
        "format_examples": Counter(),
    }


# ------------------------------------------------------------
# 10. DATA TYPE PROFILING
# ------------------------------------------------------------

def profile_dataset(input_file):
    """Profile each column without loading the full CSV into memory."""

    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "Input CSV does not contain a header."
            )

        columns = reader.fieldnames

        stats = {
            column: create_column_stats(
                column,
                EXPECTED_TYPES.get(column, "unknown")
            )
            for column in columns
        }

        for row in reader:

            for column in columns:

                column_stats = stats[column]

                column_stats["total_rows"] += 1

                value = row.get(column)

                if is_missing(value):

                    column_stats["missing_count"] += 1
                    continue

                column_stats["non_missing"] += 1

                result = validate_value(
                    column,
                    value,
                    column_stats["expected_type"]
                )

                if result == "valid":

                    column_stats["valid_count"] += 1

                elif result == "invalid":

                    column_stats["invalid_count"] += 1

                    column_stats["invalid_examples"][
                        str(value)
                    ] += 1

                elif result == "format_issue":

                    column_stats["format_issue_count"] += 1

                    column_stats["format_examples"][
                        str(value)
                    ] += 1

                else:

                    column_stats["unclassified_count"] += 1

    return stats


# ------------------------------------------------------------
# 11. REPORT GENERATION
# ------------------------------------------------------------

def format_examples(counter, limit=3):
    """Return the most frequent example values."""

    return " | ".join(
        value
        for value, _ in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0])
        )[:limit]
    )


def write_report(stats, output_file):

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=REPORT_COLUMNS
        )

        writer.writeheader()

        for column_stats in stats.values():

            writer.writerow({
                "column": column_stats["column"],
                "expected_type": column_stats["expected_type"],
                "total_rows": column_stats["total_rows"],
                "non_missing": column_stats["non_missing"],
                "missing_count": column_stats["missing_count"],
                "valid_count": column_stats["valid_count"],
                "invalid_count": column_stats["invalid_count"],
                "format_issue_count": column_stats["format_issue_count"],
                "unclassified_count": column_stats["unclassified_count"],
                "invalid_examples": format_examples(
                    column_stats["invalid_examples"]
                ),
                "format_examples": format_examples(
                    column_stats["format_examples"]
                ),
            })


# ------------------------------------------------------------
# 12. MAIN
# ------------------------------------------------------------

def main():

    stats = profile_dataset(INPUT_FILE)

    write_report(
        stats,
        OUTPUT_FILE
    )

    total_columns = len(stats)

    total_rows = (
        next(iter(stats.values()))["total_rows"]
        if stats
        else 0
    )

    total_missing = sum(
        item["missing_count"]
        for item in stats.values()
    )

    total_invalid = sum(
        item["invalid_count"]
        for item in stats.values()
    )

    total_format_issues = sum(
        item["format_issue_count"]
        for item in stats.values()
    )

    total_unclassified = sum(
        item["unclassified_count"]
        for item in stats.values()
    )

    print("\nData type validation completed successfully.")

    print(f"Input dataset: {INPUT_FILE}")
    print(f"Report saved to: {OUTPUT_FILE}")

    print(f"Columns: {total_columns}")
    print(f"Rows: {total_rows}")

    print(f"Missing values: {total_missing}")
    print(f"Invalid values: {total_invalid}")
    print(f"Format issues: {total_format_issues}")
    print(f"Unclassified values: {total_unclassified}")

    print("\nColumn validation summary:")
    print("-" * 85)

    for column, item in stats.items():

        print(
            f"{column:<25} "
            f"Invalid: {item['invalid_count']:<8} "
            f"Format: {item['format_issue_count']:<8} "
            f"Missing: {item['missing_count']}"
        )


if __name__ == "__main__":
    main()