import csv
import re

from collections import Counter
from datetime import datetime
from pathlib import Path

# 1. FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "dataset" / "synthetic_fraud_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "phase_1"
OUTPUT_FILE = OUTPUT_DIR / "1A_FORMAT_PROFILING_REPORT.csv"

# 2. COLUMN CONFIGURATION
# Boolean columns: values must be True or False.
BOOLEAN_COLUMNS = {
    "card_present",
    "high_risk_merchant",
    "weekend_transaction",
    "is_fraud",
}

# Values must be uppercase.
UPPERCASE_COLUMNS = {
    "currency",
}

# Values must be lowercase.
LOWERCASE_COLUMNS = {
    "merchant_type",
}

# First character uppercase; remaining characters lowercase.
CAPITALIZE_COLUMNS = {
    "merchant_category",
}

# Expected prefixes for identifiers.
ID_PREFIXES = {
    "customer_id": "CUST_",
    "transaction_id": "TX_",
}

# Columns checked only for leading/trailing whitespace.
WHITESPACE_ONLY_COLUMNS = {
    "card_number",
    "merchant",
    "country",
    "city",
    "city_size",
    "card_type",
    "device",
    "channel",
    "device_fingerprint",
    "ip_address",
    "distance_from_home",
    "transaction_hour",
    "velocity_last_hour",
}
AMOUNT_COLUMN = "amount"
TIMESTAMP_COLUMN = "timestamp"
ALLOWED_CARD_NUMBER_LENGTHS = {15, 16}


# Timestamp format:
# YYYY-MM-DD HH:MM:SS[.fraction]+HH:MM
# Example: 2024-09-30 15:18:24.452982+00:00
TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?"
    r"[+-]\d{2}:\d{2}$"
)
BOOLEAN_VALUES = {"true", "false"}

# 3. TIMESTAMP VALIDATION

def validate_timestamp(value):
    """Validate timestamp format and date/time values."""
    if not TIMESTAMP_PATTERN.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False

# 4. FORMAT CHECKING

def check_format(column, original_value):
    """
    Return a description of detected formatting issues.
    
    """
    if original_value is None or original_value == "":
        return None
    value = original_value
    trimmed_value = value.strip()
    issues = []

    # A. Leading/trailing whitespace
    if value != trimmed_value:
        issues.append("Leading/trailing whitespace")

    # B. Customer ID and transaction ID prefixes
    if column in ID_PREFIXES:
        expected_prefix = ID_PREFIXES[column]
        if not trimmed_value.startswith(expected_prefix):
            if trimmed_value.upper().startswith(
                expected_prefix.upper()
            ):
                issues.append(
                    f"Incorrect prefix capitalization; "
                    f"expected {expected_prefix}"
                )
            else:
                issues.append(
                    f"Missing or incorrect prefix; "
                    f"expected {expected_prefix}"
                )

    # C. Card number
    if column == "card_number":
        # Remove spaces and hyphens only for length validation.
        digits_only = trimmed_value.replace(" ", "").replace("-", "")
        # Detect formatting separators that need standardization.
        if not trimmed_value.isdigit():
            if re.fullmatch(r"[0-9 -]+", trimmed_value):
                issues.append("Card number contains formatting separators")
            else:
                issues.append("Card number contains non-digit characters")
        # Validate the number of digits after removing separators.
        if digits_only.isdigit():
            digit_count = len(digits_only)
            if digit_count not in ALLOWED_CARD_NUMBER_LENGTHS:
                allowed_lengths = " or ".join(
                    str(length)
                    for length in sorted(ALLOWED_CARD_NUMBER_LENGTHS)
                )
                issues.append(
                    f"Unexpected card-number length; "
                    f"expected {allowed_lengths} digits"
                )

    # D. Timestamp format
    if column == TIMESTAMP_COLUMN:
        if not validate_timestamp(trimmed_value):
            issues.append(
                "Invalid timestamp format or date/time"
            )

    # E. Currency must be uppercase
    if column in UPPERCASE_COLUMNS:
        if trimmed_value != trimmed_value.upper():
            issues.append("Must be uppercase")
    if column == "currency":
        if not re.fullmatch(r"[A-Z]{3}", trimmed_value):
            issues.append("Invalid currency-code format; expected three uppercase letters")

    # F. Merchant type must be lowercase
    if column in LOWERCASE_COLUMNS:
        if trimmed_value != trimmed_value.lower():
            issues.append("Must be lowercase")

    # G. Merchant category capitalization
    if column in CAPITALIZE_COLUMNS:
        if trimmed_value != trimmed_value.capitalize():
            issues.append(
                "Must have first character uppercase "
                "and remaining characters lowercase"
            )

    # H. Amount must be numeric
    if column == AMOUNT_COLUMN:
        try:
            float(trimmed_value)
        except ValueError:
            issues.append(
                "Invalid numeric format; expected a number "
                "such as 604.8"
            )

    # I. Boolean columns must contain True or False
    if column in BOOLEAN_COLUMNS:
        normalized_value = trimmed_value.lower()
        if normalized_value not in BOOLEAN_VALUES:
            issues.append(
                "Invalid boolean value; expected True or False"
            )
        elif trimmed_value not in {"True", "False"}:
            issues.append(
                "Incorrect boolean capitalization"
            )
    # J. Return detected issues only
    if issues:
        return "; ".join(issues)
    return None

# 5. MAIN PROFILING PROCESS

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count each distinct problematic value.
    issue_counts = Counter()

    # Store the details required for the report.
    issue_details = {}

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(
                "The input CSV does not contain a header."
            )
        dataset_columns = set(reader.fieldnames)
        configured_columns = (
            BOOLEAN_COLUMNS
            | UPPERCASE_COLUMNS
            | LOWERCASE_COLUMNS
            | CAPITALIZE_COLUMNS
            | set(ID_PREFIXES)
            | WHITESPACE_ONLY_COLUMNS
            | {AMOUNT_COLUMN, TIMESTAMP_COLUMN}
        )
        missing_columns = configured_columns - dataset_columns
        if missing_columns:
            print(
                "Warning: These configured columns "
                "were not found in the CSV:",
                sorted(missing_columns)
            )
        # Process one row at a time to avoid loading the
        # entire dataset into memory.
        for row in reader:
            for column in reader.fieldnames:
                # Ignore columns without configured rules.
                if column not in configured_columns:
                    continue
                original_value = row.get(column)
                # Missing values are handled by another task.
                if original_value is None or original_value == "":
                    continue
                issue = check_format(
                    column,
                    original_value
                )
                if issue is None:
                    continue
                # Group identical problematic values and issues.
                key = (
                    column,
                    original_value,
                    issue,
                )
                issue_counts[key] += 1
                if key not in issue_details:
                    issue_details[key] = {
                        "column": column,
                        "original_value": original_value,
                        "issue_detected": issue,
                    }

    # 6. WRITE PROFILING REPORT
    report_columns = [
        "column",
        "original_value",
        "count",
        "issue_detected",
    ]
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=report_columns
        )
        writer.writeheader()
        # Sort by column and original value for easier review.
        for key in sorted(
            issue_counts,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        ):
            details = issue_details[key]
            writer.writerow({
                "column": details["column"],
                "original_value": details["original_value"],
                "count": issue_counts[key],
                "issue_detected": details["issue_detected"],
            })

    # 7. SUMMARY
    print("\nFormat profiling completed successfully.")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Report saved to: {OUTPUT_FILE}")
    print(
        f"Distinct problematic values: {len(issue_counts)}"
    )
    print(
        "Total problematic occurrences: "
        f"{sum(issue_counts.values())}"
    )

if __name__ == "__main__":
    main()
