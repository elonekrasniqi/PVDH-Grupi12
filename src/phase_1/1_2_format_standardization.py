import csv
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "dataset" / "synthetic_fraud_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "phase_1"
OUTPUT_FILE = OUTPUT_DIR / "1B_FORMAT_STANDARDIZED_DATASET.csv"

ID_PREFIXES = {
    "customer_id": "CUST_",
    "transaction_id": "TX_",
}

WHITESPACE_COLUMNS = {
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

TIMESTAMP_COLUMN = "timestamp"
CARD_NUMBER_COLUMN = "card_number"
BOOLEAN_COLUMN = "card_present"
HOUR_COLUMN = "transaction_hour"


def standardize_id(value, prefix):
    """Bring identifiers to PREFIX_body.

    Handled variants: 'cust_123' / 'CUST_123' (prefix case),
    'cust-123' / 'CUST-123' (hyphen instead of underscore) and '123'
    (prefix missing). The body after the prefix is kept as it is.
    """
    value = value.strip()
    prefix_length = len(prefix)
    head = value[:prefix_length].upper().replace("-", "_")

    if head == prefix:
        return prefix + value[prefix_length:]

    if value.isdigit():
        return f"{prefix}{value}"

    return value


def standardize_hour(value):
    """Whole-number hours written as decimals ('15.0') -> '15'.

    Values that are not a whole hour between 0 and 23 are left unchanged so
    the validation step (1_3) can still report them.
    """
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and 0 <= number <= 23:
        return str(int(number))
    return value


def standardize_card_number(value):
    value = value.strip()

    if re.fullmatch(r"[0-9 -]+", value):
        return value.replace(" ", "").replace("-", "")

    return value


def standardize_timestamp(value):
    value = value.strip()

    timestamp_formats = (
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    )

    for timestamp_format in timestamp_formats:
        try:
            parsed = datetime.strptime(value, timestamp_format)

            if parsed.tzinfo is not None:
                return parsed.isoformat(sep=" ")

            if "." in value:
                return parsed.isoformat(sep=" ")

            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return value


def standardize_value(column, value):
    if value is None or value == "":
        return value

    if column in WHITESPACE_COLUMNS:
        value = value.strip()

    if column in ID_PREFIXES:
        return standardize_id(value, ID_PREFIXES[column])

    if column == CARD_NUMBER_COLUMN:
        return standardize_card_number(value)

    if column == TIMESTAMP_COLUMN:
        return standardize_timestamp(value)

    if column == HOUR_COLUMN:
        return standardize_hour(value)

    if column == BOOLEAN_COLUMN:
        normalized_value = value.strip().lower()
        if normalized_value == "yes":
            return "True"
        if normalized_value == "no":
            return "False"
        return value.strip()

    if column == "currency":
        return value.strip().upper()

    if column == "merchant_type":
        return value.strip().lower()

    if column == "merchant_category":
        return value.strip().capitalize()

    if column == "amount":
        return value.strip()

    return value


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_processed = 0
    cells_changed = 0
    changes_by_column = {}

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        if not reader.fieldnames:
            raise ValueError("The input CSV does not contain a header.")

        with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                rows_processed += 1

                for column in reader.fieldnames:
                    original_value = row.get(column)
                    standardized_value = standardize_value(column, original_value)

                    if standardized_value != original_value:
                        cells_changed += 1
                        changes_by_column[column] = (
                            changes_by_column.get(column, 0) + 1
                        )
                        row[column] = standardized_value

                writer.writerow(row)

    print("Standardization completed successfully.")
    print(f"Original dataset: {INPUT_FILE}")
    print(f"Standardized dataset: {OUTPUT_FILE}")
    print(f"Rows processed: {rows_processed}")
    print(f"Cells changed: {cells_changed}")
    print("Changes by column:")

    if changes_by_column:
        for column, count in sorted(changes_by_column.items()):
            print(f"  {column}: {count}")
    else:
        print("  No changes were made.")


if __name__ == "__main__":
    main()