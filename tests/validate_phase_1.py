import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/phase_1"

DEDUPLICATED = DATA / "3B_DUPLICATES_REMOVED_DATASET.csv"
INTEGRATED = DATA / "4A_INTEGRATED_DATASET.csv"

AGGREGATED = [
    "4B_AGGREGATED_BY_COUNTRY.csv",
    "4C_AGGREGATED_BY_CURRENCY.csv",
    "4D_AGGREGATED_BY_DATE.csv",
    "4E_AGGREGATED_BY_MERCHANT_CATEGORY.csv"
]


def count_source():
    """Rows and fraud rows of the deduplicated dataset (input of task 4)."""
    rows = fraud = unknown_currency = 0
    with DEDUPLICATED.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            fraud += row["is_fraud"].strip().lower() in ("1", "true")
            unknown_currency += row["currency"].strip().upper() == "UNKNOWN"
    return rows, fraud, unknown_currency


def main():
    expected_total, expected_fraud, expected_imputed = count_source()
    total = 0
    fraud = 0
    imputed = 0
    missing_usd = 0

    with INTEGRATED.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            total += 1

            if row["is_fraud"].strip().lower() in ("1", "true"):
                fraud += 1

            if row["currency_imputed"] == "1":
                imputed += 1

            if not row["amount_usd"].strip():
                missing_usd += 1

    print("INTEGRATED DATASET")
    print("Transactions:", total)
    print("Fraud transactions:", fraud)
    print("Imputed currencies:", imputed)
    print("Missing USD amounts:", missing_usd)

    assert total == expected_total, "Transactions lost or added during integration"
    assert fraud == expected_fraud, "Fraud count changed during integration"
    assert imputed == expected_imputed, "Not every unknown currency was imputed"
    assert missing_usd == 0, "Some USD amounts are missing"

    for name in AGGREGATED:
        path = DATA / name

        with path.open(
            encoding="utf-8-sig", newline=""
        ) as f:
            rows = list(csv.DictReader(f))

        grouped_total = sum(
            int(row["total_transactions"]) for row in rows
        )
        grouped_fraud = sum(
            int(row["fraud_transactions"]) for row in rows
        )

        assert grouped_total == total, (
            f"{name}: transaction count mismatch"
        )
        assert grouped_fraud == fraud, (
            f"{name}: fraud count mismatch"
        )

        print(f"{name}: PASS")

    print("\nALL PHASE 1 VALIDATIONS PASSED")


if __name__ == "__main__":
    main()