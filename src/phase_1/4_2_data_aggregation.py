
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "dataset/phase_1/4A_INTEGRATED_DATASET.csv"
OUTPUT_DIR = ROOT / "dataset/phase_1"


def get_group_value(row, group_column):
    if group_column == "date":
        return row["timestamp"][:10]

    if group_column == "country":
        return row["country"].strip().upper()

    if group_column == "currency_resolved":
        return row["currency_resolved"].strip().upper()

    return row[group_column].strip()


def aggregate_data(group_column, output_name):
    groups = defaultdict(lambda: {
        "total_transactions": 0,
        "fraud_transactions": 0,
        "total_amount_usd": 0.0
    })

    with INPUT.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            group = get_group_value(row, group_column)
            stats = groups[group]

            stats["total_transactions"] += 1

            if row["is_fraud"].strip().lower() in ("1", "true"):
                stats["fraud_transactions"] += 1

            if row["amount_usd"].strip():
                stats["total_amount_usd"] += float(row["amount_usd"])

    output = OUTPUT_DIR / output_name

    columns = [
        group_column,
        "total_transactions",
        "fraud_transactions",
        "fraud_rate",
        "total_amount_usd",
        "average_amount_usd"
    ]

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for group, stats in sorted(groups.items()):
            total = stats["total_transactions"]
            fraud = stats["fraud_transactions"]
            total_amount = stats["total_amount_usd"]

            writer.writerow({
                group_column: group,
                "total_transactions": total,
                "fraud_transactions": fraud,
                "fraud_rate": round(fraud / total * 100, 2),
                "total_amount_usd": round(total_amount, 2),
                "average_amount_usd": round(total_amount / total, 2)
            })

    print(f"Created: {output.name} | Groups: {len(groups)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_data(
        "country",
        "4B_AGGREGATED_BY_COUNTRY.csv"
    )

    aggregate_data(
        "currency_resolved",
        "4C_AGGREGATED_BY_CURRENCY.csv"
    )

    aggregate_data(
        "date",
        "4D_AGGREGATED_BY_DATE.csv"
    )

    aggregate_data(
        "merchant_category",
        "4E_AGGREGATED_BY_MERCHANT_CATEGORY.csv"
    )

    print("\nDATA AGGREGATION COMPLETED")


if __name__ == "__main__":
    main()
