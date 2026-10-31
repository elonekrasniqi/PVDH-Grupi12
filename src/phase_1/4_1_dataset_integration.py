
import csv
from pathlib import Path
from datetime import date
from collections import defaultdict, Counter
from bisect import bisect_right

ROOT = Path(__file__).resolve().parents[2]

TRANSACTIONS = ROOT / "dataset/phase_1/3B_DUPLICATES_REMOVED_DATASET.csv"
RATES = ROOT / "dataset/exchange_rates_2024.csv"
OUTPUT = ROOT / "dataset/phase_1/4A_INTEGRATED_DATASET.csv"


# Currency mapping based on the observed dataset relationship
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
    "USA": "USD"
}


def load_exchange_rates():
    rates = defaultdict(list)

    with RATES.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["base"].strip().upper() != "USD":
                continue

            rate = float(row["rate"])

            if rate <= 0:
                continue

            currency = row["quote"].strip().upper()
            rate_date = date.fromisoformat(row["date"])

            rates[currency].append((rate_date, rate))

    for currency in rates:
        rates[currency].sort()

    return rates


def find_exchange_rate(rates, currency, transaction_date):
    if currency == "USD":
        return 1.0, transaction_date

    history = rates.get(currency, [])

    if not history:
        return None, None

    dates = [item[0] for item in history]

    index = bisect_right(dates, transaction_date) - 1

    if index < 0:
        return None, None

    rate_date, rate = history[index]

    return rate, rate_date


def integrate_datasets():
    rates = load_exchange_rates()
    stats = Counter()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with TRANSACTIONS.open("r", encoding="utf-8-sig", newline="") as source, \
         OUTPUT.open("w", encoding="utf-8", newline="") as destination:

        reader = csv.DictReader(source)

        new_columns = [
            "currency_resolved",
            "currency_imputed",
            "exchange_rate",
            "exchange_rate_date",
            "amount_usd"
        ]

        writer = csv.DictWriter(
            destination,
            fieldnames=reader.fieldnames + new_columns
        )

        writer.writeheader()

        for row in reader:
            stats["total"] += 1

            original_currency = row["currency"].strip().upper()
            country = row["country"].strip().upper()

            currency = original_currency
            imputed = 0

            # Infer unknown currency using country
            if currency == "UNKNOWN":
                inferred_currency = COUNTRY_CURRENCY.get(country)

                if inferred_currency is not None:
                    currency = inferred_currency
                    imputed = 1
                    stats["imputed"] += 1

            row["currency_resolved"] = currency
            row["currency_imputed"] = imputed

            transaction_date = date.fromisoformat(
                row["timestamp"][:10]
            )

            rate, rate_date = find_exchange_rate(
                rates, currency, transaction_date
            )

            row["exchange_rate"] = ""
            row["exchange_rate_date"] = ""
            row["amount_usd"] = ""

            if rate is not None:
                try:
                    amount = float(row["amount"])

                    row["exchange_rate"] = rate
                    row["exchange_rate_date"] = rate_date.isoformat()
                    row["amount_usd"] = round(amount / rate, 2)

                    stats["converted"] += 1

                    if rate_date < transaction_date:
                        stats["previous_rate_used"] += 1

                except (ValueError, TypeError):
                    stats["invalid_amount"] += 1

            else:
                stats["missing_rate"] += 1
                stats[f"missing_{currency}"] += 1

            writer.writerow(row)

            if stats["total"] % 500000 == 0:
                print(f"Processed: {stats['total']:,} transactions")

    print("\nDATASET INTEGRATION COMPLETED")
    print(f"Total transactions: {stats['total']:,}")
    print(f"Converted: {stats['converted']:,}")
    print(f"Imputed currencies: {stats['imputed']:,}")
    print(f"Previous rates used: {stats['previous_rate_used']:,}")
    print(f"Missing exchange rates: {stats['missing_rate']:,}")
    print(f"Invalid amounts: {stats['invalid_amount']:,}")

    for key, value in stats.items():
        if key.startswith("missing_") and key != "missing_rate":
            print(f"{key}: {value:,}")


if __name__ == "__main__":
    integrate_datasets()
