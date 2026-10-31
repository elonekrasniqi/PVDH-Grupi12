
import csv
import unittest
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from math import isclose

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "phase_1"

SOURCE = DATA / "3B_DUPLICATES_REMOVED_DATASET.csv"
INTEGRATED = DATA / "4A_INTEGRATED_DATASET.csv"

AGGREGATIONS = {
      "country": DATA / "4B_AGGREGATED_BY_COUNTRY.csv",
      "currency_resolved": DATA / "4C_AGGREGATED_BY_CURRENCY.csv",
      "date": DATA / "4D_AGGREGATED_BY_DATE.csv",
      "merchant_category": DATA / "4E_AGGREGATED_BY_MERCHANT_CATEGORY.csv",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_value(row, column):
    if column == "date":
        return row["timestamp"][:10]
    if column in ("country", "currency_resolved"):
        return row[column].strip().upper()
    return row[column].strip()


class TestDatasetIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = read_csv(SOURCE)
        cls.integrated = read_csv(INTEGRATED)

    def test_no_transactions_lost(self):
        self.assertEqual(
            len(self.source),
            len(self.integrated)
        )

    def test_required_columns_exist(self):
        required = {
            "currency",
            "currency_resolved",
            "currency_imputed",
            "exchange_rate",
            "amount_usd",
            "is_fraud",
        }

        self.assertTrue(
            required.issubset(self.integrated[0].keys())
        )

    def test_all_amounts_converted(self):
        for row in self.integrated:
            self.assertNotEqual(
                row["amount_usd"].strip(), ""
            )

    def test_currency_imputation(self):
        count = 0

        for row in self.integrated:
            if row["currency"].strip().upper() == "UNKNOWN":
                count += 1

                self.assertEqual(
                    row["currency_imputed"], "1"
                )
                self.assertNotEqual(
                    row["currency_resolved"].strip(), ""
                )
                self.assertNotEqual(
                    row["currency_resolved"].strip().upper(),
                    "UNKNOWN"
                )

        source_unknown = sum(
            1 for row in self.source if row["currency"].strip().upper() == "UNKNOWN"
        )
        self.assertEqual(count, source_unknown)

    def test_usd_conversion_formula(self):
        for row in self.integrated:
            amount = float(row["amount"])
            rate = float(row["exchange_rate"])
            amount_usd = float(row["amount_usd"])

            self.assertGreater(rate, 0)

            expected = amount / rate

            self.assertTrue(
                isclose(
                    amount_usd,
                    expected,
                    rel_tol=1e-6,
                    abs_tol=0.02
                )
            )

    def test_usd_currency_rate(self):
        for row in self.integrated:
            if row["currency_resolved"].strip().upper() == "USD":
                self.assertTrue(
                    isclose(
                        float(row["exchange_rate"]),
                        1.0,
                        abs_tol=1e-9
                    )
                )


class TestDataAggregation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.integrated = read_csv(INTEGRATED)

    def test_all_aggregation_files_exist(self):
        for path in AGGREGATIONS.values():
            self.assertTrue(path.exists())

    def test_aggregation_totals(self):
        total = len(self.integrated)

        fraud = sum(
            row["is_fraud"].strip().lower() in ("1", "true")
            for row in self.integrated
        )

        for path in AGGREGATIONS.values():
            rows = read_csv(path)

            self.assertEqual(
                sum(int(r["total_transactions"]) for r in rows),
                total
            )

            self.assertEqual(
                sum(int(r["fraud_transactions"]) for r in rows),
                fraud
            )

    def test_fraud_rates(self):
        for path in AGGREGATIONS.values():
            for row in read_csv(path):
                total = int(row["total_transactions"])
                fraud = int(row["fraud_transactions"])

                expected = round(fraud / total * 100, 2)

                self.assertAlmostEqual(
                    float(row["fraud_rate"]),
                    expected,
                    places=2
                )

    def test_average_amounts(self):
        for path in AGGREGATIONS.values():
            for row in read_csv(path):
                total = int(row["total_transactions"])
                amount = float(row["total_amount_usd"])

                expected = amount / total

                self.assertTrue(
                    isclose(
                        float(row["average_amount_usd"]),
                        expected,
                        abs_tol=0.02
                    )
                )

    def test_group_counts_and_amounts(self):
        for column, path in AGGREGATIONS.items():
            expected = defaultdict(
                lambda: [0, 0, 0.0]
            )

            for row in self.integrated:
                key = group_value(row, column)
                stats = expected[key]

                stats[0] += 1
                stats[1] += (
                    row["is_fraud"].strip().lower()
                    in ("1", "true")
                )
                stats[2] += float(row["amount_usd"])

            actual = read_csv(path)

            self.assertEqual(len(actual), len(expected))

            for row in actual:
                key = row[column]
                count, fraud, amount = expected[key]

                self.assertEqual(
                    int(row["total_transactions"]), count
                )
                self.assertEqual(
                    int(row["fraud_transactions"]), fraud
                )
                self.assertTrue(
                    isclose(
                        float(row["total_amount_usd"]),
                        amount,
                        abs_tol=0.05
                    )
                )


if __name__ == "__main__":
    unittest.main()
