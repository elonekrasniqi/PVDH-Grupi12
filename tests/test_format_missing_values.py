import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "phase_1"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standardization = load("format_standardization", "1_2_format_standardization.py")
analysis = load("missing_value_analysis", "2_1_missing_value_analysis.py")
handling = load("missing_value_handling", "2_2_missing_value_handling.py")
integration = load("dataset_integration", "4_1_dataset_integration.py")

COLUMNS = ["transaction_id", "customer_id", "timestamp", "amount", "currency", "country",
           "merchant_category", "card_type", "channel", "distance_from_home",
           "transaction_hour", "is_fraud"]


def row(i, amount="100", currency="USD", country="USA", hour="10", fraud="False", **extra):
    values = {
        "transaction_id": f"TX_{i}", "customer_id": f"CUST_{i}",
        "timestamp": f"2024-09-30 {int(hour) if hour.isdigit() else 7:02d}:00:00+00:00",
        "amount": amount, "currency": currency, "country": country,
        "merchant_category": "Retail", "card_type": "Gold Credit", "channel": "web",
        "distance_from_home": "0", "transaction_hour": hour, "is_fraud": fraud,
    }
    values.update(extra)
    return values


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class TestFormatStandardization(unittest.TestCase):
    def test_customer_id_variants(self):
        for value in ("CUST_123", "cust_123", "cust-123", "CUST-123", " 123 "):
            self.assertEqual(standardization.standardize_id(value, "CUST_"), "CUST_123", value)

    def test_transaction_id_body_is_kept(self):
        self.assertEqual(standardization.standardize_id("tx_DF6B57EF", "TX_"), "TX_DF6B57EF")

    def test_unknown_id_left_for_validation(self):
        self.assertEqual(standardization.standardize_id("c", "TX_"), "c")

    def test_hour_decimal_to_integer(self):
        self.assertEqual(standardization.standardize_value("transaction_hour", "15.0"), "15")
        self.assertEqual(standardization.standardize_value("transaction_hour", " 0.0 "), "0")

    def test_invalid_hours_left_for_validation(self):
        for value in ("15.5", "24.0", "abc"):
            self.assertEqual(standardization.standardize_hour(value), value)


class TestMissingValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        folder = Path(cls.folder.name)
        rows = [row(1, amount="100"), row(2, amount="300"),
                row(3, amount="10000", currency="JPY", country="JAPAN"),
                row(4, amount="30000", currency="JPY", country="JAPAN"),
                row(5, amount="-999", currency="JPY", country="JAPAN"),   # placeholder
                row(6, amount="", currency="?", country="JAPAN"),         # both missing
                row(7, hour=""),                                          # hour from timestamp
                row(8, channel="", card_type="null"),
                row(9, fraud="")]                                         # no target
        cls.input = folder / "1B.csv"
        write(cls.input, rows)
        cls.output = folder / "2B.csv"
        cls.report_path = folder / "2A.csv"
        cls.rows_read, cls.rows_written, cls.removed, cls.imputed = handling.handle_missing_values(
            cls.input, cls.output)
        cls.result = {r["transaction_id"]: r for r in read(cls.output)}
        _, cls.report = analysis.run(cls.input, cls.report_path)
        cls.report = {r["column"]: r for r in cls.report}

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_rows_without_target_removed(self):
        self.assertEqual((self.rows_read, self.rows_written, self.removed), (9, 8, 1))
        self.assertNotIn("TX_9", self.result)

    def test_placeholder_amount_imputed_with_same_currency_median(self):
        self.assertEqual(self.result["TX_5"]["amount"], "20000")

    def test_missing_currency_inferred_from_country_for_median(self):
        self.assertEqual(self.result["TX_6"]["amount"], "20000")
        self.assertEqual(self.result["TX_6"]["currency"], "Unknown")   # resolved and flagged in 4_1

    def test_no_amount_left_missing_or_non_positive(self):
        self.assertTrue(all(float(r["amount"]) > 0 for r in self.result.values()))

    def test_hour_taken_from_timestamp(self):
        self.assertEqual(self.result["TX_7"]["transaction_hour"], "7")

    def test_categorical_missing_becomes_unknown(self):
        self.assertEqual(self.result["TX_8"]["channel"], "Unknown")
        self.assertEqual(self.result["TX_8"]["card_type"], "Unknown")

    def test_valid_values_unchanged(self):
        self.assertEqual(self.result["TX_1"]["amount"], "100")
        self.assertEqual(self.result["TX_1"]["transaction_hour"], "10")

    def test_report_counts_and_placeholders(self):
        amount = self.report["amount"]
        self.assertEqual(int(amount["missing_count"]), 2)
        self.assertEqual(int(amount["empty_cells"]), 1)
        self.assertIn("'-999'=1", amount["placeholder_values"])
        self.assertIn("'?'=1", self.report["currency"]["placeholder_values"])
        self.assertEqual(int(self.report["is_fraud"]["missing_count"]), 1)
        self.assertEqual(self.report["is_fraud"]["strategy"], "remove row")
        self.assertEqual(self.report["transaction_id"]["strategy"], "none needed")

    def test_report_written(self):
        self.assertTrue(self.report_path.is_file())

    def test_country_currency_mapping_matches_integration(self):
        self.assertEqual(handling.COUNTRY_CURRENCY, integration.COUNTRY_CURRENCY)


if __name__ == "__main__":
    unittest.main()