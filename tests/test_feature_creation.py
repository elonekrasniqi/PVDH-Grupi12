import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "phase_1"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / "phase_1" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


creation = load_script("feature_creation", "6_1_feature_creation.py")


def row(i, **values):
    base = {
        "transaction_id": f"TX_{i}", "customer_id": f"C{i}", "timestamp": f"2024-09-30 10:{i % 60:02d}:00+00:00",
        "amount_usd": "50", "card_number": "4111111111111111", "is_fraud": "False",
        "distance_from_home": "0.0", "transaction_hour": "14.0", "device_fingerprint": f"dev{i}",
    }
    return {**base, **values}


def make_frame():
    return pd.DataFrame([
        row(1, customer_id="ANNA", amount_usd="100", distance_from_home="1.0", transaction_hour="2.0",
            device_fingerprint="d1"),
        row(2, customer_id="ANNA", amount_usd="300", is_fraud="True", transaction_hour="23.0",
            device_fingerprint="d1"),
        row(3, customer_id="ANNA", amount_usd="-999", distance_from_home="1.0", transaction_hour="5.0",
            device_fingerprint=" shared "),  # -999 placeholder
        row(4, customer_id="BEN", amount_usd="80", card_number="371449635398431",    # amex
            device_fingerprint="shared"),
        row(5, card_number="5500 0000 0000 0004"),                                  # mastercard, with spaces
        row(6, card_number="6011000990139424"),                                     # discover
        row(7, card_number="9999999999999999", is_fraud="True"),                    # unknown network
    ])


class TestFeatureCreation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        cls.input = folder / "sample.csv"
        make_frame().to_csv(cls.input, index=False)
        cls.result, cls.report = creation.run(cls.input, cls.input, folder / "6A.csv", folder / "6B.csv")
        cls.rows = cls.result.set_index("transaction_id")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def value(self, transaction, feature):
        return self.rows.loc[transaction, feature]

    def test_original_columns_and_rows_unchanged(self):
        original = pd.read_csv(self.input, dtype=str, keep_default_na=False)
        self.assertEqual(len(self.result), len(original))
        pd.testing.assert_frame_equal(self.result[original.columns], original)

    def test_exactly_the_new_features_are_added(self):
        added = [c for c in self.result.columns if c not in make_frame().columns]
        self.assertEqual(added, list(creation.NEW_FEATURES))

    def test_customer_txn_count(self):
        self.assertEqual(self.value("TX_1", "customer_txn_count"), 3)   # ANNA
        self.assertEqual(self.value("TX_4", "customer_txn_count"), 1)   # BEN

    def test_customer_avg_amount_ignores_placeholder(self):
        self.assertEqual(self.value("TX_1", "customer_avg_amount_usd"), 200.0)   # (100 + 300) / 2, -999 ignored
        self.assertEqual(self.value("TX_4", "customer_avg_amount_usd"), 80.0)

    def test_customer_away_share(self):
        self.assertAlmostEqual(self.value("TX_1", "customer_away_share"), 2 / 3, places=4)   # ANNA: 2 of 3 away
        self.assertEqual(self.value("TX_4", "customer_away_share"), 0.0)                    # BEN: always home

    def test_customer_night_share(self):
        self.assertAlmostEqual(self.value("TX_1", "customer_night_share"), 2 / 3, places=4)  # 02:00 and 05:00; 23:00 is not night
        self.assertEqual(self.value("TX_4", "customer_night_share"), 0.0)                   # 14:00

    def test_same_value_for_every_transaction_of_a_customer(self):
        anna = self.result[self.result["customer_id"] == "ANNA"]
        for feature in ("customer_txn_count", "customer_avg_amount_usd", "customer_away_share", "customer_night_share"):
            self.assertEqual(anna[feature].nunique(), 1, feature)

    def test_customers_per_device(self):
        self.assertEqual(self.value("TX_1", "customers_per_device"), 1)   # d1: only ANNA (twice)
        self.assertEqual(self.value("TX_4", "customers_per_device"), 2)   # shared: ANNA and BEN, spaces ignored

    def test_leakage_warning_for_perfect_groups(self):
        fraud = pd.Series([1] * 150 + [0] * 150)
        perfect = pd.Series([2] * 150 + [1] * 150)          # value 2 is always fraud
        mixed = pd.Series([2, 1] * 150)
        self.assertTrue(creation.rates_and_warning(perfect, fraud)[1])
        self.assertFalse(creation.rates_and_warning(mixed, fraud)[1])

    def test_card_brand(self):
        expected = {"TX_1": "visa", "TX_4": "amex", "TX_5": "mastercard", "TX_6": "discover", "TX_7": "other"}
        for transaction, brand in expected.items():
            self.assertEqual(self.value(transaction, "card_brand"), brand)

    def test_is_fraud_is_not_used_to_build_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            make_frame().assign(is_fraud="False").to_csv(path, index=False)
            other, _ = creation.run(path, path, Path(folder) / "6A.csv", Path(folder) / "6B.csv")
        for feature in creation.NEW_FEATURES:
            pd.testing.assert_series_equal(other[feature], self.result[feature], check_names=False)

    def test_report_lists_every_new_feature(self):
        self.assertEqual(list(self.report["feature"]), list(creation.NEW_FEATURES))


def real_outputs_available():
    paths = [DATA / name for name in ("5B_STRATIFIED_SAMPLE.csv", "6A_FEATURE_CREATION_DATASET.csv")]
    if not all(path.exists() for path in paths):
        return False
    with paths[0].open(encoding="utf-8") as f:
        return not f.readline().startswith("version https://git-lfs")


@unittest.skipUnless(real_outputs_available(), "real dataset / outputs not present")
class TestRealOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        read = lambda name: pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
        cls.sample = read("5B_STRATIFIED_SAMPLE.csv")
        cls.created = read("6A_FEATURE_CREATION_DATASET.csv")

    def test_row_count_matches(self):
        self.assertEqual(len(self.created), len(self.sample))

    def test_new_features_present(self):
        for feature in creation.NEW_FEATURES:
            self.assertIn(feature, self.created.columns)

    def test_original_columns_unchanged(self):
        pd.testing.assert_frame_equal(self.created[self.sample.columns], self.sample)


if __name__ == "__main__":
    unittest.main()