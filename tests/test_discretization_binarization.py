import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "phase_1"

spec = importlib.util.spec_from_file_location(
    "discretization_binarization", ROOT / "src" / "phase_1" / "6_2_discretization_binarization.py")
discretization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discretization)

BRANDS = ["visa", "mastercard", "amex", "discover"]


def make_frame(n=120):
    """A small 6A-like dataset: amounts 1..n, activity 1..n, velocity 1..n, hours 0..23, every brand."""
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"TX_{i}",
            "is_fraud": "True" if i % 4 == 0 else "False",
            "card_present": "False", "high_risk_merchant": "True" if i % 2 else "False",
            "weekend_transaction": "False",
            "amount_usd": str(i + 1),
            "customer_txn_count": str(i + 1),
            "velocity_last_hour": ("{'num_transactions': %d, 'total_amount': 1.0}" if i % 2 else
                                   '{"num_transactions": %d, "total_amount": 1.0}') % (i + 1),
            "customer_avg_amount_usd": str((i + 1) * 10),
            "transaction_hour": f"{i % 24}.0",
            "customers_per_device": str(1 + i % 3),
            "card_brand": BRANDS[i % 4],
        })
    rows[0]["amount_usd"] = "-1337.78"          # the -999 placeholder after conversion to USD
    rows[1]["velocity_last_hour"] = "?"         # unreadable velocity text
    return pd.DataFrame(rows)


class TestDiscretizationBinarization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        cls.input = folder / "6A.csv"
        make_frame().to_csv(cls.input, index=False)
        cls.result, cls.report = discretization.run(cls.input, folder / "6B.csv", folder / "6C.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_rows_lost(self):
        self.assertEqual(len(self.result), len(make_frame()))

    # discretization
    def test_tertiles_have_equal_size(self):
        for column in ("transaction_amount_category", "transaction_frequency_category",
                       "customer_activity_level", "customer_spending_level"):
            counts = self.result[column].value_counts()
            known = counts.drop("Unknown", errors="ignore").sum()
            for level in ["Low", "Medium", "High"]:
                self.assertAlmostEqual(counts[level], known / 3, delta=1, msg=column)

    def test_tertiles_are_ordered(self):
        self.assertEqual(self.result.loc[2, "customer_activity_level"], "Low")      # 3 transactions
        self.assertEqual(self.result.loc[119, "customer_activity_level"], "High")   # 120 transactions
        self.assertEqual(self.result.loc[119, "transaction_amount_category"], "High")

    def test_placeholder_amount_is_unknown(self):
        self.assertEqual(self.result.loc[0, "transaction_amount_category"], "Unknown")

    def test_frequency_reads_velocity_text(self):
        self.assertEqual(self.result.loc[119, "transaction_frequency_category"], "High")   # 120 in the last hour
        self.assertEqual(self.result.loc[2, "transaction_frequency_category"], "Low")      # double quotes
        self.assertEqual(self.result.loc[1, "transaction_frequency_category"], "Unknown")  # unreadable text

    def test_transaction_time_period(self):
        expected = {0: "Night", 5: "Night", 6: "Morning", 11: "Morning",
                    12: "Afternoon", 17: "Afternoon", 18: "Evening", 23: "Evening"}
        for hour, period in expected.items():
            self.assertEqual(self.result.loc[hour, "transaction_time_period"], period, hour)

    # binarization
    def test_booleans_become_0_and_1(self):
        for column in discretization.BOOLEAN_COLUMNS:
            self.assertTrue(set(self.result[column]) <= {0, 1}, column)
        self.assertEqual(self.result.loc[0, "is_fraud"], 1)
        self.assertEqual(self.result.loc[1, "is_fraud"], 0)

    def test_invalid_boolean_rejected(self):
        with self.assertRaises(ValueError):
            discretization.binarize_boolean(pd.Series(["True", "maybe"], name="card_present"))

    def test_shared_device(self):
        self.assertEqual(self.result.loc[0, "is_shared_device"], 0)   # 1 customer
        self.assertEqual(self.result.loc[1, "is_shared_device"], 1)   # 2 customers
        self.assertEqual(self.result.loc[2, "is_shared_device"], 1)   # 3 customers

    def test_card_brand_one_hot(self):
        one_hot = self.result[[f"card_brand_{b}" for b in BRANDS]]
        self.assertTrue((one_hot.sum(axis=1) == 1).all())             # exactly one brand per row
        for i, brand in enumerate(BRANDS):
            self.assertEqual(self.result.loc[i, f"card_brand_{brand}"], 1)

    def test_replaced_and_skipped_columns_not_created(self):
        for column in ("is_night_transaction", "hour_period", "amount_usd_level",
                       "distance_category", "category_diversity_level"):
            self.assertNotIn(column, self.result.columns)

    # report
    def test_report_has_limits_and_fraud_rates(self):
        amount = self.report[self.report["feature"] == "transaction_amount_category"]
        self.assertEqual(list(amount["group"]), ["Low", "Medium", "High", "Unknown"])
        self.assertTrue((amount["lower"].iloc[:3] != "").all())

    def test_leakage_warning(self):
        self.assertTrue(discretization.leakage_warning(500, 1.0))
        self.assertTrue(discretization.leakage_warning(500, 0.0))
        self.assertFalse(discretization.leakage_warning(500, 0.35))
        self.assertFalse(discretization.leakage_warning(20, 1.0))     # too few rows to judge


def real_outputs_available():
    paths = [DATA / name for name in ("6A_FEATURE_CREATION_DATASET.csv", "6B_DISCRETIZED_BINARIZED_DATASET.csv")]
    if not all(path.exists() for path in paths):
        return False
    with paths[0].open(encoding="utf-8") as f:
        return not f.readline().startswith("version https://git-lfs")


@unittest.skipUnless(real_outputs_available(), "real dataset / outputs not present")
class TestRealOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        read = lambda name: pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
        cls.created = read("6A_FEATURE_CREATION_DATASET.csv")
        cls.final = read("6B_DISCRETIZED_BINARIZED_DATASET.csv")

    def test_row_count_matches(self):
        self.assertEqual(len(self.final), len(self.created))

    def test_target_is_binary_and_fraud_count_kept(self):
        self.assertEqual(set(self.final["is_fraud"]), {"0", "1"})
        self.assertEqual((self.final["is_fraud"] == "1").sum(), (self.created["is_fraud"] == "True").sum())


if __name__ == "__main__":
    unittest.main()