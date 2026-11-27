import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summary_statistics", ROOT / "src" / "phase_2" / "4_1_summary_statistics.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def make_data(n=200):
    """Small 3A-like data."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({feature: rng.integers(1, 100, n).astype(float) for feature in stats.NUMERIC})
    df["is_fraud"] = [1] * (n // 4) + [0] * (n - n // 4)
    df["country"] = rng.choice(["USA", "Japan", "Brazil"], n)
    df["transaction_hour"] = rng.integers(0, 24, n)
    df["amount_usd_capped"] = df["amount_usd"].clip(upper=50)
    return df


class TestSummaryStatistics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = make_data()

    def test_known_values(self):
        result = stats.describe_numeric(pd.Series([1, 2, 2, 3, 4, 5, 6, 7, 8, 100], name="x"))
        self.assertAlmostEqual(result["mean"], 13.8)
        self.assertEqual(result["median"], 4.5)
        self.assertEqual(result["mode"], 2)
        self.assertEqual(result["mode_frequency"], 2)
        self.assertAlmostEqual(result["trimmed_mean_10"], 4.625)   # without 1 and 100
        self.assertEqual(result["range"], 99)
        self.assertAlmostEqual(result["aad"], 17.24)
        self.assertEqual(result["mad"], 2.5)

    def test_missing_values_are_counted(self):
        result = stats.describe_numeric(pd.Series([1.0, np.nan, 3.0], name="x"))
        self.assertEqual((result["count"], result["missing"]), (2, 1))

    def test_one_row_per_numeric_feature(self):
        self.assertEqual(list(stats.numeric_summary(self.data)["feature"]), stats.NUMERIC)

    def test_class_counts_add_up(self):
        result = stats.class_summary(self.data)
        self.assertTrue((result.groupby("feature")["count"].sum() == len(self.data)).all())

    def test_frequencies_add_up(self):
        result = stats.categorical_frequencies(self.data)
        for _, table in result.groupby("feature"):
            self.assertEqual(table["count"].sum(), len(self.data))
            self.assertEqual(table["fraud_count"].sum(), self.data["is_fraud"].sum())

    def test_case_variants_are_merged(self):
        data = self.data.head(4).copy()
        data["country"] = ["Brazil", "Brazil", "BRAZIL", " brazil "]
        result = stats.categorical_frequencies(data)
        table = result[result["feature"] == "country"]
        self.assertEqual(list(table["value"]), ["Brazil"])
        self.assertEqual(table["count"].iloc[0], 4)

    def test_treatment_comparison(self):
        result = stats.treatment_comparison(self.data).set_index("statistic")
        self.assertEqual(result.loc["max", "capped"], 50)
        self.assertEqual(result.loc["max", "rows_changed"], (self.data["amount_usd"] > 50).sum())

    def test_text_target(self):
        values = stats.fraud_flag(pd.Series(["True", "False", "1", "0"]))
        self.assertEqual(list(values), [1, 0, 1, 0])

    def test_lfs_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "3A.csv"
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                stats.load(path)

    def test_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "3A.csv"
            self.data.to_csv(path, index=False)
            stats.main(path, folder)
            self.assertEqual(len(list(Path(folder).glob("4?_*"))), 6)


if __name__ == "__main__":
    unittest.main()