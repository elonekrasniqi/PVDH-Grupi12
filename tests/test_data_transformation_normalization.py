import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "phase_1"

spec = importlib.util.spec_from_file_location(
    "data_transformation_normalization", ROOT / "src" / "phase_1" / "6_3_data_transformation_normalization.py")
transformation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformation)

LEVELS = ["Low", "Medium", "High"]
PERIODS = ["Night", "Morning", "Afternoon", "Evening"]


def make_frame(n=120):
    """A small 6B-like dataset (already discretized and binarized)."""
    rows = []
    for i in range(n):
        velocity = ("{'num_transactions': %d, 'total_amount': %f, 'unique_merchants': %d, "
                    "'unique_countries': %d, 'max_single_amount': %f}")
        if i % 2:
            velocity = velocity.replace("'", '"')
        rows.append({
            "transaction_id": f"TX_{i}",
            "is_fraud": str(int(i % 4 == 0)),
            "amount_usd": str(round(math.exp(i / 12), 2)),         # strongly right-skewed
            "customer_avg_amount_usd": str((i + 1) * 10),
            "customer_txn_count": str(i + 1),
            "customer_away_share": str(round(i / (n - 1), 4)),
            "customer_night_share": str(round((i % 10) / 10, 2)),
            "customers_per_device": str(1 + i % 5),
            "transaction_hour": f"{i % 24}.0",
            "velocity_last_hour": velocity % (i + 1, (i + 1) * 1000.5, 50 + i % 50, 1 + i % 12, (i + 1) * 10.0),
            "transaction_amount_category": LEVELS[i % 3],
            "transaction_frequency_category": LEVELS[(i + 1) % 3],
            "customer_activity_level": LEVELS[(i + 2) % 3],
            "customer_spending_level": LEVELS[i % 3],
            "transaction_time_period": PERIODS[i % 4],
        })
    rows[0]["amount_usd"] = "-1337.78"                 # the -999 placeholder after USD conversion
    rows[0]["transaction_amount_category"] = "Unknown"
    rows[1]["customer_avg_amount_usd"] = ""            # empty value
    return pd.DataFrame(rows)


class TestDataTransformationNormalization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        cls.input = folder / "6B.csv"
        cls.frame = make_frame()
        cls.frame.to_csv(cls.input, index=False)
        cls.result, cls.report = transformation.run(cls.input, folder / "6D.csv", folder / "6E.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_rows_lost_and_original_columns_kept(self):
        self.assertEqual(len(self.result), len(self.frame))
        for column in self.frame.columns:
            self.assertIn(column, self.result.columns)
        self.assertEqual(list(self.result["transaction_id"]), list(self.frame["transaction_id"]))

    # transformation
    def test_velocity_text_is_parsed(self):
        self.assertEqual(self.result.loc[9, "velocity_num_transactions"], 10)
        self.assertEqual(self.result.loc[10, "velocity_unique_countries"], 11)   # single quotes
        self.assertEqual(self.result.loc[11, "velocity_unique_countries"], 12)   # double quotes
        bad = transformation.parse_velocity("?")
        self.assertTrue(all(math.isnan(v) for v in bad.values()))

    def test_cyclic_hour(self):
        self.assertAlmostEqual(self.result.loc[0, "hour_sin"], 0.0, places=5)     # 0h
        self.assertAlmostEqual(self.result.loc[0, "hour_cos"], 1.0, places=5)
        self.assertAlmostEqual(self.result.loc[6, "hour_sin"], 1.0, places=5)     # 6h
        self.assertAlmostEqual(self.result.loc[12, "hour_cos"], -1.0, places=5)   # 12h
        # 23h and 1h are equally close to 0h
        self.assertAlmostEqual(self.result.loc[23, "hour_cos"], self.result.loc[1, "hour_cos"], places=5)

    def test_ordinal_codes(self):
        self.assertEqual(self.result.loc[3, "transaction_amount_category_code"], 0)   # Low
        self.assertEqual(self.result.loc[4, "transaction_amount_category_code"], 1)   # Medium
        self.assertEqual(self.result.loc[5, "transaction_amount_category_code"], 2)   # High
        self.assertEqual(self.result.loc[0, "transaction_amount_category_code"], 1)   # Unknown -> median

    def test_time_period_one_hot(self):
        one_hot = self.result[[f"time_period_{p.lower()}" for p in PERIODS]]
        self.assertTrue((one_hot.sum(axis=1) == 1).all())
        self.assertEqual(self.result.loc[2, "time_period_afternoon"], 1)

    def test_invalid_amount_is_flagged_and_imputed(self):
        self.assertEqual(self.result.loc[0, "amount_usd_invalid"], 1)
        self.assertEqual(self.result["amount_usd_invalid"].sum(), 1)
        valid = pd.to_numeric(self.frame["amount_usd"]).iloc[1:]
        self.assertAlmostEqual(self.result.loc[0, "amount_usd"], valid.median())

    def test_empty_value_is_imputed(self):
        self.assertFalse(pd.isna(self.result.loc[1, "customer_avg_amount_usd"]))
        row = self.report[self.report["source"] == "customer_avg_amount_usd"].iloc[0]
        self.assertEqual(row["imputed_rows"], 1)

    def test_log_reduces_skew(self):
        row = self.report[self.report["feature"] == "amount_usd_log_z"].iloc[0]
        self.assertLess(abs(row["skew_after"]), abs(row["skew_before"]))

    # normalization
    def test_z_score_columns(self):
        for column in [c for c in self.result.columns if c.endswith("_z")]:
            self.assertAlmostEqual(self.result[column].mean(), 0, places=4, msg=column)
            self.assertAlmostEqual(self.result[column].std(ddof=0), 1, places=4, msg=column)

    def test_min_max_columns(self):
        for column in [c for c in self.result.columns if c.endswith("_minmax")]:
            self.assertAlmostEqual(self.result[column].min(), 0, msg=column)
            self.assertAlmostEqual(self.result[column].max(), 1, msg=column)

    def test_constant_column_does_not_divide_by_zero(self):
        values, _, scale = transformation.z_score(pd.Series([5.0, 5.0, 5.0]))
        self.assertTrue((values == 0).all())
        self.assertEqual(scale, 1.0)
        values, _, scale = transformation.min_max(pd.Series([2.0, 2.0]))
        self.assertTrue((values == 0).all())

    def test_no_missing_values_in_new_columns(self):
        new = [c for c in self.result.columns if c not in self.frame.columns]
        self.assertFalse(self.result[new].isna().any().any())

    def test_target_is_not_scaled(self):
        self.assertEqual(list(self.result["is_fraud"]), list(self.frame["is_fraud"]))
        self.assertFalse(any(c.startswith("is_fraud_") for c in self.result.columns))

    # report
    def test_report_lists_every_scaled_feature(self):
        expected = {*(f"{c}_log_z" for c in transformation.LOG_Z),
                    *(f"{c}_z" for c in transformation.Z_ONLY),
                    *(f"{c}_minmax" for c in transformation.MIN_MAX)}
        self.assertEqual(set(self.report["feature"]), expected)

    def test_missing_input_column_rejected(self):
        folder = Path(self.tmp.name)
        path = folder / "bad.csv"
        self.frame.drop(columns=["velocity_last_hour"]).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            transformation.run(path, folder / "bad_out.csv", folder / "bad_report.csv")


def real_outputs_available():
    paths = [DATA / name for name in ("6B_DISCRETIZED_BINARIZED_DATASET.csv", "6D_TRANSFORMED_NORMALIZED_DATASET.csv")]
    if not all(path.exists() for path in paths):
        return False
    with paths[0].open(encoding="utf-8") as f:
        return not f.readline().startswith("version https://git-lfs")


@unittest.skipUnless(real_outputs_available(), "real dataset / outputs not present")
class TestRealOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.discretized = pd.read_csv(DATA / "6B_DISCRETIZED_BINARIZED_DATASET.csv", dtype=str, keep_default_na=False)
        cls.final = pd.read_csv(DATA / "6D_TRANSFORMED_NORMALIZED_DATASET.csv", low_memory=False)

    def test_row_count_and_target_kept(self):
        self.assertEqual(len(self.final), len(self.discretized))
        self.assertEqual(int(self.final["is_fraud"].sum()), int((self.discretized["is_fraud"] == "1").sum()))

    def test_scaled_ranges(self):
        for column in [c for c in self.final.columns if c.endswith("_minmax")]:
            self.assertGreaterEqual(self.final[column].min(), 0)
            self.assertLessEqual(self.final[column].max(), 1)
        for column in [c for c in self.final.columns if c.endswith("_z")]:
            self.assertAlmostEqual(self.final[column].mean(), 0, places=3)


if __name__ == "__main__":
    unittest.main()
