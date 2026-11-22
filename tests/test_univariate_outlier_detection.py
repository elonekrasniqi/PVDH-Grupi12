import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "univariate", ROOT / "src" / "phase_2" / "1_1_univariate_outlier_detection.py")
univariate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(univariate)


def make_data(n=300):
    """Small 8A-like data with two outliers added on purpose."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"transaction_id": [f"TX_{i}" for i in range(n)]})
    for feature in univariate.FEATURES:
        df[feature] = rng.normal(100, 10, n).round(2)
    df.loc[0, "amount_usd"] = 1000     # very high
    df.loc[1, "amount_usd"] = -500     # very low
    df["velocity_unique_countries"] = 12.0   # same value everywhere -> IQR = 0
    df["customers_per_device"] = 1.0         # mostly 1, like in 8A -> IQR = 0
    df.loc[2, "customers_per_device"] = 4.0
    # the *_log_z columns that Phase I (task 8) adds to 8A
    for feature, column in univariate.LOG_Z_COLUMNS.items():
        log_values = np.log1p(df[feature].clip(lower=0))
        df[column] = (log_values - log_values.mean()) / log_values.std(ddof=0)
    return df


class TestUnivariateOutlierDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.result, cls.report = univariate.detect(cls.data)

    def test_no_rows_removed(self):
        self.assertEqual(len(self.result), len(self.data))

    def test_added_outliers_are_found(self):
        for row in (0, 1):
            self.assertEqual(self.result.loc[row, "amount_usd_z_outlier"], 1)
            self.assertEqual(self.result.loc[row, "amount_usd_iqr_outlier"], 1)

    def test_zscore_formula(self):
        values = self.data["customer_txn_count"]
        z = (values - values.mean()) / values.std(ddof=0)
        expected = (z.abs() > 3).astype(int)
        self.assertEqual(list(self.result["customer_txn_count_z_outlier"]), list(expected))

    def test_iqr_formula(self):
        values = self.data["customer_txn_count"]
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        expected = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).astype(int)
        self.assertEqual(list(self.result["customer_txn_count_iqr_outlier"]), list(expected))

    def test_iqr_zero_is_not_used(self):
        self.assertEqual(self.result["velocity_unique_countries_iqr_outlier"].sum(), 0)

    def test_customers_per_device_included(self):
        self.assertIn("customers_per_device", univariate.FEATURES)
        self.assertEqual(self.result["customers_per_device_iqr_outlier"].sum(), 0)   # IQR = 0
        self.assertEqual(self.result.loc[2, "customers_per_device_z_outlier"], 1)

    def test_log_zscore_uses_phase_1_columns(self):
        for feature, column in univariate.LOG_Z_COLUMNS.items():
            expected = (self.data[column].abs() > 3).astype(int)
            self.assertEqual(list(self.result[feature + "_zlog_outlier"]), list(expected))
        self.assertNotIn("customer_txn_count_zlog_outlier", self.result.columns)

    def test_extreme_iqr_is_part_of_iqr(self):
        for feature in univariate.FEATURES:
            extreme = self.result[feature + "_iqr_extreme_outlier"]
            normal = self.result[feature + "_iqr_outlier"]
            self.assertTrue(((extreme == 1) <= (normal == 1)).all() or normal.sum() == 0)
        self.assertEqual(self.result.loc[0, "amount_usd_iqr_extreme_outlier"], 1)

    def test_summary_columns(self):
        primary = [f + s for f in univariate.FEATURES for s in ("_z_outlier", "_iqr_outlier")]
        expected = (self.result[primary].sum(axis=1) > 0).astype(int)
        self.assertEqual(list(self.result["is_univariate_outlier"]), list(expected))

        all_methods = [c for c in self.result.columns if c.endswith(
            ("_z_outlier", "_zlog_outlier", "_iqr_outlier", "_iqr_extreme_outlier"))
            and not c.startswith("is_")]
        self.assertEqual(len(all_methods), 3 * len(univariate.FEATURES) + len(univariate.LOG_Z_COLUMNS))
        expected_all = (self.result[all_methods].sum(axis=1) > 0).astype(int)
        self.assertEqual(list(self.result["is_univariate_outlier_all_methods"]), list(expected_all))
        # every primary outlier is also an outlier for all methods
        self.assertTrue((self.result["is_univariate_outlier"] <= self.result["is_univariate_outlier_all_methods"]).all())

    def test_report_has_all_methods(self):
        expected_rows = 3 * len(univariate.FEATURES) + len(univariate.LOG_Z_COLUMNS)
        self.assertEqual(len(self.report), expected_rows)
        self.assertEqual(set(self.report["method"]), {"Z-score", "Z-score (log)", "IQR", "IQR (extreme)"})

    def test_files_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "8A.csv"
            self.data.to_csv(path, index=False)
            univariate.main(path, folder)
            for name in ("1A_UNIVARIATE_OUTLIERS_DATASET.csv", "1B_UNIVARIATE_OUTLIER_REPORT.csv",
                         "1C_UNIVARIATE_OUTLIER_PLOTS.png"):
                self.assertTrue((Path(folder) / name).exists(), name)


if __name__ == "__main__":
    unittest.main()