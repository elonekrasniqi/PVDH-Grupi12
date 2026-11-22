import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


univariate = load_module("univariate", ROOT / "src" / "phase_2" / "1_1_univariate_outlier_detection.py")
treatment = load_module("treatment", ROOT / "src" / "phase_2" / "3_1_outlier_treatment.py")


def make_data(n=400):
    """Small 8A-like data (customers with several transactions) + 1A flags + 2E flags."""
    rng = np.random.default_rng(7)
    customers = np.repeat([f"CUST_{i}" for i in range(n // 4)], 4)
    df = pd.DataFrame({
        "transaction_id": [f"TX_{i}" for i in range(n)],
        "customer_id": customers,
        "is_fraud": rng.integers(0, 2, n),
        "currency_resolved": rng.choice(["EUR", "USD", "NGN"], n),
        "amount_usd": np.exp(rng.normal(6, 1, n)).round(2),
        "velocity_num_transactions": rng.integers(200, 1200, n).astype(float),
        "customers_per_device": 1,
        "velocity_unique_countries": 12.0,
    })
    per_customer = pd.DataFrame({
        "customer_id": pd.unique(customers),
        "customer_avg_amount_usd": np.exp(rng.normal(6.5, 0.5, n // 4)).round(2),
        "customer_txn_count": rng.normal(90, 10, n // 4).round(),
        "customer_away_share": rng.normal(0.33, 0.05, n // 4).clip(0, 1),
        "customer_night_share": rng.normal(0.24, 0.04, n // 4).clip(0, 1),
    })
    df = df.merge(per_customer, on="customer_id", how="left")

    df.loc[0, "amount_usd"] = 0.01                 # very small (micro amount)
    df.loc[1, "amount_usd"] = 2_000_000            # very large
    df.loc[df.customer_id == "CUST_5", "customer_txn_count"] = 400   # one customer, 4 rows
    df.loc[2, "customers_per_device"] = 3          # valid shared device
    df.loc[3, "velocity_unique_countries"] = 2.0   # few countries = few transactions
    df.loc[3, "velocity_num_transactions"] = 1.0

    df["velocity_unique_merchants"] = np.minimum(105, df["velocity_num_transactions"] + 1)
    df["velocity_max_single_amount"] = df["amount_usd"] * 3
    df["velocity_total_amount"] = df["velocity_max_single_amount"] * (df["velocity_num_transactions"] / 2 + 1)

    for feature, column in univariate.LOG_Z_COLUMNS.items():
        log_values = np.log1p(df[feature])
        df[column] = (log_values - log_values.mean()) / log_values.std(ddof=0)

    flagged, _ = univariate.detect(df)
    flagged["isolation_forest_outlier"] = 0
    flagged["lof_outlier"] = 0
    flagged.loc[[0, 1, 10], "isolation_forest_outlier"] = 1
    flagged.loc[[1, 11], "lof_outlier"] = 1
    flagged["both_methods_outlier"] = flagged["isolation_forest_outlier"] & flagged["lof_outlier"]
    return flagged


class TestOutlierTreatment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.treated, cls.review, cls.report, cls.domain = treatment.run(cls.data)

    def test_no_rows_removed_and_originals_unchanged(self):
        self.assertEqual(len(self.treated), len(self.data))
        for feature in treatment.FEATURES:
            pd.testing.assert_series_equal(self.treated[feature], self.data[feature], check_dtype=False)

    def test_extreme_amounts_confirmed_and_capped(self):
        row = self.report.set_index("feature").loc["amount_usd"]
        for index in (0, 1):
            self.assertEqual(self.treated.loc[index, "amount_usd_outlier"], 1)
        self.assertAlmostEqual(self.treated.loc[0, "amount_usd_capped"], row["lower_bound"], places=3)
        self.assertAlmostEqual(self.treated.loc[1, "amount_usd_capped"], row["upper_bound"], places=3)

    def test_capping_only_changes_confirmed_outliers(self):
        for feature, rule in treatment.FEATURE_RULES.items():
            if rule["kind"] not in treatment.TREATED_KINDS:
                continue
            changed = self.treated[feature + "_capped"] != self.treated[feature]
            self.assertTrue((changed <= (self.treated[feature + "_outlier"] == 1)).all(), feature)

    def test_customer_level_feature_is_counted_once_per_customer(self):
        rows = self.treated.customer_id == "CUST_5"
        self.assertTrue((self.treated.loc[rows, "customer_txn_count_outlier"] == 1).all())
        report_row = self.report.set_index("feature").loc["customer_txn_count"]
        self.assertEqual(report_row["outlier_customers"], 1)
        self.assertEqual(report_row["high_outliers"], 4)

    def test_valid_rare_values_are_not_treated(self):
        for feature in ("customers_per_device", "velocity_unique_countries", "velocity_unique_merchants",
                        "velocity_total_amount"):
            self.assertNotIn(feature + "_capped", self.treated.columns)
        review = self.review.set_index(["feature", "method"])
        self.assertEqual(review.loc[("customers_per_device", "Z-score"), "verdict"], "rejected")

    def test_raw_scale_detection_on_skewed_money_is_rejected(self):
        review = self.review.set_index(["feature", "method"])
        for method in ("Z-score", "IQR", "IQR (extreme)"):
            self.assertEqual(review.loc[("amount_usd", method), "verdict"], "rejected")

    def test_every_detection_has_a_verdict_and_reason(self):
        self.assertFalse(self.review["verdict"].isna().any())
        self.assertTrue((self.review["reason"].str.len() > 0).all())
        self.assertEqual(set(self.review["verdict"]) - {"accepted", "corrected", "rejected", "supporting only"},
                         set())

    def test_summary_columns(self):
        flags = [f + "_outlier" for f, r in treatment.FEATURE_RULES.items() if r["kind"] in treatment.TREATED_KINDS]
        expected = (self.treated[flags].sum(axis=1) > 0).astype(int)
        self.assertEqual(list(self.treated["is_confirmed_outlier"]), list(expected))
        incorrect = (self.treated["is_univariate_outlier"] == 1) & (expected == 0)
        self.assertEqual(list(self.treated["is_incorrect_detection"]), list(incorrect.astype(int)))
        self.assertEqual(self.treated.loc[1, "outlier_confidence"], "high")      # univariate + both
        self.assertEqual(self.treated.loc[10, "outlier_confidence"], "low")      # one multivariate only

    def test_domain_errors_are_imputed(self):
        broken = self.data.copy()
        broken.loc[5, "customer_away_share"] = 1.7
        checked, report = treatment.domain_check(broken)
        self.assertEqual(checked.loc[5, "customer_away_share_domain_error"], 1)
        self.assertLessEqual(checked["customer_away_share"].max(), 1)
        self.assertEqual(int(self.domain["violations"].sum()), 0)

    def test_files_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            univariate_path = Path(folder) / "1A.csv"
            multivariate_path = Path(folder) / "2E.csv"
            self.data.to_csv(univariate_path, index=False)
            self.data[["transaction_id", "isolation_forest_outlier", "lof_outlier",
                       "both_methods_outlier"]].to_csv(multivariate_path, index=False)
            treatment.main(univariate_path, multivariate_path, folder)
            for name in ("3A_OUTLIER_TREATED_DATASET.csv", "3B_DETECTION_REVIEW_REPORT.csv",
                         "3C_OUTLIER_TREATMENT_REPORT.csv", "3D_OUTLIER_HANDLING_SUMMARY.csv",
                         "3E_OUTLIER_TREATMENT_PLOTS.png", "3F_DETECTION_REVIEW_PLOT.png",
                         "3G_INCORRECT_DETECTION_EVIDENCE.png", "3H_OUTLIER_HANDLING_SUMMARY_PLOT.png"):
                self.assertTrue((Path(folder) / name).exists(), name)

    def test_misaligned_inputs_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            univariate_path = Path(folder) / "1A.csv"
            multivariate_path = Path(folder) / "2E.csv"
            self.data.to_csv(univariate_path, index=False)
            shuffled = self.data.sample(frac=1, random_state=1)
            shuffled[["transaction_id", "isolation_forest_outlier", "lof_outlier",
                      "both_methods_outlier"]].to_csv(multivariate_path, index=False)
            with self.assertRaises(ValueError):
                treatment.load(univariate_path, multivariate_path)


if __name__ == "__main__":
    unittest.main()
