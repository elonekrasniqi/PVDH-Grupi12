import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "phase_1"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / "phase_1" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("feature_relationship_analysis", "9_1_feature_relationship_analysis.py")
selection = load("feature_subset_selection", "10_1_feature_subset_selection.py")


def make_frame(n=20000, seed=0):
    """A small 6D-like dataset with known relationships to is_fraud."""
    rng = np.random.default_rng(seed)
    fraud = (rng.random(n) < 0.25).astype(int)
    frame = pd.DataFrame({"transaction_id": [f"TX_{i}" for i in range(n)], "is_fraud": fraud})

    for column in analysis.NUMERIC_FEATURES:
        frame[column] = rng.normal(size=n)                       # noise by default
    frame["amount_usd_log_z"] = fraud * 1.5 + rng.normal(size=n)  # strong signal
    frame["transaction_amount_category_code_minmax"] = (          # ~ copy of amount -> redundant
        pd.qcut(frame["amount_usd_log_z"], 3, labels=False) / 2)
    frame["hour_cos"] = fraud * 0.8 + rng.normal(size=n)          # moderate signal

    for column in analysis.BINARY_FEATURES:
        frame[column] = rng.integers(0, 2, n)                    # noise
    frame["weekend_transaction"] = 0                             # constant
    frame["distance_from_home"] = np.where(rng.random(n) < 0.8, fraud, 1 - fraud)
    frame["card_present"] = np.where(fraud == 1, rng.integers(0, 2, n), 0)  # card_present=1 -> always fraud
    period = rng.integers(0, 4, n)
    period = np.where((fraud == 1) & (rng.random(n) < 0.6), 0, period)      # night -> more fraud
    for i, name in enumerate(["night", "morning", "afternoon", "evening"]):
        frame[f"time_period_{name}"] = (period == i).astype(int)

    countries = np.array(["Nigeria", "USA", "Germany", "Brazil"])
    risky = (fraud == 1) & (rng.random(n) < 0.7)                              # fraud leans to NG / BR
    country = np.where(risky, rng.choice(countries[[0, 3]], n), rng.choice(countries, n))
    frame["country"] = country
    frame["currency_resolved"] = pd.Series(country).map(                     # same as country -> redundant
        {"Nigeria": "NGN", "USA": "USD", "Germany": "EUR", "Brazil": "BRL"})
    frame["city"] = "Unknown City"
    frame["city_size"] = "medium"
    for column in ("merchant_category", "merchant_type", "card_type", "device", "channel"):
        frame[column] = rng.choice(["a", "b", "c"], n)
    frame["card_type"] = rng.choice(["Gold Credit", "GOLD CREDIT", "Basic Debit"], n)
    frame["merchant"] = [f"m{i % 80}" for i in range(n)]                     # high cardinality

    frame["customer_id"] = [f"C{i % 50}" for i in range(n)]                  # identifier
    frame["amount_usd"] = np.exp(frame["amount_usd_log_z"])                  # superseded raw column
    return frame


class TestAnalysisAndSelection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        cls.folder = folder
        cls.frame = make_frame()
        cls.input = folder / "6D.csv"
        cls.frame.to_csv(cls.input, index=False)
        cls.report_path = folder / "6F.csv"
        cls.rows, cls.report, cls.pairs = analysis.analyze(cls.input, cls.report_path, folder / "6F_pairs.csv")
        cls.by_feature = cls.report.set_index("feature")
        cls.subset, cls.summary = selection.select_features(
            cls.input, cls.report_path, folder / "6G.csv", folder / "6G_report.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def recommendation(self, feature):
        return self.by_feature.loc[feature, "recommendation"]

    # analysis
    def test_every_column_is_reported_once(self):
        expected = set(self.frame.columns) - {"is_fraud"}
        self.assertEqual(set(self.report["feature"]), expected)
        self.assertFalse(self.report["feature"].duplicated().any())

    def test_strong_features_retained(self):
        for feature in ("amount_usd_log_z", "hour_cos", "distance_from_home", "country", "time_period_night"):
            self.assertEqual(self.recommendation(feature), "retain", feature)

    def test_noise_excluded(self):
        for feature in ("velocity_unique_merchants_z", "high_risk_merchant", "merchant_category"):
            self.assertEqual(self.recommendation(feature), "exclude_low_association", feature)

    def test_leakage_constant_high_cardinality(self):
        self.assertEqual(self.recommendation("card_present"), "exclude_possible_target_leakage")
        self.assertEqual(self.by_feature.loc["card_present", "potential_leakage"], "1.0")
        self.assertEqual(self.recommendation("weekend_transaction"), "exclude_constant")
        self.assertEqual(self.recommendation("city"), "exclude_constant")
        self.assertEqual(self.recommendation("merchant"), "exclude_high_cardinality")

    def test_redundant_features_dropped_for_stronger_partner(self):
        self.assertEqual(self.recommendation("transaction_amount_category_code_minmax"), "exclude_redundant")
        self.assertEqual(self.by_feature.loc["transaction_amount_category_code_minmax", "redundant_with"],
                         "amount_usd_log_z")
        self.assertEqual(self.recommendation("currency_resolved"), "exclude_redundant")   # tie -> country kept
        self.assertEqual(self.by_feature.loc["currency_resolved", "redundant_with"], "country")

    def test_pairs_file(self):
        pair = self.pairs[(self.pairs["feature_a"] == "country") & (self.pairs["feature_b"] == "currency_resolved")]
        self.assertEqual(len(pair), 1)
        self.assertAlmostEqual(pair["relationship"].iloc[0], 1.0, places=4)
        self.assertTrue(pair["redundant"].iloc[0])
        self.assertTrue((self.pairs["relationship"].abs() >= analysis.PAIR_REPORT_THRESHOLD).all())

    def test_identifiers_and_superseded_columns(self):
        self.assertEqual(self.recommendation("transaction_id"), "exclude_identifier")
        self.assertEqual(self.recommendation("customer_id"), "exclude_identifier")
        self.assertEqual(self.recommendation("amount_usd"), "exclude_superseded")

    def test_card_type_spelling_is_merged(self):
        self.assertEqual(int(self.by_feature.loc["card_type", "levels"]), 2)   # 'gold credit', 'basic debit'

    def test_cramers_v(self):
        a = pd.Series(list("aabb") * 10)
        self.assertAlmostEqual(analysis.cramers_v(a, a), 1.0)
        self.assertAlmostEqual(analysis.cramers_v(a, pd.Series([0, 1, 0, 1] * 10)), 0.0)

    def test_missing_column_rejected(self):
        path = self.folder / "bad.csv"
        self.frame.drop(columns=["hour_cos"]).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            analysis.analyze(path, self.folder / "bad_r.csv", self.folder / "bad_p.csv")

    # selection
    def test_subset_has_only_retained_features(self):
        retained = set(self.report.loc[self.report["recommendation"] == "retain", "feature"])
        self.assertEqual(set(self.summary["source_feature"]), retained)
        self.assertEqual(self.subset.columns[0], "transaction_id")
        self.assertEqual(self.subset.columns[-1], "is_fraud")
        for excluded in ("card_present", "currency_resolved", "merchant", "transaction_amount_category_code_minmax"):
            self.assertFalse(any(c.startswith(excluded) for c in self.subset.columns), excluded)

    def test_categorical_one_hot_with_reference_level(self):
        country_columns = [c for c in self.subset.columns if c.startswith("country_")]
        self.assertEqual(len(country_columns), 3)                       # 4 countries - 1 reference
        reference = self.frame["country"].str.casefold().value_counts().index[0]
        self.assertNotIn(f"country_{reference}", country_columns)
        is_reference = self.frame["country"].str.casefold() == reference
        self.assertTrue((self.subset.loc[is_reference, country_columns].sum(axis=1) == 0).all())
        self.assertTrue((self.subset.loc[~is_reference, country_columns].sum(axis=1) == 1).all())

    def test_numeric_values_copied_unchanged(self):
        self.assertTrue(np.allclose(self.subset["amount_usd_log_z"], self.frame["amount_usd_log_z"]))

    def test_rows_and_target_kept(self):
        self.assertEqual(len(self.subset), len(self.frame))
        self.assertEqual(list(self.subset["is_fraud"]), list(self.frame["is_fraud"]))
        self.assertFalse(self.subset.isna().any().any())

    def test_missing_report_rejected(self):
        with self.assertRaises(ValueError):
            selection.select_features(self.input, self.folder / "nope.csv", self.folder / "o.csv", self.folder / "s.csv")

    def test_leaking_retained_feature_rejected(self):
        report = pd.read_csv(self.report_path, dtype=str, keep_default_na=False)
        report.loc[report["feature"] == "card_present", "recommendation"] = "retain"
        path = self.folder / "bad_report.csv"
        report.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            selection.select_features(self.input, path, self.folder / "o.csv", self.folder / "s.csv")


def real_outputs_available():
    paths = [DATA / n for n in ("8A_TRANSFORMED_NORMALIZED_DATASET.csv", "9A_FEATURE_RELATIONSHIP_REPORT.csv",
                                "10A_FEATURE_SUBSET_DATASET.csv")]
    if not all(p.exists() for p in paths):
        return False
    with paths[0].open(encoding="utf-8") as f:
        return not f.readline().startswith("version https://git-lfs")


@unittest.skipUnless(real_outputs_available(), "real dataset / outputs not present")
class TestRealOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = pd.read_csv(DATA / "8A_TRANSFORMED_NORMALIZED_DATASET.csv", usecols=["transaction_id", "is_fraud"])
        cls.report = pd.read_csv(DATA / "9A_FEATURE_RELATIONSHIP_REPORT.csv", keep_default_na=False)
        cls.subset = pd.read_csv(DATA / "10A_FEATURE_SUBSET_DATASET.csv")

    def test_rows_and_fraud_kept(self):
        self.assertEqual(len(self.subset), len(self.source))
        self.assertEqual(int(self.subset["is_fraud"].sum()), int(self.source["is_fraud"].sum()))

    def test_leakage_features_not_selected(self):
        for feature in ("card_present", "device", "channel"):
            self.assertEqual(self.report.set_index("feature").loc[feature, "recommendation"],
                             "exclude_possible_target_leakage")
            self.assertFalse(any(c.startswith(feature) for c in self.subset.columns))

    def test_no_missing_values(self):
        self.assertFalse(self.subset.isna().any().any())


if __name__ == "__main__":
    unittest.main()
