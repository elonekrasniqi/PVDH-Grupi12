import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = ROOT / "src" / "phase_1" / "6_1_feature_relationship_analysis.py"
SELECTION_SCRIPT = ROOT / "src" / "phase_1" / "6_2_feature_subset_selection.py"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("feature_relationship_analysis", ANALYSIS_SCRIPT)
selection = load_script("feature_subset_selection", SELECTION_SCRIPT)


class TestFeatureRelationshipAnalysis(unittest.TestCase):
    def test_cramers_v_detects_perfect_association(self):
        self.assertAlmostEqual(
            analysis.cramers_v({"non-fraud": [50, 0], "fraud": [0, 50]}),
            1.0,
        )

    def test_point_biserial_detects_perfect_association(self):
        self.assertAlmostEqual(
            analysis.point_biserial((2, 0.0, 0.0), (2, 2.0, 2.0)),
            1.0,
        )

    def test_high_cardinality_categorical_feature_is_not_recommended(self):
        result = analysis.build_feature_row(
            "merchant",
            "categorical",
            "cramers_v",
            0.2,
            1000,
            analysis.MAX_CATEGORICAL_LEVELS + 1,
        )
        self.assertEqual(result["recommendation"], "exclude_high_cardinality")

    def test_report_flags_perfect_fraud_categories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            input_path = directory / "sample.csv"
            report_path = directory / "report.csv"
            columns = (
                *analysis.NUMERIC_FEATURES,
                *analysis.CATEGORICAL_FEATURES,
                analysis.TARGET_COLUMN,
            )
            with input_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                for index in range(200):
                    fraud = index >= 100
                    row = {
                        "amount_usd": str(index),
                        "distance_from_home": "1" if fraud else "0",
                        "transaction_hour": "8" if fraud else "12",
                        "exchange_rate": "1",
                        "country": "Brazil" if fraud else "USA",
                        "currency_resolved": "BRL" if fraud else "USD",
                        "merchant_category": "Grocery",
                        "merchant_type": "online",
                        "merchant": "Shop",
                        "city": "Fraud City" if fraud else "Safe City",
                        "city_size": "large" if fraud else "medium",
                        "card_type": "credit",
                        "card_present": "True" if fraud else "False",
                        "device": "POS" if fraud else "Browser",
                        "channel": "pos" if fraud else "web",
                        "high_risk_merchant": "False",
                        "weekend_transaction": "False",
                        "currency_imputed": "0",
                        "is_fraud": str(fraud),
                    }
                    writer.writerow(row)

            rows_read, results = analysis.analyze(input_path, report_path)
            self.assertEqual(rows_read, 200)
            by_feature = {row["feature"]: row for row in results}
            self.assertEqual(
                by_feature["distance_from_home"]["recommendation"],
                "retain_candidate",
            )
            self.assertIn(
                "true",
                by_feature["card_present"]["potential_leakage"],
            )
            self.assertEqual(
                by_feature["card_present"]["recommendation"],
                "exclude_possible_target_leakage",
            )
            self.assertTrue(report_path.exists())

    def test_subset_uses_analysis_report_and_excludes_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            input_path = directory / "sample.csv"
            report_path = directory / "report.csv"
            output_path = directory / "selected.csv"
            report_columns = analysis.REPORT_COLUMNS
            with report_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=report_columns)
                writer.writeheader()
                for feature in selection.SELECTED_FEATURES:
                    writer.writerow(
                        {
                            "feature": feature,
                            "potential_leakage": "",
                            "recommendation": "retain_candidate",
                        }
                    )

            columns = (
                *selection.OUTPUT_COLUMNS,
                "channel",
                "device",
                "card_present",
                "transaction_id",
            )
            with input_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "distance_from_home": "1",
                        "transaction_hour": "8",
                        "country": "Brazil",
                        "city": "Unknown City",
                        "city_size": "medium",
                        "is_fraud": "True",
                        "channel": "pos",
                        "device": "NFC Payment",
                        "card_present": "True",
                        "transaction_id": "TX1",
                    }
                )

            rows_written, rows_excluded = selection.select_features(
                input_path, report_path, output_path
            )
            with output_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                rows = list(reader)

            self.assertEqual((rows_written, rows_excluded), (1, 0))
            self.assertEqual(
                reader.fieldnames,
                list(selection.OUTPUT_COLUMNS),
            )
            self.assertEqual(rows[0]["is_fraud"], "1")
            self.assertNotIn("channel", rows[0])
            self.assertNotIn("device", rows[0])
            self.assertNotIn("transaction_id", rows[0])

    def test_subset_rejects_feature_report_that_flags_selected_feature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.csv"
            with report_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("feature", "potential_leakage", "recommendation"),
                )
                writer.writeheader()
                for feature in selection.SELECTED_FEATURES:
                    writer.writerow(
                        {
                            "feature": feature,
                            "potential_leakage": (
                                "leaky" if feature == selection.SELECTED_FEATURES[0] else ""
                            ),
                            "recommendation": "retain_candidate",
                        }
                    )
            with self.assertRaisesRegex(ValueError, "potential target leakage"):
                selection.validate_analysis_report(report_path)


if __name__ == "__main__":
    unittest.main()
