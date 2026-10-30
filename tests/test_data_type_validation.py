
"""Unit tests for Phase I data type validation."""

import csv
import importlib.util
import tempfile
import unittest

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = (
    PROJECT_ROOT
    / "src"
    / "phase_1"
    / "1_3_data_type_validation.py"
)

spec = importlib.util.spec_from_file_location(
    "data_type_validation",
    SCRIPT_PATH
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DataTypeValidationTests(unittest.TestCase):

    def test_valid_values(self):
        """Check valid data types."""

        self.assertEqual(
            module.validate_value(
                "amount", "604.8", "float"
            ),
            "valid"
        )

        self.assertEqual(
            module.validate_value(
                "transaction_hour", "15", "integer_0_23"
            ),
            "valid"
        )

        self.assertEqual(
            module.validate_value(
                "card_present", "True", "boolean"
            ),
            "valid"
        )

        self.assertEqual(
            module.validate_value(
                "timestamp",
                "2024-09-30 15:18:24+00:00",
                "datetime"
            ),
            "valid"
        )

        self.assertEqual(
            module.validate_value(
                "customer_id", "CUST_12345", "identifier"
            ),
            "valid"
        )

    def test_invalid_values(self):
        """Check invalid numeric, boolean and identifier values."""

        self.assertEqual(
            module.validate_value(
                "amount", "nan", "float"
            ),
            "invalid"
        )

        self.assertEqual(
            module.validate_value(
                "amount", "inf", "float"
            ),
            "invalid"
        )

        self.assertEqual(
            module.validate_value(
                "transaction_hour", "24", "integer_0_23"
            ),
            "invalid"
        )

        self.assertEqual(
            module.validate_value(
                "transaction_hour", "15.5", "integer_0_23"
            ),
            "invalid"
        )

        self.assertEqual(
            module.validate_value(
                "card_present", "maybe", "boolean"
            ),
            "invalid"
        )

        self.assertEqual(
            module.validate_value(
                "transaction_id", "c", "identifier"
            ),
            "invalid"
        )

    def test_format_issues(self):
        """Check valid types that require standardization."""

        cases = [
            ("transaction_hour", "15.0", "integer_0_23"),
            ("card_present", "yes", "boolean"),
            ("card_present", "no", "boolean"),
            ("customer_id", "59395", "identifier"),
            ("customer_id", "cust-90462", "identifier"),
            ("timestamp", "30/09/2024 08:27:01", "datetime"),
        ]

        for column, value, expected_type in cases:
            with self.subTest(column=column, value=value):
                self.assertEqual(
                    module.validate_value(
                        column, value, expected_type
                    ),
                    "format_issue"
                )

    def test_missing_values(self):
        """Check missing value detection."""

        for value in ["", "NULL", "none", "NaN", "N/A", None]:
            with self.subTest(value=value):
                self.assertTrue(module.is_missing(value))

        self.assertFalse(module.is_missing("0"))
        self.assertFalse(module.is_missing("False"))

    def test_report_counts_and_unknown_column(self):
        """Check profiling counts and generated CSV report."""

        with tempfile.TemporaryDirectory() as directory:

            source = Path(directory) / "input.csv"
            target = Path(directory) / "report.csv"

            with source.open(
                "w",
                encoding="utf-8",
                newline=""
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "amount",
                        "transaction_hour",
                        "card_present",
                        "customer_id",
                        "extra_column",
                    ]
                )

                writer.writeheader()

                writer.writerows([
                    {
                        "amount": "100.5",
                        "transaction_hour": "15.0",
                        "card_present": "yes",
                        "customer_id": "CUST_123",
                        "extra_column": "abc",
                    },
                    {
                        "amount": "nan",
                        "transaction_hour": "24",
                        "card_present": "maybe",
                        "customer_id": "cust-90462",
                        "extra_column": "xyz",
                    },
                    {
                        "amount": "",
                        "transaction_hour": "10",
                        "card_present": "False",
                        "customer_id": "",
                        "extra_column": "",
                    },
                ])

            stats = module.profile_dataset(source)
            module.write_report(stats, target)

            self.assertTrue(target.exists())

            self.assertEqual(
                stats["amount"]["total_rows"], 3
            )

            self.assertEqual(
                stats["amount"]["missing_count"], 2
            )

            self.assertEqual(
                stats["amount"]["invalid_count"], 0
            )

            self.assertEqual(
                stats["transaction_hour"]["format_issue_count"],
                1
            )

            self.assertEqual(
                stats["card_present"]["invalid_count"], 1
            )

            self.assertEqual(
                stats["customer_id"]["format_issue_count"], 1
            )

            self.assertEqual(
                stats["extra_column"]["unclassified_count"], 2
            )

            with target.open(
                "r",
                encoding="utf-8",
                newline=""
            ) as file:
                report = list(csv.DictReader(file))

            self.assertEqual(len(report), 5)


if __name__ == "__main__":
    unittest.main()
