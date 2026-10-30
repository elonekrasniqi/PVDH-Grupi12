"""Tests for Phase I data quality comparison."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'src' / 'phase_1' / '1_4_data_quality_comparison.py'
spec = importlib.util.spec_from_file_location('quality_comparison', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def stats(missing=0, valid=2, invalid=1, fmt=1, total=4):
    return {'expected_type': 'boolean', 'total_rows': total,
            'non_missing': total - missing, 'missing_count': missing,
            'valid_count': valid, 'invalid_count': invalid,
            'format_issue_count': fmt, 'unclassified_count': 0}


class QualityComparisonTests(unittest.TestCase):
    def test_comparison_and_csv(self):
        before = {'card_present': stats()}
        after = {'card_present': stats(valid=3, invalid=0, fmt=1)}
        rows = module.compare_stats(before, after)
        self.assertEqual(rows[0]['change_invalid_count'], -1)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'report.csv'
            module.save_report(rows, target)
            with target.open(newline='', encoding='utf-8') as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(saved[0]['after_invalid_count'], '0')

    def test_reject_different_columns(self):
        with self.assertRaises(ValueError):
            module.compare_stats({'a': stats()}, {'b': stats()})

    def test_reject_different_row_counts(self):
        with self.assertRaises(ValueError):
            module.compare_stats({'a': stats()}, {'a': stats(total=5)})

    def test_reject_empty_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                module.save_report([], Path(directory) / 'out.csv')


if __name__ == '__main__':
    unittest.main()
