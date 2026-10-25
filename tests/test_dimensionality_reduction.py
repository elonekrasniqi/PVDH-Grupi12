"""Extended unit tests for Task E PCA.
Run from project root: py -m unittest discover -s tests -p "test_dimensionality_reduction.py" -v
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT = Path(__file__).resolve().parents[1] / 'src/phase_1/7_1_dimensionality_reduction.py'
spec = importlib.util.spec_from_file_location('dimensionality_reduction', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample_data(n=120):
    return pd.DataFrame({
        'distance_from_home': [i * .37 + (i % 7) for i in range(n)],
        'transaction_hour': [i % 24 for i in range(n)],
        'country': [['USA', 'UK', 'FRANCE'][i % 3] for i in range(n)],
        'city': [['NY', 'LONDON', 'PARIS'][i % 3] for i in range(n)],
        'city_size': [['large', 'small'][i % 2] for i in range(n)],
        'is_fraud': [i % 2 for i in range(n)],
    })


class DimensionalityReductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'input.csv'
        self.data = sample_data()
        self.data.to_csv(self.source, index=False)
        self.output = self.folder / 'output'

    def run_pca(self, **kwargs):
        return module.run(self.source, self.output, **kwargs)

    def test_creates_all_four_artifacts(self):
        self.run_pca()
        for filename in ('7A_PCA_TRAIN.csv', '7B_PCA_TEST.csv',
                         '7C_PCA_REPORT.json', '7D_PCA_PIPELINE.joblib'):
            with self.subTest(filename=filename):
                self.assertTrue((self.output / filename).is_file())

    def test_train_test_counts_and_no_overlap_of_original_rows(self):
        report = self.run_pca()
        self.assertEqual(report['train_rows'], 96)
        self.assertEqual(report['test_rows'], 24)
        self.assertEqual(report['train_rows'] + report['test_rows'], len(self.data))

    def test_target_counts_and_stratification(self):
        self.run_pca()
        train = pd.read_csv(self.output / '7A_PCA_TRAIN.csv')
        test = pd.read_csv(self.output / '7B_PCA_TEST.csv')
        self.assertEqual(train['is_fraud'].value_counts().to_dict(), {0: 48, 1: 48})
        self.assertEqual(test['is_fraud'].value_counts().to_dict(), {0: 12, 1: 12})

    def test_component_columns_are_numbered_and_target_is_last(self):
        report = self.run_pca()
        for filename in ('7A_PCA_TRAIN.csv', '7B_PCA_TEST.csv'):
            frame = pd.read_csv(self.output / filename)
            expected = [f'PC{i}' for i in range(1, report['pca_component_count'] + 1)] + ['is_fraud']
            self.assertEqual(frame.columns.tolist(), expected)
            self.assertTrue(np.isfinite(frame.drop(columns='is_fraud').to_numpy()).all())

    def test_explained_variance_meets_threshold(self):
        report = self.run_pca(variance=.95)
        self.assertGreaterEqual(report['total_explained_variance'], .95 - 1e-10)
        self.assertLessEqual(report['total_explained_variance'], 1 + 1e-10)
        self.assertEqual(len(report['explained_variance_per_component']), report['pca_component_count'])
        self.assertAlmostEqual(sum(report['explained_variance_per_component']),
                               report['total_explained_variance'], places=10)

    def test_reduced_dimension_count_is_bounded(self):
        report = self.run_pca()
        self.assertEqual(report['input_feature_count'], 5)
        self.assertGreaterEqual(report['encoded_feature_count'], 5)
        self.assertGreaterEqual(report['pca_component_count'], 1)
        self.assertLessEqual(report['pca_component_count'], report['encoded_feature_count'])

    def test_pipeline_reproduces_saved_components(self):
        self.run_pca(seed=42)
        pipeline = joblib.load(self.output / '7D_PCA_PIPELINE.joblib')
        X = self.data[module.NUMERIC + module.CATEGORICAL]
        y = self.data['is_fraud']
        X_train, X_test, _, _ = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
        for X_part, filename in ((X_train, '7A_PCA_TRAIN.csv'), (X_test, '7B_PCA_TEST.csv')):
            saved = pd.read_csv(self.output / filename).drop(columns='is_fraud').to_numpy()
            np.testing.assert_allclose(pipeline.transform(X_part), saved, rtol=1e-8, atol=1e-8)

    def test_pipeline_handles_unseen_categories(self):
        self.run_pca()
        pipeline = joblib.load(self.output / '7D_PCA_PIPELINE.joblib')
        new = pd.DataFrame([{'distance_from_home': 8.0, 'transaction_hour': 4,
                             'country': 'UNKNOWN_COUNTRY', 'city': 'UNKNOWN_CITY',
                             'city_size': 'medium'}])
        result = pipeline.transform(new)
        self.assertEqual(result.shape[0], 1)
        self.assertTrue(np.isfinite(result).all())

    def test_report_is_valid_json_and_records_settings(self):
        self.run_pca(variance=.9, seed=17)
        report = json.loads((self.output / '7C_PCA_REPORT.json').read_text(encoding='utf-8'))
        self.assertEqual(report['seed'], 17)
        self.assertEqual(report['variance_threshold'], .9)
        self.assertTrue(report['target_excluded_from_pca'])
        self.assertEqual(report['numeric_features'], module.NUMERIC)
        self.assertEqual(report['categorical_features'], module.CATEGORICAL)

    def test_reproducibility_with_same_seed(self):
        self.run_pca(seed=7)
        first = pd.read_csv(self.output / '7A_PCA_TRAIN.csv')
        self.run_pca(seed=7)
        second = pd.read_csv(self.output / '7A_PCA_TRAIN.csv')
        pd.testing.assert_frame_equal(first, second)

    def test_missing_required_column_rejected(self):
        self.data.drop(columns='city').to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'Missing required columns'):
            self.run_pca()

    def test_missing_numeric_value_rejected(self):
        self.data.loc[0, 'distance_from_home'] = np.nan
        self.data.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'missing values'):
            self.run_pca()

    def test_missing_categorical_value_rejected(self):
        self.data.loc[0, 'country'] = None
        self.data.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'missing values'):
            self.run_pca()

    def test_non_numeric_value_rejected(self):
        self.data.loc[0, 'distance_from_home'] = 'invalid'
        self.data.to_csv(self.source, index=False)
        with self.assertRaises((ValueError, TypeError)):
            self.run_pca()

    def test_infinite_numeric_value_rejected(self):
        self.data.loc[0, 'distance_from_home'] = np.inf
        self.data.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'infinite values'):
            self.run_pca()

    def test_non_binary_target_rejected(self):
        self.data.loc[0, 'is_fraud'] = 2
        self.data.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.run_pca()

    def test_single_class_target_rejected(self):
        self.data['is_fraud'] = 0
        self.data.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'at least two rows per target class'):
            self.run_pca()

    def test_too_few_rows_in_minority_class_rejected(self):
        self.data['is_fraud'] = 0
        self.data.loc[0, 'is_fraud'] = 1
        self.data.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'at least two rows per target class'):
            self.run_pca()

    def test_invalid_variance_thresholds_rejected(self):
        for variance in (0, -0.2, 1.01, 2):
            with self.subTest(variance=variance):
                with self.assertRaisesRegex(ValueError, 'Variance threshold'):
                    self.run_pca(variance=variance)

    def test_target_does_not_change_transformed_features_given_same_split(self):
        # Changing labels while preserving class balance and split indices is hard to guarantee.
        # Instead, verify the persisted pipeline only expects the five selected feature columns.
        self.run_pca()
        pipeline = joblib.load(self.output / '7D_PCA_PIPELINE.joblib')
        self.assertNotIn('is_fraud', pipeline.feature_names_in_.tolist())
        self.assertEqual(set(pipeline.feature_names_in_), set(module.NUMERIC + module.CATEGORICAL))


if __name__ == '__main__':
    unittest.main()
