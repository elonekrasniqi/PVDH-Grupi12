import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "src/phase_1/12_1_adasyn_oversampling.py"
spec = importlib.util.spec_from_file_location("adasyn_oversampling", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(1.5, 1, (20, 2))])
    return pd.DataFrame({"PC1": X[:, 0], "PC2": X[:, 1], "is_fraud": [0] * 100 + [1] * 20})


class AdasynTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "11A_PCA_TRAIN.csv"
        self.output = self.folder / "12A.csv"
        self.report = self.folder / "12B.csv"
        self.data = sample_data()
        self.data.to_csv(self.source, index=False)

    def run_adasyn(self, **kwargs):
        return module.run(self.source, self.output, self.report, **kwargs)

    def test_default_balances_classes(self):
        result, report = self.run_adasyn()
        self.assertEqual(result["is_fraud"].value_counts().to_dict(), {0: 100, 1: 100})
        self.assertEqual(report["synthetic_rows"].tolist(), [0, 80])
        self.assertTrue(self.output.is_file() and self.report.is_file())

    def test_custom_ratio(self):
        result, _ = self.run_adasyn(ratio=0.5)
        self.assertEqual(int(result["is_fraud"].sum()), 50)

    def test_original_rows_unchanged(self):
        result, _ = self.run_adasyn()
        pd.testing.assert_frame_equal(result.iloc[:120], pd.read_csv(self.source))
        pd.testing.assert_frame_equal(pd.read_csv(self.source), self.data)

    def test_synthetic_rows_stay_inside_minority_range(self):
        result, _ = self.run_adasyn()
        original, new = self.data[self.data.is_fraud == 1], result.iloc[120:]
        for column in ("PC1", "PC2"):
            self.assertGreaterEqual(new[column].min(), original[column].min() - 1e-9)
            self.assertLessEqual(new[column].max(), original[column].max() + 1e-9)

    def test_harder_rows_get_more_synthetic_rows(self):
        X, y = self.data[["PC1", "PC2"]].to_numpy(), self.data["is_fraud"].to_numpy()
        _, _, r, g = module.adasyn(X, y)
        self.assertEqual(g.sum(), 80)
        self.assertTrue((g[r == 0] == 0).all())
        self.assertGreater(g[r == r.max()].mean(), g[r == r.min()].mean())

    def test_same_seed_same_output(self):
        first, _ = self.run_adasyn()
        second, _ = self.run_adasyn()
        pd.testing.assert_frame_equal(first, second)

    def test_rejects_test_set_and_bad_parameters(self):
        with self.assertRaises(ValueError):
            module.run(self.folder / "11B_PCA_TEST.csv", self.output, self.report)
        for kwargs in ({"ratio": 0}, {"ratio": 1.5}, {"ratio": 0.1}, {"k": 20}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.run_adasyn(**kwargs)


if __name__ == "__main__":
    unittest.main()
