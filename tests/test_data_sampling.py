import csv
import importlib.util
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "phase_1" / "5_1_data_sampling.py"
DATA = ROOT / "dataset" / "phase_1"

# The script name starts with a digit, so load it by path.
spec = importlib.util.spec_from_file_location("data_sampling", SCRIPT)
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def strata(rows):
    return Counter(sampling.stratum_of(row) for row in rows)


def make_dataset(path, n=5000, seed=0):
    """Small dataset with the same shape as 4F: fraud rate differs by country."""
    rng = random.Random(seed)
    fraud_rate = {"USA": 0.08, "UK": 0.08, "BRAZIL": 0.40, "NIGERIA": 0.37, "JAPAN": 0.07}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transaction_id", "country", "is_fraud", "amount_usd"])
        for i in range(n):
            country = rng.choice(list(fraud_rate))
            fraud = rng.random() < fraud_rate[country]
            writer.writerow([f"T{i % 4990}", country, str(fraud), round(rng.uniform(1, 5000), 2)])


class SamplingChecks:
    """Checks shared by the generated-data and real-data test cases."""

    def test_samples_have_expected_size(self):
        expected = round(len(self.population) * self.fraction)
        for method in ("simple_random", "stratified", "systematic"):
            self.assertEqual(len(self.samples[method]), expected, method)
        self.assertLessEqual(len(self.samples["balanced"]), expected)

    def test_no_duplicate_rows_and_all_rows_from_population(self):
        # Compare whole rows: transaction_id is not unique in the dataset
        # (a few different transactions share the same ID).
        population = Counter(tuple(row.values()) for row in self.population)
        for method, rows in self.samples.items():
            sample = Counter(tuple(row.values()) for row in rows)
            self.assertFalse(sample - population, f"{method} has duplicate or unknown rows")

    def test_columns_unchanged(self):
        for method, rows in self.samples.items():
            self.assertEqual(list(rows[0]), list(self.population[0]), method)

    def test_stratified_keeps_every_stratum_proportional(self):
        pop = strata(self.population)
        sample = strata(self.samples["stratified"])
        expected = sampling.allocate_proportionally(pop, len(self.samples["stratified"]))
        self.assertEqual(dict(sample), {k: v for k, v in expected.items() if v})

    def test_balanced_sample_is_half_fraud(self):
        labels = Counter(sampling.is_fraud(row) for row in self.samples["balanced"])
        self.assertEqual(labels[True], labels[False])

    def test_report_matches_samples(self):
        report = {row["method"]: row for row in read_csv(self.output_dir / sampling.REPORT_FILE)}
        self.assertEqual(int(report["population"]["rows"]), len(self.population))
        for method, rows in self.samples.items():
            self.assertEqual(int(report[method]["rows"]), len(rows))
            fraud = sum(sampling.is_fraud(row) for row in rows)
            self.assertEqual(int(report[method]["fraud_transactions"]), fraud)


class TestSamplingOnGeneratedData(SamplingChecks, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.tmp.name)
        cls.input = cls.output_dir / "input.csv"
        cls.fraction = 0.1
        make_dataset(cls.input)
        sampling.run_sampling(cls.input, cls.output_dir, cls.fraction, seed=7)
        cls.population = read_csv(cls.input)
        cls.samples = {
            method: read_csv(cls.output_dir / name)
            for method, name in sampling.OUTPUT_FILES.items()
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_seed_gives_same_sample(self):
        with tempfile.TemporaryDirectory() as other:
            sampling.run_sampling(self.input, Path(other), self.fraction, seed=7)
            for name in sampling.OUTPUT_FILES.values():
                self.assertEqual(
                    (self.output_dir / name).read_text(),
                    (Path(other) / name).read_text(),
                    name,
                )

    def test_allocation_adds_up_exactly(self):
        counts = {"a": 333, "b": 333, "c": 334}
        self.assertEqual(sum(sampling.allocate_proportionally(counts, 101).values()), 101)

    def test_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            sampling.run_sampling(self.input, self.output_dir, 0)


def real_outputs_available():
    try:
        first = (DATA / "4A_INTEGRATED_DATASET.csv").open(encoding="utf-8").readline()
    except FileNotFoundError:
        return False
    # Skip if the file is still a Git LFS pointer (run `git lfs pull`).
    return not first.startswith("version https://git-lfs") and all(
        (DATA / name).exists() for name in sampling.OUTPUT_FILES.values()
    )


@unittest.skipUnless(real_outputs_available(), "real dataset / sampling outputs not present")
class TestSamplingOnRealData(SamplingChecks, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = DATA
        cls.fraction = sampling.DEFAULT_FRACTION
        cls.population = read_csv(DATA / "4A_INTEGRATED_DATASET.csv")
        cls.samples = {
            method: read_csv(DATA / name)
            for method, name in sampling.OUTPUT_FILES.items()
        }


if __name__ == "__main__":
    unittest.main()