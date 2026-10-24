"""
Data sampling: creates 10% samples of 4F_INTEGRATED_DATASET.csv
(simple random, stratified by is_fraud x country, systematic, balanced)
and a report comparing each sample with the full dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = ROOT / "dataset/phase_1/4F_INTEGRATED_DATASET.csv"
DEFAULT_OUTPUT_DIR = ROOT / "dataset/phase_1"

DEFAULT_FRACTION = 0.10
DEFAULT_SEED = 42
STRATA_COLUMNS = ("is_fraud", "country")

OUTPUT_FILES = {
    "simple_random": "5A_SIMPLE_RANDOM_SAMPLE.csv",
    "stratified": "5B_STRATIFIED_SAMPLE.csv",
    "systematic": "5C_SYSTEMATIC_SAMPLE.csv",
    "balanced": "5D_BALANCED_SAMPLE.csv",
}
REPORT_FILE = "5E_SAMPLING_REPORT.csv"


def is_fraud(row) -> bool:
    return row["is_fraud"].strip() == "True"


def stratum_of(row) -> tuple:
    return (
        is_fraud(row),
        row["country"].strip().upper(),
    )


def allocate_proportionally(counts: dict, total_size: int) -> dict:
    """Proportional allocation; largest-remainder rounding keeps the exact total."""
    population = sum(counts.values())
    quotas = {key: count * total_size / population for key, count in counts.items()}
    allocation = {key: int(quota) for key, quota in quotas.items()}

    remaining = total_size - sum(allocation.values())
    by_remainder = sorted(
        counts, key=lambda key: (quotas[key] - allocation[key], key), reverse=True
    )
    for key in by_remainder[:remaining]:
        allocation[key] += 1

    return allocation


def profile_population(input_path: Path):
    total = 0
    strata = Counter()
    classes = Counter()
    population = _empty_stats()

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        for row in reader:
            total += 1
            key = stratum_of(row)
            strata[key] += 1
            classes[key[0]] += 1
            _add_to_stats(population, row, key)

    return fieldnames, total, strata, classes, population


def plan_simple_random(total, sample_size, rng):
    return set(rng.sample(range(total), sample_size))


def plan_systematic(total, sample_size, rng):
    step = total / sample_size
    start = rng.random() * step
    return {int(start + i * step) for i in range(sample_size)}


def plan_stratified(strata, sample_size, rng):
    allocation = allocate_proportionally(strata, sample_size)
    return {
        key: set(rng.sample(range(strata[key]), allocation[key]))
        for key in sorted(strata)
    }


def plan_balanced(classes, sample_size, rng):
    per_class = min(min(classes.values()), sample_size // len(classes))
    return {
        key: set(rng.sample(range(classes[key]), per_class))
        for key in sorted(classes)
    }


def write_samples(input_path, output_dir, fieldnames, plans):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    writers = {}
    for method, name in OUTPUT_FILES.items():
        files[method] = (output_dir / name).open("w", encoding="utf-8", newline="")
        writers[method] = csv.DictWriter(files[method], fieldnames=fieldnames)
        writers[method].writeheader()

    stratum_position = Counter()
    class_position = Counter()
    stats = {method: _empty_stats() for method in OUTPUT_FILES}

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            for index, row in enumerate(csv.DictReader(file)):
                key = stratum_of(row)
                label = key[0]

                selected = {
                    "simple_random": index in plans["simple_random"],
                    "systematic": index in plans["systematic"],
                    "stratified": stratum_position[key] in plans["stratified"][key],
                    "balanced": class_position[label] in plans["balanced"][label],
                }
                stratum_position[key] += 1
                class_position[label] += 1

                for method, keep in selected.items():
                    if keep:
                        writers[method].writerow(row)
                        _add_to_stats(stats[method], row, key)
    finally:
        for handle in files.values():
            handle.close()

    return stats


def _empty_stats():
    return {"rows": 0, "fraud": 0, "amount_sum": 0.0, "amount_rows": 0,
            "countries": Counter()}


def _add_to_stats(stats, row, key):
    stats["rows"] += 1
    stats["fraud"] += key[0]
    stats["countries"][key[1]] += 1
    amount = row.get("amount_usd", "").strip()
    if amount:
        stats["amount_sum"] += float(amount)
        stats["amount_rows"] += 1


def _r(value):
    return round(value, 2) + 0.0


def summarize(method, stats, population):
    rows = stats["rows"]
    fraud_rate = stats["fraud"] / rows * 100
    mean_amount = stats["amount_sum"] / stats["amount_rows"] if stats["amount_rows"] else 0.0

    pop_rate = population["fraud"] / population["rows"] * 100
    pop_mean = population["amount_sum"] / population["amount_rows"]

    max_country_gap = max(
        abs(stats["countries"][c] / rows - population["countries"][c] / population["rows"]) * 100
        for c in population["countries"]
    )

    return {
        "method": method,
        "rows": rows,
        "sample_fraction_pct": _r(rows / population["rows"] * 100),
        "fraud_transactions": stats["fraud"],
        "fraud_rate_pct": _r(fraud_rate),
        "fraud_rate_diff_pp": _r(fraud_rate - pop_rate),
        "mean_amount_usd": _r(mean_amount),
        "mean_amount_diff_pct": _r((mean_amount - pop_mean) / pop_mean * 100),
        "max_country_share_diff_pp": _r(max_country_gap),
    }


def write_report(output_dir, population, stats):
    rows = [summarize("population", population, population)]
    rows += [summarize(method, stats[method], population) for method in OUTPUT_FILES]

    path = output_dir / REPORT_FILE
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def run_sampling(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT_DIR,
                 fraction=DEFAULT_FRACTION, seed=DEFAULT_SEED):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    fieldnames, total, strata, classes, population = profile_population(input_path)
    if total == 0:
        raise ValueError(f"{input_path.name} has no rows")

    sample_size = max(1, round(total * fraction))

    plans = {
        "simple_random": plan_simple_random(total, sample_size, random.Random(seed)),
        "systematic": plan_systematic(total, sample_size, random.Random(seed + 1)),
        "stratified": plan_stratified(strata, sample_size, random.Random(seed + 2)),
        "balanced": plan_balanced(classes, sample_size, random.Random(seed + 3)),
    }

    stats = write_samples(input_path, output_dir, fieldnames, plans)
    report = write_report(output_dir, population, stats)

    print("DATA SAMPLING COMPLETED")
    print(f"Population: {total:,} rows | {len(strata)} strata ({' x '.join(STRATA_COLUMNS)})")
    print(f"Fraction: {fraction:.0%} | Seed: {seed}\n")
    header = f"{'method':<14}{'rows':>9}{'fraud %':>9}{'diff pp':>9}{'mean USD':>11}{'country pp':>12}"
    print(header)
    print("-" * len(header))
    for row in report:
        print(f"{row['method']:<14}{row['rows']:>9,}{row['fraud_rate_pct']:>9}"
              f"{row['fraud_rate_diff_pp']:>9}{row['mean_amount_usd']:>11}"
              f"{row['max_country_share_diff_pp']:>12}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Phase I - data sampling")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION,
                        help="sample size as a fraction of the dataset (default 0.10)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_sampling(args.input, args.output_dir, args.fraction, args.seed)


if __name__ == "__main__":
    main()