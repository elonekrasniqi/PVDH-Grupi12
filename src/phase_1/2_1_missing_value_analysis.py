"""Missing-value identification and treatment strategy

"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "phase_1" / "1B_FORMAT_STANDARDIZED_DATASET.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "phase_1" / "2A_MISSING_VALUES_REPORT.csv"
HANDLER = Path(__file__).with_name("2_2_missing_value_handling.py")

REPORT_COLUMNS = (
    "column", "total_rows", "missing_count", "missing_pct",
    "empty_cells", "placeholder_values", "strategy", "reason",
)


def load_handler(path: Path = HANDLER):
    spec = importlib.util.spec_from_file_location("phase1_missing_value_handling", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_missing_values(input_path: Path, handler=None):
    """Returns (row_count, {column: {"empty": n, "placeholders": Counter}})."""
    handler = handler or load_handler()
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"Input dataset has no header: {input_path}")
        stats = {c: {"empty": 0, "placeholders": Counter()} for c in reader.fieldnames}
        rows = 0
        for row in reader:
            rows += 1
            for column, value in row.items():
                if column not in stats or not handler.is_missing_value(column, value):
                    continue
                text = (value or "").strip()
                if text == "":
                    stats[column]["empty"] += 1
                else:
                    stats[column]["placeholders"][text] += 1
    return rows, stats


def build_report(rows: int, stats: dict, handler=None) -> list[dict]:
    handler = handler or load_handler()
    report = []
    for column, item in stats.items():
        placeholders = item["placeholders"]
        missing = item["empty"] + sum(placeholders.values())
        if missing == 0:
            strategy, reason = "none needed", "No missing values."
        else:
            strategy, reason = handler.STRATEGIES.get(column, handler.NOT_IMPUTED)
        report.append({
            "column": column,
            "total_rows": rows,
            "missing_count": missing,
            "missing_pct": round(missing / rows * 100, 4) if rows else 0.0,
            "empty_cells": item["empty"],
            "placeholder_values": "; ".join(
                f"{value!r}={count}" for value, count in placeholders.most_common(5)
            ),
            "strategy": strategy,
            "reason": reason,
        })
    return report


def write_report(report: list[dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(report)


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    handler = load_handler()
    rows, stats = analyze_missing_values(input_path, handler)
    report = build_report(rows, stats, handler)
    write_report(report, output_path)
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Missing-value identification (writes 2A).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows, report = run(args.input, args.output)
    with_missing = [r for r in report if r["missing_count"]]
    print("MISSING VALUE ANALYSIS COMPLETED")
    print(f"Dataset: {args.input}")
    print(f"Rows scanned: {rows:,}")
    print(f"Missing values: {sum(r['missing_count'] for r in report):,} "
          f"in {len(with_missing)} of {len(report)} columns")
    for r in sorted(with_missing, key=lambda r: -r["missing_count"]):
        extra = f" (placeholders: {r['placeholder_values']})" if r["placeholder_values"] else ""
        print(f"  {r['column']:<20} {r['missing_count']:>8,} ({r['missing_pct']:.2f}%){extra}")
        print(f"  {'':<20} strategy: {r['strategy']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()