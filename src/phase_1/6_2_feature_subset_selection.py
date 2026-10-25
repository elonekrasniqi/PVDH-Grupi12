from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "dataset" / "phase_1" / "5B_STRATIFIED_SAMPLE.csv"
DEFAULT_REPORT = ROOT / "dataset" / "phase_1" / "6A_FEATURE_RELATIONSHIP_REPORT.csv"
DEFAULT_OUTPUT = ROOT / "dataset" / "phase_1" / "6B_FEATURE_SUBSET_DATASET.csv"

SELECTED_FEATURES = (
    "distance_from_home",
    "transaction_hour",
    "country",
    "city",
    "city_size",
)
TARGET_COLUMN = "is_fraud"
OUTPUT_COLUMNS = (*SELECTED_FEATURES, TARGET_COLUMN)
VALID_TARGETS = {"true": "1", "false": "0", "1": "1", "0": "0"}
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}


def validate_analysis_report(report_path: Path) -> None:
    if not report_path.is_file():
        raise ValueError(
            f"Feature relationship report not found: {report_path}. "
            "Run 6_1_feature_relationship_analysis.py first."
        )

    with report_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required_columns = {
            "feature",
            "potential_leakage",
            "recommendation",
        }
        if reader.fieldnames is None or required_columns.difference(reader.fieldnames):
            raise ValueError(f"Invalid feature relationship report: {report_path}")
        results = {row["feature"]: row for row in reader}

    for feature in SELECTED_FEATURES:
        result = results.get(feature)
        if result is None:
            raise ValueError(
                f"Selected feature {feature!r} is absent from {report_path}."
            )
        if result["recommendation"] != "retain_candidate":
            raise ValueError(
                f"Selected feature {feature!r} is not recommended by "
                f"the relationship report: {result['recommendation']}."
            )
        if result["potential_leakage"]:
            raise ValueError(
                f"Selected feature {feature!r} is flagged for potential target leakage."
            )


def select_features(
    input_path: Path,
    report_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must not overwrite the input dataset.")
    validate_analysis_report(report_path)

    rows_written = 0
    rows_excluded = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open("r", newline="", encoding="utf-8-sig") as source,
        output_path.open("w", newline="", encoding="utf-8") as destination,
    ):
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Input dataset has no header: {input_path}")
        required = {*OUTPUT_COLUMNS}
        missing_columns = required.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Input dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row_number, row in enumerate(reader, start=2):
            target = (row.get(TARGET_COLUMN) or "").strip().casefold()
            normalized_target = VALID_TARGETS.get(target)
            if normalized_target is None:
                rows_excluded += 1
                continue

            selected = {column: row[column] for column in SELECTED_FEATURES}
            missing_features = [
                column
                for column, value in selected.items()
                if (value or "").strip().casefold() in NULL_TOKENS
            ]
            if missing_features:
                raise ValueError(
                    f"Selected feature values are missing at row {row_number}: "
                    + ", ".join(missing_features)
                )
            selected[TARGET_COLUMN] = normalized_target
            writer.writerow(selected)
            rows_written += 1

    if rows_written == 0:
        raise ValueError(f"No valid rows were selected from {input_path}.")
    return rows_written, rows_excluded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the selected feature subset from the stratified sample."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows_written, rows_excluded = select_features(
        args.input, args.report, args.output
    )
    print(f"Rows written: {rows_written:,}")
    print(f"Rows excluded for missing/invalid target: {rows_excluded:,}")
    print(f"Input: {args.input}")
    print(f"Relationship report: {args.report}")
    print(f"Output: {args.output}")
    print(f"Selected features: {', '.join(SELECTED_FEATURES)}")
    print("Target: is_fraud")
    print(
        "Excluded: identifiers, weak/redundant and high-cardinality features, "
        "and card_present, device, and channel because of perfect fraud-category "
        "associations."
    )


if __name__ == "__main__":
    main()
