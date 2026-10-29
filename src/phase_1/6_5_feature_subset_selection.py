"""
Feature subset selection on the transformed & normalized dataset (6D), driven
by the feature relationship report (6F).

  - keeps exactly the features the report marks "retain" (no hard-coded list);
  - numeric / binary features are copied as they are (already normalized in 6D);
  - retained categorical features are one-hot encoded (0/1 per category); the
    most frequent category is the reference level and gets no column, so the
    one-hot columns do not always sum to 1 (same dummy-variable-trap rule as
    the time_period_* columns in the analysis);
  - transaction_id is kept as the row identifier and is_fraud as the target.

Outputs: 6G_FEATURE_SUBSET_DATASET.csv and 6G_FEATURE_SUBSET_REPORT.csv
(one row per output column: which feature it comes from and why).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "phase_1"
DEFAULT_INPUT = DATA / "6D_TRANSFORMED_NORMALIZED_DATASET.csv"
DEFAULT_REPORT = DATA / "6F_FEATURE_RELATIONSHIP_REPORT.csv"
DEFAULT_OUTPUT = DATA / "6G_FEATURE_SUBSET_DATASET.csv"
DEFAULT_SUMMARY = DATA / "6G_FEATURE_SUBSET_REPORT.csv"

TARGET = "is_fraud"
ID_COLUMN = "transaction_id"
RETAIN = "retain"


def slug(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(value).strip().casefold()).strip("_") or "unknown"


def read_report(report_path: Path) -> pd.DataFrame:
    if not Path(report_path).is_file():
        raise ValueError(f"Feature relationship report not found: {report_path}. "
                         "Run 6_4_feature_relationship_analysis.py first.")
    report = pd.read_csv(report_path, dtype=str, keep_default_na=False)
    required = {"feature", "feature_type", "potential_leakage", "recommendation"}
    if required - set(report.columns):
        raise ValueError(f"Invalid feature relationship report: {report_path}")
    selected = report[report["recommendation"] == RETAIN]
    if selected.empty:
        raise ValueError("The relationship report does not retain any feature.")
    leaking = selected[selected["potential_leakage"] != ""]
    if not leaking.empty:
        raise ValueError(f"Retained features flagged for target leakage: {list(leaking['feature'])}")
    return selected


def select_features(input_path=DEFAULT_INPUT, report_path=DEFAULT_REPORT,
                    output_path=DEFAULT_OUTPUT, summary_path=DEFAULT_SUMMARY):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Output path must not overwrite the input dataset.")
    selected = read_report(report_path)
    data = pd.read_csv(input_path, low_memory=False)
    missing = {ID_COLUMN, TARGET, *selected["feature"]} - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    target = pd.to_numeric(data[TARGET], errors="coerce")
    if not target.isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be 0/1.")

    columns = {ID_COLUMN: data[ID_COLUMN]}
    summary = []
    for _, row in selected.iterrows():
        feature, kind = row["feature"], row["feature_type"]
        if kind == "categorical":
            values = data[feature].astype(str).str.strip().str.casefold()
            reference = values.value_counts().index[0]      # most frequent level
            summary.append({"column": f"{feature}_{slug(reference)}", "source_feature": feature,
                            "feature_type": kind,
                            "encoding": f"not written: reference level ('{reference}' = all {feature}_* are 0)",
                            "mutual_information": row.get("mutual_information", ""),
                            "association": row.get("association", "")})
            for level in sorted(values.unique()):
                if level == reference:
                    continue
                name = f"{feature}_{slug(level)}"
                columns[name] = (values == level).astype(int)
                summary.append({"column": name, "source_feature": feature, "feature_type": kind,
                                "encoding": f"one-hot ({feature} == '{level}')",
                                "mutual_information": row.get("mutual_information", ""),
                                "association": row.get("association", "")})
        else:
            values = pd.to_numeric(data[feature], errors="coerce")
            if values.isna().any():
                raise ValueError(f"Selected feature {feature} has missing values.")
            columns[feature] = values
            summary.append({"column": feature, "source_feature": feature, "feature_type": kind,
                            "encoding": "as in 6D (already transformed / normalized)",
                            "mutual_information": row.get("mutual_information", ""),
                            "association": row.get("association", "")})
    columns[TARGET] = target.astype(int)
    subset = pd.DataFrame(columns)

    if len(subset) != len(data):
        raise ValueError("Row count changed during feature selection.")
    if subset.isna().any().any():
        raise ValueError("Feature subset contains missing values.")
    duplicated = subset.columns[subset.columns.duplicated()]
    if len(duplicated):
        raise ValueError(f"Duplicate output columns: {list(duplicated)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(summary_path, index=False)
    return subset, summary


def main():
    parser = argparse.ArgumentParser(description="Phase I - feature subset selection (on 6D)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    subset, summary = select_features(args.input, args.report, args.output, args.summary)

    features = summary["source_feature"].unique()
    written = summary[~summary["encoding"].str.startswith("not written")]
    print("FEATURE SUBSET SELECTION COMPLETED")
    print(f"Rows: {len(subset):,} | selected features: {len(features)} | "
          f"output feature columns: {len(written)} (+ {ID_COLUMN}, {TARGET})")
    print(f"Fraud rate: {subset[TARGET].mean():.4f}")
    print("Selected features: " + ", ".join(features))


if __name__ == "__main__":
    main()
