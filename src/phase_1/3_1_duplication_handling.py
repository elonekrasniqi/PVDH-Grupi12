from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


def find_root() -> Path:
   
    here = Path(__file__).resolve()
    for folder in here.parents:
        if (folder / "dataset").is_dir():
            return folder
    return here.parent


ROOT = find_root()
DEFAULT_INPUT = ROOT / "dataset" / "phase_1" / "2B_MISSING_VALUES_HANDLED_DATASET.csv"
DEFAULT_REPORT = ROOT / "dataset" / "phase_1" / "3F_DUPLICATES_REPORT.csv"
DEFAULT_OUTPUT = ROOT / "dataset" / "phase_1" / "3F_DUPLICATES_REMOVED_DATASET.csv"
KEY_COLUMN = "transaction_id"
REPORT_COLUMNS = ("row_number", "transaction_id", "duplicate_type", "duplicate_of_row", "action")

REMOVE = "remove"
KEEP = "keep"
EXACT = "exact duplicate"
NEAR = "same record, id differs only in case"
COLLISION = "id collision: different records share an id"
REMOVED_STATUSES = (EXACT, NEAR)

Finding = tuple[int, str, str, int, str]  


def digest(parts: Iterable[str]) -> bytes:
    """Compact fingerprint of a whole row, so large files fit in memory."""
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.digest()


def is_standard_id(value: str) -> bool:
    """True when the part after the prefix is lower-case ('TX_df6b57ef')."""
    prefix, separator, body = value.partition("_")
    return body == body.lower() if separator else value == value.lower()


def read_columns(reader: csv.DictReader) -> list[str]:
    columns = reader.fieldnames
    if columns is None:
        raise ValueError("Input dataset has no header.")
    if KEY_COLUMN not in columns:
        raise ValueError(f"Input is missing the key column: {KEY_COLUMN}")
    return list(columns)


# ---------------------------------------------------------------- analysis --
def scan_duplicates(input_path: Path) -> tuple[int, list[Finding]]:
    """Find duplicates. Reads the dataset only; nothing is changed.

    A duplicate is the same record stored more than once. Two rows are the
    same record when every column is identical, apart from the case of the
    transaction_id (the earlier steps left some ids in upper case).
      EXACT     - identical copy of a kept row (removed)
      NEAR      - same record, id written in a different case (removed)
      COLLISION - same id (ignoring case) but a different record. These are
                  separate transactions, so they are reported and kept.
    Among copies of one record the row whose id is already in standard form
    is kept; otherwise the first one in the file.
    """
    # folded fingerprint (record with case-insensitive id) -> [(row, id, raw fingerprint)]
    records: dict[bytes, list[tuple[int, str, bytes]]] = defaultdict(list)
    records_of_id: dict[str, list[bytes]] = defaultdict(list)
    rows_read = 0

    with input_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = read_columns(reader)
        for row_number, row in enumerate(reader, start=2):
            rows_read += 1
            values = [row.get(column) or "" for column in columns]
            key = row.get(KEY_COLUMN) or ""
            folded = digest(v.casefold() if c == KEY_COLUMN else v for c, v in zip(columns, values))
            if key and folded not in records:
                records_of_id[key.casefold()].append(folded)
            records[folded].append((row_number, key, digest(values)))

    findings: list[Finding] = []
    kept: dict[bytes, tuple[int, str, bytes]] = {}
    for folded, members in records.items():
        keeper = next((m for m in members if is_standard_id(m[1])), members[0])
        kept[folded] = keeper
        for row_number, key, raw in members:
            if row_number != keeper[0]:
                status = EXACT if raw == keeper[2] else NEAR
                findings.append((row_number, key, status, keeper[0], REMOVE))

    for folded_list in records_of_id.values():
        if len(folded_list) > 1:
            base = min(kept[f][0] for f in folded_list)
            for folded in folded_list:
                row_number, key, _ = kept[folded]
                if row_number != base:
                    findings.append((row_number, key, COLLISION, base, KEEP))

    findings.sort(key=lambda f: f[0])
    return rows_read, findings


def analyze_duplicates(input_path: Path, report_path: Path | None) -> tuple[int, Counter[str]]:
    """ANALYSIS stage: find duplicates"""
    rows_read, findings = scan_duplicates(input_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as report:
            writer = csv.writer(report)
            writer.writerow(REPORT_COLUMNS)
            writer.writerows(findings)
    return rows_read, Counter(f[2] for f in findings)


# ---------------------------------------------------------------- handling --
def load_report(report_path: Path) -> dict[int, str]:
    """Rows the analysis marked for removal: file row number -> transaction_id."""
    if not report_path.exists():
        raise ValueError(
            f"Analysis report not found: {report_path}. Run the analysis stage first."
        )
    with report_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or set(REPORT_COLUMNS).difference(reader.fieldnames):
            raise ValueError(f"Not a valid duplicates report: {report_path}")
        return {int(r["row_number"]): r["transaction_id"] for r in reader if r["action"] == REMOVE}


def remove_reported_duplicates(
    input_path: Path, report_path: Path, output_path: Path
) -> tuple[int, int]:
    """HANDLING stage: delete exactly the rows the analysis marked for removal.

    It does not look for duplicates itself. Each row is checked against the
    report (same row number, same transaction_id) so a report from another or
    modified file is rejected instead of deleting the wrong rows.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must not overwrite the source dataset.")
    to_remove = load_report(report_path)

    rows_written = 0
    rows_removed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open("r", newline="", encoding="utf-8") as source,
        output_path.open("w", newline="", encoding="utf-8") as destination,
    ):
        reader = csv.DictReader(source)
        columns = read_columns(reader)
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row_number, row in enumerate(reader, start=2):
            if row_number in to_remove:
                if (row.get(KEY_COLUMN) or "") != to_remove[row_number]:
                    raise ValueError(
                        f"Report does not match the dataset at row {row_number}; "
                        "run the analysis stage again."
                    )
                rows_removed += 1
                continue
            writer.writerow(row)
            rows_written += 1
    if rows_removed != len(to_remove):
        raise ValueError(
            "Report lists rows that are not in the dataset; run the analysis stage again."
        )
    return rows_written, rows_removed


# ------------------------------------------------------------ verification --
def verify_removal(
    output_path: Path, rows_before: int, rows_removed: int, collisions_before: int
) -> tuple[int, Counter[str]]:
    """Run the analysis again on the output and check the result."""
    rows_after, after = analyze_duplicates(output_path, None)
    problems = []
    remaining = {s: after[s] for s in REMOVED_STATUSES if after[s]}
    if remaining:
        problems.append(f"duplicates still present: {remaining}")
    if rows_after != rows_before - rows_removed:
        problems.append(f"expected {rows_before - rows_removed:,} rows, found {rows_after:,}")
    if after[COLLISION] != collisions_before:
        problems.append(
            f"id collisions changed from {collisions_before:,} to {after[COLLISION]:,}"
        )
    if problems:
        raise ValueError("Duplicate removal check failed: " + "; ".join(problems))
    return rows_after, after


def print_analysis(rows_read: int, counts: Counter[str], report_path: Path) -> None:
    removable = sum(counts[s] for s in REMOVED_STATUSES)
    print("ANALYSIS (dataset unchanged)")
    print(f"  Rows read: {rows_read:,}")
    print(f"  Duplicate rows to remove: {removable:,}")
    for status in REMOVED_STATUSES:
        print(f"    {status}: {counts[status]:,}")
    print(f"  Id collisions (different records, kept): {counts[COLLISION]:,}")
    print(f"  Report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Duplicate step in two stages: 'analyze' finds duplicates and writes a "
            "report, 'handle' removes exactly the reported rows and verifies the "
            "result. The default 'all' runs both."
        )
    )
    parser.add_argument("--stage", choices=("analyze", "handle", "all"), default="all")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.stage in ("analyze", "all"):
        rows_read, counts = analyze_duplicates(args.input, args.report)
        if rows_read == 0:
            raise ValueError(f"Input dataset is empty: {args.input}")
        print_analysis(rows_read, counts, args.report)

    if args.stage in ("handle", "all"):
        rows_written, rows_removed = remove_reported_duplicates(
            args.input, args.report, args.output
        )
        print("HANDLING (rows marked for removal deleted)")
        print(f"  Rows removed: {rows_removed:,}")
        print(f"  Rows written: {rows_written:,}")
        print(f"  Output: {args.output}")

        # the report was just written by (or checked against) this input
        collisions_before = sum(
            1 for _ in _collision_rows(args.report)
        )
        rows_after, after = verify_removal(
            args.output, rows_written + rows_removed, rows_removed, collisions_before
        )
        print("VERIFICATION (analysis run again on the output)")
        print(f"  Duplicate rows remaining: {sum(after[s] for s in REMOVED_STATUSES):,}")
        print(f"  Id collisions still present (kept): {after[COLLISION]:,}")
        print(f"  Rows in output: {rows_after:,}")
        print("Strategy: a duplicate is the same record stored again (id case ignored); the copy")
        print("  with the standard-form id is kept. Different records sharing an id are kept.")


def _collision_rows(report_path: Path) -> Iterable[str]:
    with report_path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["duplicate_type"] == COLLISION:
                yield row["row_number"]


if __name__ == "__main__":
    main()