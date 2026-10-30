"""Compare raw vs format-standardized data using identical validation rules.

Run: py src/phase_1/1_4_data_quality_comparison.py
Does not modify either input CSV. Requires 1_3_data_type_validation.py.
"""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'dataset' / 'synthetic_fraud_data.csv'
STANDARDIZED = ROOT / 'dataset' / 'phase_1' / '1F_FORMAT_STANDARDIZED_DATASET.csv'
OUTPUT = ROOT / 'dataset' / 'phase_1' / '1H_DATA_QUALITY_COMPARISON.csv'
VALIDATOR = Path(__file__).with_name('1_3_data_type_validation.py')
METRICS = ('total_rows', 'non_missing', 'missing_count', 'valid_count',
           'invalid_count', 'format_issue_count', 'unclassified_count')


def load_validator(path=VALIDATOR):
    path = Path(path)
    spec = importlib.util.spec_from_file_location('phase1_type_validator', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import validator: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'profile_dataset'):
        raise AttributeError('Validator must provide profile_dataset(path). Use the updated 1_3 script.')
    return module


def compare_stats(before, after):
    """Return one before/after report row per column, preserving all columns."""
    if set(before) != set(after):
        raise ValueError('Different column sets in raw and standardized datasets')
    rows = []
    for column in before:
        a, b = before[column], after[column]
        if a['total_rows'] != b['total_rows']:
            raise ValueError(f'Row count differs for {column}: {a["total_rows"]} vs {b["total_rows"]}')
        if a['expected_type'] != b['expected_type']:
            raise ValueError(f'Expected type differs for {column}')
        row = {'column': column, 'expected_type': a['expected_type']}
        for metric in METRICS:
            row[f'before_{metric}'] = a[metric]
            row[f'after_{metric}'] = b[metric]
            row[f'change_{metric}'] = b[metric] - a[metric]
        rows.append(row)
    return rows


def save_report(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No columns to report')
    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    for source in (RAW, STANDARDIZED):
        if not source.is_file():
            raise FileNotFoundError(f'Missing dataset: {source}')
        with source.open('r', encoding='utf-8-sig') as file:
            first_line = file.readline()
        if first_line.startswith('version https://git-lfs.github.com/spec/v1'):
            raise RuntimeError(f'{source} is a Git LFS pointer, not a CSV. Run: git lfs pull')

    validator = load_validator()
    before = validator.profile_dataset(RAW)
    after = validator.profile_dataset(STANDARDIZED)
    rows = compare_stats(before, after)
    save_report(rows, OUTPUT)
    print('Data quality comparison completed successfully.')
    print(f'Before: {RAW}')
    print(f'After:  {STANDARDIZED}')
    print(f'Report: {OUTPUT}')
    print(f'Rows: {rows[0]["before_total_rows"]} | Columns: {len(rows)}')
    for metric in ('missing_count', 'invalid_count', 'format_issue_count', 'unclassified_count'):
        x = sum(r[f'before_{metric}'] for r in rows)
        y = sum(r[f'after_{metric}'] for r in rows)
        print(f'{metric}: {x} -> {y} (change {y-x:+d})')
    print('Note: comparison covers FORMAT STANDARDIZATION only, not the full preprocessing pipeline.')
    print('A format issue can remain if 1_2 does not standardize that column (e.g. transaction_hour).')


if __name__ == '__main__':
    main()
