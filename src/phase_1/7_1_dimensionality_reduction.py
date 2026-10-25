"""Task E: leakage-safe dimensionality reduction of the selected fraud features."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'dataset/phase_1/6B_FEATURE_SUBSET_DATASET.csv'
DEFAULT_DIR = ROOT / 'dataset/phase_1'
NUMERIC = ['distance_from_home', 'transaction_hour']
CATEGORICAL = ['country', 'city', 'city_size']
TARGET = 'is_fraud'


def run(input_path: Path, output_dir: Path, variance: float = 0.95, seed: int = 42):
    if not 0 < variance <= 1:
        raise ValueError('Variance threshold must be in (0, 1].')
    data = pd.read_csv(input_path, dtype={k: 'string' for k in CATEGORICAL})
    required = NUMERIC + CATEGORICAL + [TARGET]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data[required].isna().any().any():
        raise ValueError('Selected feature dataset contains missing values.')
    for feature in NUMERIC:
        data[feature] = pd.to_numeric(data[feature], errors='raise')
    if not np.isfinite(data[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError('Numeric features contain infinite values.')
    y = pd.to_numeric(data[TARGET], errors='raise').astype(int)
    if not y.isin([0, 1]).all():
        raise ValueError('Target must be binary (0/1).')
    if y.value_counts().min() < 2 or y.nunique() != 2:
        raise ValueError('Need at least two rows per target class for stratified splitting.')
    X = data[NUMERIC + CATEGORICAL].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    preprocessing = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ], sparse_threshold=0.0)
    train_encoded = preprocessing.fit_transform(X_train)
    test_encoded = preprocessing.transform(X_test)
    # PCA can center features; standardize the one-hot output as well only if justified.
    pca = PCA(n_components=variance, svd_solver='full')
    train_reduced = pca.fit_transform(train_encoded)
    test_reduced = pca.transform(test_encoded)
    columns = [f'PC{i + 1}' for i in range(train_reduced.shape[1])]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'train': output_dir / '7A_PCA_TRAIN.csv',
        'test': output_dir / '7B_PCA_TEST.csv',
        'report': output_dir / '7C_PCA_REPORT.json',
        'model': output_dir / '7D_PCA_PIPELINE.joblib',
    }
    for key, features, labels in [('train', train_reduced, y_train), ('test', test_reduced, y_test)]:
        frame = pd.DataFrame(features, columns=columns)
        frame[TARGET] = labels.to_numpy()
        frame.to_csv(paths[key], index=False)
    report = {
        'input': str(input_path), 'seed': seed, 'test_fraction': 0.2,
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'input_feature_count': len(NUMERIC) + len(CATEGORICAL),
        'encoded_feature_count': train_encoded.shape[1],
        'pca_component_count': len(columns), 'variance_threshold': variance,
        'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'explained_variance_per_component': pca.explained_variance_ratio_.tolist(),
        'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'target_excluded_from_pca': True,
        'note': 'PCA is fitted only on the training partition; test is transformed without refitting.'
    }
    paths['report'].write_text(json.dumps(report, indent=2), encoding='utf-8')
    joblib.dump(Pipeline([('preprocessing', preprocessing), ('pca', pca)]), paths['model'])
    print(f"Train rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print(f"Encoded features: {train_encoded.shape[1]}; PCA components: {len(columns)}")
    print(f"Explained variance: {report['total_explained_variance']:.4%}")
    for label, path in paths.items():
        print(f'{label}: {path}')
    return report


def main():
    parser = argparse.ArgumentParser(description='Leakage-safe PCA for fraud feature subset.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--variance', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.variance, args.seed)


if __name__ == '__main__':
    main()
