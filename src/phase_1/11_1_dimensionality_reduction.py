"""PCA on the final selected feature dataset (6G).

Run after 6_5_feature_subset_selection.py. All feature columns in 6G are
numeric and already transformed/encoded by the preceding pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'dataset/phase_1/10A_FEATURE_SUBSET_DATASET.csv'
DEFAULT_DIR = ROOT / 'dataset/phase_1'
TARGET = 'is_fraud'
ID_COLUMN = 'transaction_id'


def run(input_path: Path, output_dir: Path, variance: float = 0.95, seed: int = 42):
    input_path, output_dir = Path(input_path), Path(output_dir)
    if not 0 < variance < 1:
        raise ValueError('Variance threshold must be in (0, 1) for variance-based PCA.')
    if not input_path.is_file():
        raise FileNotFoundError(f'Feature subset not found: {input_path}')
    with input_path.open('rb') as source:
        if source.read(40).startswith(b'version https://git-lfs.github.com/spec'):
            raise ValueError('Input is a Git LFS pointer. Run: git lfs pull')

    data = pd.read_csv(input_path, low_memory=False)
    if TARGET not in data.columns:
        raise ValueError(f'Missing required column: {TARGET}')
    if data.columns.duplicated().any():
        raise ValueError('Duplicate input column names.')
    features = [c for c in data.columns if c not in (TARGET, ID_COLUMN)]
    if len(features) < 2:
        raise ValueError('PCA needs at least two selected feature columns.')
    X = data[features].apply(pd.to_numeric, errors='raise')
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Selected features contain missing or non-finite values.')
    y = pd.to_numeric(data[TARGET], errors='raise')
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('Target must be binary (0/1).')
    y = y.astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError('Need at least two rows per target class for stratified splitting.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # 6G is already normalized/encoded. No second scaling or one-hot encoding.
    pipeline = Pipeline([('pca', PCA(n_components=variance, svd_solver='full'))])
    train_reduced = pipeline.fit_transform(X_train)
    test_reduced = pipeline.transform(X_test)
    pca = pipeline.named_steps['pca']
    columns = [f'PC{i + 1}' for i in range(train_reduced.shape[1])]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'train': output_dir / '11A_PCA_TRAIN.csv',
        'test': output_dir / '11B_PCA_TEST.csv',
        'report': output_dir / '11C_PCA_REPORT.json',
        'model': output_dir / '11D_PCA_PIPELINE.joblib',
    }
    for key, values, labels in [('train', train_reduced, y_train), ('test', test_reduced, y_test)]:
        frame = pd.DataFrame(values, columns=columns)
        frame[TARGET] = labels.to_numpy()
        frame.to_csv(paths[key], index=False)
    report = {
        'input': str(input_path), 'seed': seed, 'test_fraction': 0.2,
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'input_feature_count': len(features), 'encoded_feature_count': len(features),
        'pca_component_count': len(columns), 'variance_threshold': variance,
        'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'explained_variance_per_component': pca.explained_variance_ratio_.tolist(),
        'input_features': features, 'numeric_features': features, 'categorical_features': [],
        'target_excluded_from_pca': True,
        'note': 'Uses already transformed/encoded 6G features; PCA fitted only on the training partition.'
    }
    paths['report'].write_text(json.dumps(report, indent=2), encoding='utf-8')
    joblib.dump(pipeline, paths['model'])
    print(f'Train rows: {len(X_train):,}; test rows: {len(X_test):,}')
    print(f'Input features: {len(features)}; PCA components: {len(columns)}')
    print(f"Explained variance: {report['total_explained_variance']:.4%}")
    for label, path in paths.items():
        print(f'{label}: {path}')
    return report


def main():
    parser = argparse.ArgumentParser(description='PCA on selected 6G fraud features.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--variance', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.variance, args.seed)


if __name__ == '__main__':
    main()
