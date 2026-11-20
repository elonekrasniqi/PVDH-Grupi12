
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/phase_2/2_1_multivariate_outlier_detection.py"
)

spec = importlib.util.spec_from_file_location("outlier_task", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        rng.normal(size=(100, len(module.FEATURES))),
        columns=module.FEATURES,
    )

    df["is_fraud"] = [0] * 90 + [1] * 10

    return df


def test_outputs_and_row_alignment():
    df = sample()
    result, report = module.detect(df)

    assert len(result) == len(df)

    assert result.source_row.tolist() == list(range(len(df)))

    assert result.is_fraud.tolist() == df.is_fraud.tolist()

    assert set(result.isolation_forest_outlier.unique()) <= {0, 1}

    assert set(result.lof_outlier.unique()) <= {0, 1}

    assert report["both_methods_outliers"] <= min(
        report["isolation_forest_outliers"],
        report["lof_outliers"],
    )


def test_missing_feature_fails():
    df = sample().drop(columns=module.FEATURES[0])

    try:
        module.detect(df)
    except ValueError as e:
        assert "Missing required" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_nonfinite_fails():
    df = sample()

    df.loc[0, module.FEATURES[0]] = np.nan

    try:
        module.detect(df)
    except ValueError as e:
        assert "non-finite" in str(e)
    else:
        raise AssertionError("Expected ValueError")
