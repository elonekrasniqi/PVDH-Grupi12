# PVDH-Grupi12

## Phase I: feature analysis and selection (first version, on the sample)

This was the first version, run on the stratified sample before the
transformation step. It is replaced by the analysis on the transformed data
further below (`6_4` / `6_5`). Run from the repository root, in order:

```powershell
python .\src\phase_1\6_1_feature_relationship_analysis.py
python .\src\phase_1\6_2_feature_subset_selection.py
```

Both scripts use `dataset/phase_1/5B_STRATIFIED_SAMPLE.csv`, the 10% sample
stratified by fraud label and country. The analysis writes
`6A_FEATURE_RELATIONSHIP_REPORT.csv`; the selection step reads that report and
writes `6B_FEATURE_SUBSET_DATASET.csv`. The subset excludes direct identifiers,
potential target leakage, high-cardinality, redundant, and weakly associated
features. Use `--input`, `--output`, and `--report` to override the default
paths. The source sample is never modified.

## Phase I: data transformation and normalization

```powershell
python .\src\phase_1\6_3_data_transformation_normalization.py
```

The script reads the discretization & binarization output
`6B_DISCRETIZED_BINARIZED_DATASET.csv`, keeps all its columns and adds:

- **Transformation**: the `velocity_last_hour` text split into five numeric
  `velocity_*` columns; `hour_sin` / `hour_cos` (cyclic hour, so 23h sits next
  to 0h); Low/Medium/High levels as ordinal `*_code` columns (0/1/2);
  `transaction_time_period` as one-hot `time_period_*` columns; `log1p` on the
  strongly right-skewed money columns (`amount_usd`, `customer_avg_amount_usd`,
  `velocity_total_amount`). Invalid amounts (<= 0, the converted `-999`
  placeholder) and empty values are replaced with the median;
  `amount_usd_invalid` marks the replaced amounts.
- **Normalization**: z-score (`*_z`, `*_log_z`) for unbounded numbers and
  min-max to [0, 1] (`*_minmax`) for shares, `customers_per_device` and the
  ordinal codes. The target `is_fraud` is not scaled.

It writes `6D_TRANSFORMED_NORMALIZED_DATASET.csv` and
`6E_TRANSFORMATION_NORMALIZATION_REPORT.csv` (method, rows imputed, skew before
and after, and the mean/std or min/range used, so the same scaling can be
applied to new data). Tests: `python -m pytest tests/test_data_transformation_normalization.py`.

## Phase I: feature relationship analysis and subset selection (on 6D)

Run after the transformation step, from the repository root:

```powershell
python .\src\phase_1\6_4_feature_relationship_analysis.py
python .\src\phase_1\6_5_feature_subset_selection.py
```

`6_4` reads `6D_TRANSFORMED_NORMALIZED_DATASET.csv` and lists every column in
`6F_FEATURE_RELATIONSHIP_REPORT.csv`:

- identifiers and raw columns already replaced by a transformed version are
  marked `exclude_identifier` / `exclude_superseded`;
- each candidate feature is measured against `is_fraud` (point-biserial for
  numeric/binary, Cramér's V for categorical, mutual information for all) and
  kept when |association| >= 0.05 (0.04 for Cramér's V) or mutual information
  >= 0.01;
- possible target leakage (a value with >= 100 rows that is always or never
  fraud: `card_present`, `device`, `channel`), constant and high-cardinality
  features are excluded;
- relevant features are compared with each other (Spearman / Cramér's V); of a
  pair with |relationship| >= 0.85 the one with lower mutual information is
  dropped (for example `is_shared_device` vs `customers_per_device_minmax`,
  `currency_resolved` vs `country`). The pairs are in
  `6F_FEATURE_CORRELATION_PAIRS.csv`;
- if all levels of a one-hot group are kept, the weakest one is dropped
  (dummy-variable trap).

`6_5` keeps exactly the features the report marks `retain`, copies the numeric
ones as they are and one-hot encodes the categorical ones (the most frequent
category is the reference level without a column). It writes
`6G_FEATURE_SUBSET_DATASET.csv` (`transaction_id`, the features, `is_fraud`)
and `6G_FEATURE_SUBSET_REPORT.csv` (where each column comes from).
Tests: `python -m pytest tests/test_feature_analysis_selection_transformed.py`.
