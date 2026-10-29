# PVDH-Grupi12

## Phase I: feature analysis and selection

Run the Task B scripts from the repository root, in order:

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
