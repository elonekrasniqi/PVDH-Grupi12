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