# Smart Banking Analytics — Fraud Detection
## Phase I: Data Preprocessing

**University of Prishtina — Faculty of Electrical and Computer Engineering (FIEK)**  
**Master's Program: Computer and Software Engineering**  
**Course: Data Preparation and Visualization**  
**Group: 12**

**Team Members**
- Brikena Kastrati
- Elonë Krasniqi
- Fjolla Kadriu

---

## 1. Introduction

The objective of **Smart Banking Analytics** is to prepare banking transaction data for future fraud-detection analysis. Transaction records can contain inconsistent formats, missing values, duplicate records, different currencies, and features that are not directly suitable for machine learning. These issues must be addressed before reliable analysis and modeling can take place.

This repository implements **Phase I — Data Preprocessing**. We focused on data cleaning, currency conversion, aggregation, sampling, feature engineering, feature analysis and selection, transformation, normalization, and dimensionality reduction.

**Scope:** This document reports preprocessing work only. Fraud prediction models, classification accuracy, dashboards, and data visualizations are **not** claimed as Phase I results.

## 2. Dataset and project objective

The main input is `dataset/synthetic_fraud_data.csv`, a **synthetic** transaction dataset. It contains **409,099 initial rows** in the version used for our Phase I work. The additional file `dataset/exchange_rates_2024.csv` provides exchange-rate information used for converting transaction amounts to USD.

Examples of fields include transaction identifiers, customer identifiers, timestamps, transaction amounts and currencies, countries, merchant categories, and the target `is_fraud`. The target is binary: **1 indicates fraud; 0 indicates a non-fraud transaction**.

We chose this dataset because fraud detection depends on both transaction-level characteristics and contextual information, such as time, location, spending behavior, and transaction frequency. The aim of this phase is to produce consistent and usable features for subsequent modeling, **not** to conclude that fraud can already be predicted accurately.

### Initial data quality results

| Measure | Result |
|---|---:|
| Original transaction rows | 409,099 |
| Rows after duplicate removal | 399,972 |
| Removed duplicate rows | 9,127 |
| Fraudulent transactions in cleaned dataset | 86,166 |
| Non-fraudulent transactions in cleaned dataset | 313,806 |
| Fraud rate after cleaning | Approximately 21.54% |

These figures refer to the validated dataset version used during the project. Downstream steps may use samples rather than all 399,972 rows.

## 3. Tools and technologies

- **Python** for the preprocessing scripts.
- **pandas** for reading, cleaning, grouping, and transforming tabular data.
- **NumPy** for numerical calculations.
- **scikit-learn** for feature analysis, train/test splitting, and PCA.
- **SciPy** for statistical measures used in feature analysis.
- **joblib** for saving the fitted PCA pipeline.
- **unittest / pytest** for automated validation.
- **Git and GitHub** for version control and collaboration.
- **Git LFS** for large dataset or generated artifact files where applicable.

The implementation is organized as separate scripts, allowing us to test individual stages and preserve intermediate outputs for inspection.

---

# 4. Phase I — Preprocessing methods and results

## 4.1 Data profiling and format standardization

**Scripts:** `1_1_data_profiling.py`, `1_2_format_standardization.py`, `1_3_data_type_validation.py`, `1_4_data_quality_comparison.py`

We first examined the input structure and checked whether important fields followed consistent formats. Format standardization is necessary because inconsistent identifiers, date representations, or values can cause incorrect comparisons, merges, and calculations.

Our scripts profile the source data and standardize supported identifiers, card-number representations, timestamps, and other specified fields. The cleaned representation is then used by later stages.

**Outputs**
- `1F_FORMAT_PROFILING_REPORT.csv` — format profiling findings.
- `1F_FORMAT_STANDARDIZED_DATASET.csv` — standardized transactions.
- `1G_DATA_TYPE_REPORT.csv` — data-type validation report for the original dataset.
- `1H_DATA_QUALITY_COMPARISON.csv` — column-level quality comparison before and after format standardization.

### Data type validation

The `1_3_data_type_validation.py` script checks all **24 columns** against expected data types and formats. It distinguishes missing values, invalid values, format issues, and unclassified values. In the original **409,099-row** dataset, the validation reported **41,634 missing cells**, **1 invalid value**, **465,622 format issues**, and **0 unclassified values**. These are **cell-level counts**, not counts of distinct affected transactions.

### Data quality comparison: before vs. after format standardization

The `1_4_data_quality_comparison.py` script evaluates the original dataset and `1F_FORMAT_STANDARDIZED_DATASET.csv` using the same validation rules.

| Quality metric | Before standardization | After standardization | Change |
|---|---:|---:|---:|
| Rows | 409,099 | 409,099 | 0 |
| Columns | 24 | 24 | 0 |
| Missing cells | 41,634 | 41,634 | 0 |
| Invalid values | 1 | 1 | 0 |
| Format issues | 465,622 | 417,539 | -48,083 |
| Unclassified values | 0 | 0 | 0 |

**Result:** Standardization reduced reported format issues by **48,083 (10.33%)**, without changing the number of rows. For example, **11,952 timestamp format issues** were resolved and `customer_id` format issues decreased from **16,751 to 8,441**.

The **417,539 remaining format issues** are concentrated in two fields: `transaction_hour` (**409,098**) and `customer_id` (**8,441**). In particular, decimal hour representations such as `15.0` are classified as format issues by the validator because it expects an integer-hour representation; they are not necessarily semantically invalid hours. The remaining customer identifier formats require further evaluation before any additional normalization. The single invalid `transaction_id` value also remains flagged.

**Scope and interpretation:** This is a comparison **only of format standardization**, not the complete cleaning pipeline. Therefore, the unchanged missing-value count does not imply that the later missing-value handling step failed. The team retained the existing standardization logic rather than modifying values solely to force the validation report to zero.

**Why this matters:** The reports quantify improvements, identify unresolved inconsistencies transparently, and make it possible to review data quality without assuming that every format warning is a data error.

## 4.2 Missing-value identification and handling

**Scripts:** `2_1_missing_value_analysis.py`, `2_2_missing_value_handling.py`

We analyzed missing fields and implemented treatment rules instead of deleting every incomplete transaction. The amount-handling strategy uses a **median within the relevant currency group**, with a global median fallback when necessary. The implementation also checks important values, including the validity of the fraud target.

**Output:** `2B_MISSING_VALUES_HANDLED_DATASET.csv`

**Why this matters:** Missing-value treatment reduces data loss and helps prevent invalid numeric inputs from reaching later analysis. The treatment strategy must still be reviewed for any deployment scenario where training/test isolation is required.

## 4.3 Duplicate identification and removal

**Script:** `3_1_duplication_handling.py`

Duplicate records can inflate transaction counts, fraud rates, and aggregate statistics. We implemented a duplicate scan and removal process, including a report of identified duplicate records. The process distinguishes a repeated stored record from an independent transaction.

**Results**
- Original transactions: **409,099**
- Transactions after cleaning: **399,972**
- Duplicates removed: **9,127**

**Outputs**
- `3F_DUPLICATES_REPORT.csv`
- `3F_DUPLICATES_REMOVED_DATASET.csv`

**Why this matters:** Later aggregation and feature creation should not treat repeated copies of the same transaction as separate banking events.

## 4.4 Dataset integration and data aggregation

**Scripts:** `4_1_dataset_integration.py`, `4_2_data_aggregation.py`

### Integration

Banking transactions can be recorded in different currencies. Comparing raw amounts without currency conversion would be misleading. We integrated the cleaned transactions with exchange-rate data and produced a common **USD-denominated amount** for analysis.

In our validated integration run:
- **399,972** transactions were converted.
- **14,827** missing currency values were imputed using country-to-currency information.
- **0** missing conversions remained.
- **0** invalid converted amounts remained.

**Output:** `4F_INTEGRATED_DATASET.csv`

### Aggregation

We summarized the integrated dataset across four dimensions:

| Output | Grouping | Purpose |
|---|---|---|
| `4A_AGGREGATED_BY_COUNTRY.csv` | Country | Compare transaction volume, amount, and fraud frequency across countries |
| `4B_AGGREGATED_BY_CURRENCY.csv` | Currency | Compare transactions and fraud across original currencies |
| `4C_AGGREGATED_BY_DATE.csv` | Date | Examine changes in transaction and fraud activity over time |
| `4D_AGGREGATED_BY_MERCHANT_CATEGORY.csv` | Merchant category | Summarize fraud patterns by transaction type |

The date aggregation covered two normalized transaction dates in the validated dataset:

| Date | Transactions | Fraud transactions | Fraud rate |
|---|---:|---:|---:|
| 2024-09-30 | 240,910 | 47,874 | 19.87% |
| 2024-10-01 | 159,062 | 38,292 | 24.07% |

**Interpretation:** The observed fraud rate differs across the two dates. This is a descriptive result for the synthetic dataset, not evidence of a causal time effect.

## 4.5 Data sampling

**Script:** `5_1_data_sampling.py`

The integrated dataset is large enough that smaller samples are useful for development and analysis. We implemented four sampling strategies, using a **10% default sampling fraction** and a fixed random seed for reproducibility.

| Sampling method | Description | Output |
|---|---|---|
| Simple random | Selects records randomly | `5A_SIMPLE_RANDOM_SAMPLE.csv` |
| Stratified | Preserves the distribution of specified groups | `5B_STRATIFIED_SAMPLE.csv` |
| Systematic | Selects records using a regular interval | `5C_SYSTEMATIC_SAMPLE.csv` |
| Balanced | Produces a sample with adjusted class representation | `5D_BALANCED_SAMPLE.csv` |

The sampling report is saved as `5E_SAMPLING_REPORT.csv`.

**Important distinction:** These are sampling outputs for preprocessing and analysis. They are **not the same** as the train/test split performed by PCA. A balanced sample must not be treated as a naturally representative test set.

## 4.6 Feature creation (feature engineering)

**Script:** `6_1_feature_creation.py`

We added **six transaction-related features** to enrich the information available for subsequent analysis. The script uses the stratified sample (`5B`) as its main input and the integrated dataset (`4F`) as supporting transaction history.

The new features describe aspects of transaction behavior, including customer spending patterns, night-time activity, activity away from the usual location, device sharing, and card characteristics. Examples of feature calculations include customer-average USD amount, the share of transactions away from the usual location, and the share occurring at night.

**Outputs**
- `6A_FEATURE_CREATION_DATASET.csv`
- `6B_FEATURE_CREATION_REPORT.csv`

**Why this matters:** A single transaction amount may not fully describe suspicious behavior. Derived features can provide context about customer behavior and device use.

**Limitation:** History-based features require careful temporal validation. A fraud model should not use transactions that happened after the transaction being evaluated.

## 4.7 Discretization and binarization

**Script:** `6_2_discretization_binarization.py`

Discretization transforms numerical values into meaningful groups. Binarization expresses conditions as binary indicators, usually 0 or 1.

In this stage we prepared categories and binary fields derived from transaction behavior and time. These representations can make selected patterns easier to analyze and allow categorical information to be incorporated into numerical processing.

**Outputs**
- `6B_DISCRETIZED_BINARIZED_DATASET.csv`
- `6C_DISCRETIZATION_BINARIZATION_REPORT.csv`

**Why this matters:** Some relationships are easier to interpret using ranges or yes/no indicators than using the original continuous values.

## 4.8 Data transformation and normalization

**Script:** `6_3_data_transformation_normalization.py`

Transaction variables may have very different numerical ranges. A monetary amount, a count, and an hour of the day cannot always be compared directly.

The implementation includes:
- **Numeric extraction** from transaction-velocity information.
- **Cyclical time encoding** using `hour_sin` and `hour_cos`, because 23:00 and 00:00 are adjacent times.
- **Categorical encoding** for selected ordered or time-related categories.
- **Logarithmic transformation** (`log1p`) for selected skewed monetary variables.
- **Z-score standardization** to express values relative to their mean and standard deviation.
- **Min-max normalization** to map values to a common range.

For z-score standardization, the principle is:

`z = (x - mean) / standard_deviation`

For min-max normalization:

`x_scaled = (x - minimum) / (maximum - minimum)`

The target `is_fraud` is preserved and is not scaled.

**Outputs**
- `6D_TRANSFORMED_NORMALIZED_DATASET.csv`
- `6E_TRANSFORMATION_NORMALIZATION_REPORT.csv`

**Why this matters:** Appropriate transformations can reduce differences in numerical scale and prepare features for methods such as PCA.

**Methodological note:** For unbiased model evaluation, statistics used for learned transformations should be fitted only on training data and reused for test data. The end-to-end preprocessing workflow still requires this check.

## 4.9 Feature relationship analysis

**Script:** `6_4_feature_relationship_analysis.py`

Not every available feature is equally useful. We analyzed candidate features in relation to the target `is_fraud` and examined redundancy between features.

The implementation uses measures appropriate to feature type, including:
- **Point-biserial correlation** for suitable numeric-to-binary relationships.
- **Cramér's V** for categorical associations.
- **Mutual information** for measuring statistical dependence.
- **Feature-to-feature relationships** to identify potentially redundant variables.

The analysis also flags identifiers and potentially problematic features that should not be used directly for prediction.

**Outputs**
- `6F_FEATURE_RELATIONSHIP_REPORT.csv`
- `6F_FEATURE_CORRELATION_PAIRS.csv`

**Why this matters:** Selecting features based on relevance and redundancy can reduce noise and improve interpretability. Association with the target does not establish causation.

## 4.10 Feature subset selection

**Script:** `6_5_feature_subset_selection.py`

Using the relationship report, we retained the features marked as suitable and excluded features flagged as identifiers, redundant, or potentially leaking target information. Selected categorical variables are converted to numerical columns through encoding.

**Outputs**
- `6G_FEATURE_SUBSET_DATASET.csv`
- `6G_FEATURE_SUBSET_REPORT.csv`

The `6G` dataset is the **input to the updated PCA stage**. It contains selected feature columns alongside `transaction_id` and the target `is_fraud`.

**Why this matters:** PCA should operate on an intended set of numerical features rather than blindly processing every raw field.

## 4.11 Dimensionality reduction — Principal Component Analysis (PCA)

**Script:** `7_1_dimensionality_reduction.py`

PCA reduces the number of input dimensions by finding new components that summarize variation in the original features. This is useful when many variables contain overlapping information.

We updated PCA to use the latest feature-subset output, `6G_FEATURE_SUBSET_DATASET.csv`, rather than the older input file. The script excludes `transaction_id` and `is_fraud` from PCA input features.

### Procedure

1. Read and validate the selected feature dataset.
2. Separate features from the fraud target.
3. Split the input into **80% training and 20% testing**, stratified by `is_fraud`.
4. Fit PCA on the training partition.
5. Choose enough components to retain at least **95% explained variance**.
6. Transform both partitions using the fitted PCA.
7. Save transformed datasets, the report, and the fitted pipeline.

### Verified local PCA results

| Metric | Result |
|---|---:|
| Rows used in this PCA run | 39,997 |
| Training rows | 31,997 |
| Testing rows | 8,000 |
| Original input features | 31 |
| Principal components retained | **15** |
| Explained variance retained | **96.3602%** |
| Required explained variance | 95% |
| PCA unit tests | **20 passed / 20 run** |

The number of features was reduced from **31 to 15**, a reduction of approximately **51.6%**, while preserving **96.36%** of the variance in the PCA training data.

**Interpretation:** The selected features contain enough shared structure that fewer principal components can represent most of their variance. Explained variance is **not** the same as fraud-prediction accuracy, which will require model training and evaluation in a later phase.

### PCA artifacts

| File | Description |
|---|---|
| `7A_PCA_TRAIN.csv` | Principal-component values for training records |
| `7B_PCA_TEST.csv` | Principal-component values for testing records |
| `7C_PCA_REPORT.json` | PCA parameters and explained-variance results |
| `7D_PCA_PIPELINE.joblib` | Serialized fitted PCA processing pipeline |

The updated PCA completed successfully in the reported local run. After correcting a pandas test-fixture dtype issue, **all 20 PCA unit tests passed**.

---

# 5. Summary of Phase I results

| Stage | Main outcome |
|---|---|
| Data cleaning | Reduced 409,099 original records to 399,972 after duplicate removal |
| Missing-value treatment | Implemented missing-value checks and treatment strategies |
| Currency integration | Converted 399,972 records to a common USD-based representation in the validated integration run |
| Currency imputation | Handled 14,827 missing currency entries; no missing conversions remained in that run |
| Aggregation | Generated summaries by country, currency, date, and merchant category |
| Sampling | Implemented four sampling methods |
| Feature creation | Added six transaction-related features |
| Discretization and binarization | Produced grouped and binary feature representations |
| Transformation and normalization | Prepared transformed, encoded, and scaled features |
| Feature analysis and selection | Produced relationship reports and a selected feature dataset |
| PCA | Reduced 31 features to 15 components, retaining 96.3602% variance |
| Testing | Updated PCA unit test suite passed: 20/20 |

The results above refer to the documented runs. Not every validation claim applies automatically to a future dataset version.

# 6. Project structure

```text
PVDH-Grupi12/
├── dataset/
│   ├── synthetic_fraud_data.csv
│   ├── exchange_rates_2024.csv
│   └── phase_1/
│       ├── 1F_...        # Standardization
│       ├── 1G_...        # Data type validation
│       ├── 1H_...        # Data quality before/after comparison
│       ├── 2B_...        # Missing-value handling
│       ├── 3F_...        # Duplicate handling
│       ├── 4A_... 4F_... # Integration and aggregation
│       ├── 5A_... 5E_... # Sampling
│       ├── 6A_... 6G_... # Feature processing
│       └── 7A_... 7D_... # PCA artifacts
├── src/
│   └── phase_1/           # Preprocessing scripts
├── tests/                 # Automated tests
└── README.md
```

# 7. How to run the project

### Clone the repository

```bash
git clone https://github.com/elonekrasniqi/PVDH-Grupi12.git
cd PVDH-Grupi12
```

### Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the dependencies

```bash
python -m pip install pandas numpy scipy scikit-learn joblib pytest
```

If the project has a maintained `requirements.txt`, it can be used instead:

```bash
python -m pip install -r requirements.txt
```

Some large data files may use Git LFS:

```bash
git lfs pull
```

### Run the preprocessing scripts

Run these from the repository root, in dependency order, after confirming that the expected input files are present:

```powershell
py .\src\phase_1\1_1_data_profiling.py
py .\src\phase_1\1_2_format_standardization.py
py .\src\phase_1\1_3_data_type_validation.py
py .\src\phase_1\1_4_data_quality_comparison.py
py .\src\phase_1\2_1_missing_value_analysis.py
py .\src\phase_1\2_2_missing_value_handling.py
py .\src\phase_1\3_1_duplication_handling.py
py .\src\phase_1\4_1_dataset_integration.py
py .\src\phase_1\4_2_data_aggregation.py
py .\src\phase_1\5_1_data_sampling.py
py .\src\phase_1\6_1_feature_creation.py
py .\src\phase_1\6_2_discretization_binarization.py
py .\src\phase_1\6_3_data_transformation_normalization.py
py .\src\phase_1\6_4_feature_relationship_analysis.py
py .\src\phase_1\6_5_feature_subset_selection.py
py .\src\phase_1\7_1_dimensionality_reduction.py
```

**Note:** The `py` command is a Windows Python launcher. On other systems, use the appropriate Python executable. Running the complete pipeline will regenerate intermediate outputs.

### Run PCA only

If `dataset/phase_1/6G_FEATURE_SUBSET_DATASET.csv` already exists:

```powershell
py .\src\phase_1\7_1_dimensionality_reduction.py
```

### Run tests

PCA tests:

```powershell
py -m unittest discover -s tests -p "test_dimensionality_reduction.py" -v
```

Complete test suite:

```powershell
py -m pytest tests -v
```

**Validation status:** The updated PCA test suite was run successfully with **20/20 passing**. A successful run of the *entire* repository test suite has not been established by that PCA result alone.