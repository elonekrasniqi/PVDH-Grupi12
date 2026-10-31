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

The target `is_fraud` is binary: **1 indicates fraud; 0 indicates a non-fraud transaction**.

### Attributes and data types

The dataset has **24 attributes**. The expected type of each one is defined in `1_3_data_type_validation.py` and checked in `1C_DATA_TYPE_REPORT.csv`.

| Attribute | Description | Attribute type | Stored as |
|---|---|---|---|
| `transaction_id` | Unique transaction identifier (`TX_...`) | Nominal (identifier) | string |
| `customer_id` | Customer identifier (`CUST_...`) | Nominal (identifier) | string |
| `card_number` | Card number (15–16 digits) | Nominal (identifier) | string |
| `timestamp` | Date and time of the transaction (with time zone) | Interval (datetime) | datetime |
| `merchant_category` | Merchant category (Retail, Travel, Grocery, ...) | Nominal | category |
| `merchant_type` | Sub-type of the merchant | Nominal | category |
| `merchant` | Merchant name | Nominal (high cardinality) | string |
| `amount` | Transaction amount in the local currency | Ratio, continuous | float |
| `currency` | ISO currency code (USD, EUR, JPY, ...) | Nominal | category |
| `country` | Country of the transaction | Nominal | category |
| `city` | City of the transaction | Nominal | category |
| `city_size` | Size of the city (medium, large, ...) | Ordinal | category |
| `card_type` | Card product (e.g. Basic Credit, Gold Credit) | Nominal | category |
| `card_present` | Card physically present | Binary | boolean |
| `device` | Device / payment method used | Nominal | category |
| `channel` | Channel (web, mobile, pos) | Nominal | category |
| `device_fingerprint` | Device identifier | Nominal (identifier) | string |
| `ip_address` | IP address | Nominal (identifier) | string |
| `distance_from_home` | 1 = transaction far from the customer's usual location | Binary | 0/1 |
| `high_risk_merchant` | Merchant marked as high risk | Binary | boolean |
| `transaction_hour` | Hour of the transaction (0–23) | Discrete, interval (cyclic) | integer |
| `weekend_transaction` | Transaction made on a weekend | Binary | boolean |
| `velocity_last_hour` | Customer activity in the previous hour: number of transactions, total amount, unique merchants, unique countries, largest amount | Structured text with 5 ratio values | text (dictionary) |
| `is_fraud` | Target: fraudulent transaction | Binary (target) | boolean |

We chose this dataset because fraud detection depends on both transaction-level characteristics and contextual information, such as time, location, spending behavior, and transaction frequency. The aim of this phase is to produce consistent and usable features for subsequent modeling, **not** to conclude that fraud can already be predicted accurately.

### Initial data quality results

| Measure | Result |
|---|---:|
| Original transaction rows | 409,099 |
| Invalid rows removed (missing target) | 1 |
| Removed duplicate rows | 9,126 |
| Rows after duplicate removal | 399,972 |
| Fraudulent transactions in cleaned dataset | 86,166 |
| Non-fraudulent transactions in cleaned dataset | 313,806 |
| Fraud rate after cleaning | Approximately 21.54% |

These figures refer to the validated dataset version used during the project. Downstream steps may use samples rather than all cleaned rows.

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

The tasks below are numbered in execution order. The task number is the prefix of its scripts and outputs: task *N* has scripts `N_1_...py`, `N_2_...py`, ... and outputs `NA_...`, `NB_...`, ... For example, task 1 → `1_1_data_profiling.py` → `1A_FORMAT_PROFILING_REPORT.csv`.

## 4.1 Task 1 — Data profiling and format standardization

**Scripts:** `1_1_data_profiling.py`, `1_2_format_standardization.py`, `1_3_data_type_validation.py`, `1_4_data_quality_comparison.py`

We first examined the input structure and checked whether important fields followed consistent formats. Format standardization is necessary because inconsistent identifiers, date representations, or values can cause incorrect comparisons, merges, and calculations.

Our scripts profile the source data and standardize:
- identifiers: `cust-123`, `cust_123` and `123` → `CUST_123` (same for `TX_`);
- card numbers: spaces and hyphens removed;
- timestamps: `30/09/2024 13:42:58` → ISO format `2024-09-30 13:42:58`;
- `transaction_hour`: decimal hours such as `15.0` → integer `15`;
- `card_present`: `yes`/`no` → `True`/`False`;
- letter case and surrounding whitespace of categorical columns.

The cleaned representation is then used by later stages.

**Outputs**
- `1A_FORMAT_PROFILING_REPORT.csv` — format profiling findings.
- `1B_FORMAT_STANDARDIZED_DATASET.csv` — standardized transactions.
- `1C_DATA_TYPE_REPORT.csv` — data-type validation report for the original dataset.
- `1D_DATA_QUALITY_COMPARISON.csv` — column-level quality comparison before and after format standardization.

### Data type validation

The `1_3_data_type_validation.py` script checks all **24 columns** against expected data types and formats. It distinguishes missing values, invalid values, format issues, and unclassified values. Empty cells and placeholders such as `?`, `null` and `n/a` are counted as missing. In the original **409,099-row** dataset, the validation reported **54,142 missing cells**, **1 invalid value**, **465,622 format issues**, and **0 unclassified values**. These are **cell-level counts**, not counts of distinct affected transactions.

### Data quality comparison: before vs. after format standardization

The `1_4_data_quality_comparison.py` script evaluates the original dataset and `1B_FORMAT_STANDARDIZED_DATASET.csv` using the same validation rules.

| Quality metric | Before standardization | After standardization | Change |
|---|---:|---:|---:|
| Rows | 409,099 | 409,099 | 0 |
| Columns | 24 | 24 | 0 |
| Missing cells | 54,142 | 54,142 | 0 |
| Invalid values | 1 | 1 | 0 |
| Format issues | 465,622 | 0 | -465,622 |
| Unclassified values | 0 | 0 | 0 |

**Result:** Standardization resolves the format issues found by the validator without changing the number of rows: **11,952** timestamp issues, **27,821** `card_present` values (`yes`/`no`), **16,751** `customer_id` values (numeric-only and `cust-` variants) and **409,098** decimal `transaction_hour` values. After standardization the validator reports **0** format issues.

The single invalid `transaction_id` value (`c`) belongs to a malformed row whose other fields are empty; it is removed in task 2 because its target is missing.

**Scope and interpretation:** This is a comparison **only of format standardization**, not the complete cleaning pipeline. Therefore, the unchanged missing-value count is expected: missing values are handled in task 2.

**Why this matters:** The reports quantify improvements, identify unresolved inconsistencies transparently, and make it possible to review data quality without assuming that every format warning is a data error.

## 4.2 Task 2 — Missing-value identification and handling

**Scripts:** `2_1_missing_value_analysis.py`, `2_2_missing_value_handling.py`

**Identification (`2_1`).** A value is missing when the cell is empty or holds a placeholder (`?`, `null`, `none`, `n/a`, `na`, `nan`). For `amount`, a value **≤ 0** is also missing: the dataset uses **`-999`** as a "no amount" placeholder, and a real transaction amount is always positive. `2A_MISSING_VALUES_REPORT.csv` lists, for every column, the number and percentage of missing values, how many are empty cells and which placeholders occur, and the strategy applied with its reason.

| Column | Missing | % | Strategy | Reason |
|---|---:|---:|---|---|
| `is_fraud` | 1 | 0.00% | Remove the row | The target cannot be imputed; a guessed label would bias the fraud rate |
| `amount` | 6,995 (3,547 empty + 3,448 `-999`) | 1.71% | Median of the same currency; currency inferred from country when missing; global median fallback | Amounts are in local currency and right-skewed: a per-currency median is robust and keeps the currency scale |
| `transaction_hour` | 1 | 0.00% | Hour taken from `timestamp`; median fallback | The hour is fully determined by the timestamp |
| `distance_from_home` | 1 | 0.00% | Median | Very few values missing |
| `currency` | 15,149 | 3.70% | `Unknown`, then resolved from country in task 4 (`currency_imputed = 1`) | Each country uses one currency |
| `merchant_category` / `card_type` / `channel` | 10,743 / 7,890 / 16,795 | 2.63% / 1.93% / 4.11% | Category `Unknown` | Nominal attributes: an explicit level keeps the row without inventing a category |

Placeholders found: `?` and `N/A` (all spellings) in the categorical columns, and `-999.0` in `amount`. The single missing value in `is_fraud`, `transaction_hour` and `distance_from_home` belongs to one malformed row, which is removed because its target is missing. In total, **57,590** missing values were identified in the standardized dataset.

**Handling (`2_2`).** The rules above are applied in one pass; valid values are written back unchanged. The median for a missing amount is computed **within the currency** because amounts are not yet converted to USD at this stage: a median that mixed JPY, NGN and USD amounts would be meaningless. When the currency itself is missing, it is inferred from the country (the same country-to-currency mapping as task 4) just for this purpose.

**Outputs**
- `2A_MISSING_VALUES_REPORT.csv` — missing values per column, placeholders, strategy and reason.
- `2B_MISSING_VALUES_HANDLED_DATASET.csv` — dataset after handling.

**Why this matters:** Missing-value treatment reduces data loss and keeps invalid values such as `-999` out of the USD conversion, the aggregations and the features. Outliers (extreme but valid amounts) are not changed here; they are treated in Phase II.

## 4.3 Task 3 — Duplicate identification and removal

**Script:** `3_1_duplication_handling.py`

Duplicate records can inflate transaction counts, fraud rates, and aggregate statistics. We implemented a duplicate scan and removal process, including a report of identified duplicate records. The process distinguishes a repeated stored record from an independent transaction.

**Results**
- Original transactions: **409,099**
- Rows entering this step: **409,098** (one invalid row was removed in task 2)
- Duplicates removed: **9,126** — 5,350 exact copies and 3,776 copies whose `transaction_id` differs only in letter case
- Transactions after cleaning: **399,972**
- ID collisions kept: **19** (different transactions that share an ID)

**Outputs**
- `3A_DUPLICATES_REPORT.csv`
- `3B_DUPLICATES_REMOVED_DATASET.csv`

**Why this matters:** Later aggregation and feature creation should not treat repeated copies of the same transaction as separate banking events.

## 4.4 Task 4 — Dataset integration and data aggregation

**Scripts:** `4_1_dataset_integration.py`, `4_2_data_aggregation.py`

### Integration

Banking transactions can be recorded in different currencies. Comparing raw amounts without currency conversion would be misleading. We integrated the cleaned transactions with exchange-rate data and produced a common **USD-denominated amount** for analysis.

In our validated integration run:
- **399,972** transactions were converted.
- **14,827** missing currency values were imputed using country-to-currency information.
- **0** missing conversions remained.
- **0** invalid converted amounts remained.

**Output:** `4A_INTEGRATED_DATASET.csv`

### Aggregation

We summarized the integrated dataset across four dimensions:

| Output | Grouping | Purpose |
|---|---|---|
| `4B_AGGREGATED_BY_COUNTRY.csv` | Country | Compare transaction volume, amount, and fraud frequency across countries |
| `4C_AGGREGATED_BY_CURRENCY.csv` | Currency | Compare transactions and fraud across original currencies |
| `4D_AGGREGATED_BY_DATE.csv` | Date | Examine changes in transaction and fraud activity over time |
| `4E_AGGREGATED_BY_MERCHANT_CATEGORY.csv` | Merchant category | Summarize fraud patterns by transaction type |

The date aggregation covered two normalized transaction dates in the validated dataset:

| Date | Transactions | Fraud transactions | Fraud rate |
|---|---:|---:|---:|
| 2024-09-30 | 240,910 | 47,874 | 19.87% |
| 2024-10-01 | 159,062 | 38,292 | 24.07% |

**Interpretation:** The observed fraud rate differs across the two dates. This is a descriptive result for the synthetic dataset, not evidence of a causal time effect.

## 4.5 Task 5 — Data sampling

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

## 4.6 Task 6 — Feature creation (feature engineering)

**Script:** `6_1_feature_creation.py`

We added **six transaction-related features** to enrich the information available for subsequent analysis. The script uses the stratified sample (`5B`) as its main input and the integrated dataset (`4A`) as supporting transaction history.

The new features describe aspects of transaction behavior, including customer spending patterns, night-time activity, activity away from the usual location, device sharing, and card characteristics. Examples of feature calculations include customer-average USD amount, the share of transactions away from the usual location, and the share occurring at night.

**Outputs**
- `6A_FEATURE_CREATION_DATASET.csv`
- `6B_FEATURE_CREATION_REPORT.csv`

**Why this matters:** A single transaction amount may not fully describe suspicious behavior. Derived features can provide context about customer behavior and device use.

**Limitation:** History-based features require careful temporal validation. A fraud model should not use transactions that happened after the transaction being evaluated.

## 4.7 Task 7 — Discretization and binarization

**Script:** `7_1_discretization_binarization.py`

Discretization transforms numerical values into meaningful groups. Binarization expresses conditions as binary indicators, usually 0 or 1.

In this stage we prepared categories and binary fields derived from transaction behavior and time. These representations can make selected patterns easier to analyze and allow categorical information to be incorporated into numerical processing.

**Outputs**
- `7A_DISCRETIZED_BINARIZED_DATASET.csv`
- `7B_DISCRETIZATION_BINARIZATION_REPORT.csv`

**Why this matters:** Some relationships are easier to interpret using ranges or yes/no indicators than using the original continuous values.

## 4.8 Task 8 — Data transformation and normalization

**Script:** `8_1_data_transformation_normalization.py`

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
- `8A_TRANSFORMED_NORMALIZED_DATASET.csv`
- `8B_TRANSFORMATION_NORMALIZATION_REPORT.csv`

**Why this matters:** Appropriate transformations can reduce differences in numerical scale and prepare features for methods such as PCA.

**Methodological note:** For unbiased model evaluation, statistics used for learned transformations should be fitted only on training data and reused for test data. The end-to-end preprocessing workflow still requires this check.

## 4.9 Task 9 — Feature relationship analysis

**Script:** `9_1_feature_relationship_analysis.py`

Not every available feature is equally useful. We analyzed candidate features in relation to the target `is_fraud` and examined redundancy between features.

The implementation uses measures appropriate to feature type, including:
- **Point-biserial correlation** for suitable numeric-to-binary relationships.
- **Cramér's V** for categorical associations.
- **Mutual information** for measuring statistical dependence.
- **Feature-to-feature relationships** to identify potentially redundant variables.

The analysis also flags identifiers and potentially problematic features that should not be used directly for prediction. A feature is marked as **possible target leakage** when one of its values (with at least 100 rows) is always or never fraud. This check is applied to categorical, binary and low-cardinality numeric features (at most 10 distinct values).

**Results:** of 42 candidate features, **9 are retained**. Excluded for possible target leakage: `device`, `card_present`, `channel`, `is_shared_device` and `customers_per_device_minmax` (transactions on a device shared by two or more customers are always fraud in this dataset). Other exclusions are redundant features (e.g. `currency_resolved` duplicates `country`, `city_size` duplicates `city`), low association with the target, constant columns (`weekend_transaction`) and high cardinality (`merchant`).

**Outputs**
- `9A_FEATURE_RELATIONSHIP_REPORT.csv`
- `9B_FEATURE_CORRELATION_PAIRS.csv`

**Why this matters:** Selecting features based on relevance and redundancy can reduce noise and improve interpretability. Association with the target does not establish causation.

## 4.10 Task 10 — Feature subset selection

**Script:** `10_1_feature_subset_selection.py`

Using the relationship report, we retained the features marked as suitable and excluded features flagged as identifiers, redundant, or potentially leaking target information. Selected categorical variables are converted to numerical columns through encoding.

**Selected features (9):** `amount_usd_log_z`, `hour_sin`, `hour_cos`, `distance_from_home`, `time_period_night`, `time_period_morning`, `time_period_afternoon`, `country`, `city`. After one-hot encoding of `country` and `city` this gives **28 feature columns** for PCA.

**Outputs**
- `10A_FEATURE_SUBSET_DATASET.csv`
- `10B_FEATURE_SUBSET_REPORT.csv`

The `10A` dataset is the **input to the PCA stage**. It contains selected feature columns alongside `transaction_id` and the target `is_fraud`.

**Why this matters:** PCA should operate on an intended set of numerical features rather than blindly processing every raw field.

## 4.11 Task 11 — Dimensionality reduction (PCA)

**Script:** `11_1_dimensionality_reduction.py`

PCA reduces the number of input dimensions by finding new components that summarize variation in the original features. This is useful when many variables contain overlapping information.

PCA uses the feature-subset output of task 10, `10A_FEATURE_SUBSET_DATASET.csv`. The script excludes `transaction_id` and `is_fraud` from PCA input features.

### Procedure

1. Read and validate the selected feature dataset.
2. Separate features from the fraud target.
3. Split the input into **80% training and 20% testing**, stratified by `is_fraud`.
4. Fit PCA on the training partition.
5. Choose enough components to retain at least **95% explained variance**.
6. Transform both partitions using the fitted PCA.
7. Save transformed datasets, the report, and the fitted pipeline.

### PCA results

| Metric | Result |
|---|---:|
| Rows used in this PCA run | 39,997 |
| Training rows | 31,997 |
| Testing rows | 8,000 |
| Original input features | 28 |
| Principal components retained | **14** |
| Explained variance retained | **95.2226%** |
| Required explained variance | 95% |
| PCA unit tests | **20 passed / 20 run** |

The number of features was reduced from **28 to 14**, a reduction of **50%**, while preserving **95.22%** of the variance in the PCA training data.

**Interpretation:** The selected features contain enough shared structure that fewer principal components can represent most of their variance. Explained variance is **not** the same as fraud-prediction accuracy, which will require model training and evaluation in a later phase.

### PCA artifacts

| File | Description |
|---|---|
| `11A_PCA_TRAIN.csv` | Principal-component values for training records |
| `11B_PCA_TEST.csv` | Principal-component values for testing records |
| `11C_PCA_REPORT.json` | PCA parameters and explained-variance results |
| `11D_PCA_PIPELINE.joblib` | Serialized fitted PCA processing pipeline |


---

# 5. Summary of Phase I results

| Stage | Main outcome |
|---|---|
| Format standardization | Standardized identifiers, timestamps, hours, card numbers and booleans; format issues 465,622 → 0 |
| Missing-value treatment | Identified missing values and placeholders (`?`, `-999`) per column and applied a documented strategy for each (`2A` report) |
| Data cleaning | Reduced 409,099 original records to 399,972 (1 invalid row and 9,126 duplicates removed) |
| Currency integration | Converted 399,972 records to a common USD-based representation |
| Currency imputation | Resolved 14,827 missing currency entries from country; no missing conversions remained |
| Aggregation | Generated summaries by country, currency, date, and merchant category |
| Sampling | Implemented four sampling methods |
| Feature creation | Added six transaction-related features |
| Discretization and binarization | Produced grouped and binary feature representations |
| Transformation and normalization | Prepared transformed, encoded, and scaled features |
| Feature analysis and selection | Produced relationship reports and a selected feature dataset |
| PCA | Reduced 28 features to 14 components, retaining 95.2226% variance |
| Testing | Full test suite: 136 passed; `validate_phase_1.py` passed |

The results above refer to the documented runs. Not every validation claim applies automatically to a future dataset version.

# 6. Project structure

```text
PVDH-Grupi12/
├── dataset/
│   ├── synthetic_fraud_data.csv
│   ├── exchange_rates_2024.csv
│   └── phase_1/
│       ├── 1A_ ... 1D_    # Task 1: format profiling, standardization, type validation, quality comparison
│       ├── 2A_, 2B_       # Task 2: missing-value report, handled dataset
│       ├── 3A_, 3B_       # Task 3: duplicates report, deduplicated dataset
│       ├── 4A_ ... 4E_    # Task 4: integrated dataset, aggregations
│       ├── 5A_ ... 5E_    # Task 5: samples and sampling report
│       ├── 6A_, 6B_       # Task 6: feature creation
│       ├── 7A_, 7B_       # Task 7: discretization & binarization
│       ├── 8A_, 8B_       # Task 8: transformation & normalization
│       ├── 9A_, 9B_       # Task 9: feature relationship analysis
│       ├── 10A_, 10B_     # Task 10: feature subset selection
│       └── 11A_ ... 11D_  # Task 11: PCA artifacts
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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the dependencies

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
python .\src\phase_1\1_1_data_profiling.py
python .\src\phase_1\1_2_format_standardization.py
python .\src\phase_1\1_3_data_type_validation.py
python .\src\phase_1\1_4_data_quality_comparison.py
python .\src\phase_1\2_1_missing_value_analysis.py
python .\src\phase_1\2_2_missing_value_handling.py
python .\src\phase_1\3_1_duplication_handling.py
python .\src\phase_1\4_1_dataset_integration.py
python .\src\phase_1\4_2_data_aggregation.py
python .\src\phase_1\5_1_data_sampling.py
python .\src\phase_1\6_1_feature_creation.py
python .\src\phase_1\7_1_discretization_binarization.py
python .\src\phase_1\8_1_data_transformation_normalization.py
python .\src\phase_1\9_1_feature_relationship_analysis.py
python .\src\phase_1\10_1_feature_subset_selection.py
python .\src\phase_1\11_1_dimensionality_reduction.py
```

**Note:** If `python` is not found, try `py` (Windows launcher) or `python3` (macOS / Linux). Running the complete pipeline will regenerate intermediate outputs.

### Run PCA only

If `dataset/phase_1/10A_FEATURE_SUBSET_DATASET.csv` already exists:

```powershell
python .\src\phase_1\11_1_dimensionality_reduction.py
```

### Run tests

PCA tests:

```powershell
python -m unittest discover -s tests -p "test_dimensionality_reduction.py" -v
```

Complete test suite:

```powershell
python -m pytest tests -v
```

Tests that check the generated outputs are skipped automatically when the data files are not available (for example, when only Git LFS pointers are present).