# FoDS Project: Thyroid Cancer Malignancy Prediction

This repository contains the full data science workflow for predicting thyroid cancer diagnosis from a synthetic thyroid cancer risk dataset. The project includes preprocessing, statistical association testing, model training, feature-design comparison, and subgroup evaluation.

## Project Structure

- `data/` input dataset
- `config.py` shared paths, column names, split settings, and feature-set definitions
- `src/preprocessing/` preprocessing pipeline and exploratory outputs
- `src/statistical_testing/` chi-square and Mann-Whitney statistical tests
- `src/model_training/` machine learning model training and evaluation
- `outputs/preprocessing/csv/` preprocessing summaries and tables
- `outputs/preprocessing/plots/` preprocessing figures
- `outputs/statistical_testing/csv/` statistical test tables
- `outputs/statistical_testing/plots/` statistical test figures
- `outputs/model_training/csv/` model metrics and feature-selection tables
- `outputs/model_training/plots/` model training and evaluation figures

## Setup

Run all commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run Order

```bash
python3 -m src.preprocessing.pipeline
python3 -m src.statistical_testing.analysis
python3 -m src.model_training.train
```

The model training script may take several minutes because it trains three models, compares predefined feature designs, performs feature selection, and runs subgroup analyses.

## Pipeline Steps

### 1. Preprocessing

```bash
python3 -m src.preprocessing.pipeline
```

This loads the raw data, drops missing rows, performs a stratified train/test split, removes outliers from the training set only, and creates exploratory summaries and figures.

Outputs are written to:

```text
outputs/preprocessing/csv/
outputs/preprocessing/plots/
```

### 2. Statistical Testing

```bash
python3 -m src.statistical_testing.analysis
```

This runs chi-square tests for categorical variables and Mann-Whitney U tests for continuous variables, with Benjamini-Hochberg FDR correction.

Outputs are written to:

```text
outputs/statistical_testing/csv/
outputs/statistical_testing/plots/
```

### 3. Model Training and Evaluation

```bash
python3 -m src.model_training.train
```

This trains and evaluates Logistic Regression, Random Forest, and Histogram Gradient Boosting models. It also compares predefined feature designs, performs forward feature selection, evaluates dummy baselines, and runs country/ethnicity subgroup analyses.

Outputs are written to:

```text
outputs/model_training/csv/
outputs/model_training/plots/
```

Running this script also saves the trained model artifact locally.

## Reproducibility

The workflow uses a fixed random seed of `42` and an 80/20 stratified train/test split. Preprocessing transformations are fit on the training data only and then applied to held-out data to avoid data leakage.

The `outputs/` directory is generated automatically when the pipeline scripts are run. Each stage writes tabular outputs to a `csv/` subfolder and figures to a `plots/` subfolder.
