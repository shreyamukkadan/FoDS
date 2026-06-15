# FoDS Project

Group project repository for the thyroid cancer risk data science workflow.

## Project Structure

- `data/` dataset files
- `src/preprocessing/` shared preprocessing and exploratory-output code
- `src/statistical_testing/` statistical testing code
- `src/model_training/` machine learning model training and evaluation code
- `outputs/preprocessing/` preprocessing summaries and exploratory figures
- `outputs/statistical_testing/` statistical testing outputs
- `outputs/model_training/` model training and evaluation outputs

## Setup

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running The Code

Run all commands from the project root.

### Preprocessing And Exploratory Outputs

Run the shared preprocessing and exploratory-output pipeline:

```bash
python -m src.preprocessing.pipeline
```

This writes preprocessing summaries, group diagnostics, and exploratory figures to:

```text
outputs/preprocessing/
```

To quickly check that the model-ready preprocessing function works:

```bash
python -c "from src.preprocessing.pipeline import load_and_preprocess; load_and_preprocess(verbose=True)"
```

### Statistical Testing

Run the statistical tests:

```bash
python -m src.statistical_testing.analysis
```

This writes:

```text
outputs/statistical_testing/chi_square_results.csv
outputs/statistical_testing/mann_whitney_results.csv
outputs/statistical_testing/plots/
```

### Model Training And Evaluation

Run the full machine learning pipeline:

```bash
python -m src.model_training.train
```

This writes model metrics, feature selection tables, plots, model summaries, and saved model artifacts to:

```text
outputs/model_training/
```

The model training pipeline compares Logistic Regression, Random Forest, and Histogram Gradient Boosting.

## Suggested Full Run Order

```bash
python -m src.preprocessing.pipeline
python -m src.statistical_testing.analysis
python -m src.model_training.train
```

## Branch Workflow

- `main` is the stable shared branch
- each team member creates a separate branch from `main`
- finished work gets merged back into `main`
