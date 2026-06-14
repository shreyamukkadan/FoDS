# FoDS Project

Group project repository for the thyroid cancer risk data science workflow.

## Project Structure

- `data/` dataset files
- `notebooks/` exploratory notebooks and exported P1 figures
- `src/p1/` shared preprocessing code
- `src/p2/` statistical testing code
- `src/p3/` machine learning model training and evaluation code
- `outputs/p2/` statistical testing outputs
- `outputs/p3/` timestamped P3 model outputs

## Setup

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If you are on macOS and P3 fails with a missing `libomp.dylib` error from XGBoost, install the OpenMP runtime with Homebrew:

```bash
brew install libomp
```

You can check that XGBoost loads correctly with:

```bash
python3 -c "import xgboost; print('xgboost import OK')"
```

## Running The Code

Run all commands from the project root.

### P1: Exploratory Data Analysis

Open the notebook:

```bash
jupyter notebook notebooks/p1_eda.ipynb
```

Or start JupyterLab if you prefer:

```bash
jupyter lab
```

To quickly check that the shared P1 preprocessing code runs from the terminal:

```bash
python3 -c "from src.p1.preprocessing import load_and_preprocess; load_and_preprocess('data/thyroid_cancer_risk_data.csv')"
```

### P2: Statistical Testing

Run the statistical tests:

```bash
python3 -m src.p2.statistical_testing
```

This writes:

```text
outputs/p2/chi_square_results.csv
outputs/p2/mann_whitney_results.csv
```

### P3: Model Training And Evaluation

Run the full machine learning pipeline:

```bash
python3 -m src.p3.models
```

This creates a new timestamped output folder such as:

```text
outputs/p3/run_YYYYMMDD_HHMMSS/
```

The P3 folder contains model metrics, feature selection tables, plots, model summaries, and saved model artifacts.

## Suggested Full Run Order

```bash
python3 -m src.p2.statistical_testing
python3 -m src.p3.models
```

Run the P1 notebook separately through Jupyter when you want to regenerate or inspect the exploratory analysis.

## Branch Workflow

- `main` is the stable shared branch
- each team member creates a separate branch from `main`
- finished work gets merged back into `main`
