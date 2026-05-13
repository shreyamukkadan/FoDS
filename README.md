# FoDS Project

Group project repository for the thyroid cancer risk data science workflow.

## Project structure

- `data/` raw dataset files
- `notebooks/` exploratory notebooks, including the P1 EDA notebook
- `src/p1/` shared preprocessing code
- `src/p2/` statistical testing code
- `src/p3/` machine learning model training and evaluation
- `src/p4/` feature selection and K-Means clustering analysis
- `outputs/p1/` EDA figures
- `outputs/p2/` statistical testing result tables
- `outputs/p3/` model comparison tables, plots, and saved best model
- `outputs/p4/` feature selection and clustering tables/plots

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If your system uses `python3`/`pip3`, use:

```bash
pip3 install -r requirements.txt
```

## Running the analysis

Run scripts from the project root:

```bash
python3 -m src.p2.statistical_testing
python3 -m src.p3.models
python3 -m src.p4.feature_importance
```

The P1 exploratory analysis is in:

```text
notebooks/p1_eda.ipynb
```

## Branch workflow

- `main` is the stable shared branch
- each team member creates a separate branch from `main`
- finished work gets merged back into `main`
