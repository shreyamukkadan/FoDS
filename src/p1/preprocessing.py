"""
src/preprocessing.py
Thyroid Cancer Risk — Foundations of Data Science Group Project

This file contains the preprocessing pipeline as a clean, importable function.
It is used by all group members so everyone works with identical cleaned data.

Usage (in any notebook):
    from src.p1.preprocessing import load_and_preprocess

    X_train_sc, X_test_sc, y_train, y_test, feature_names = load_and_preprocess()
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Build the path to the CSV relative to this file's location.
# This works regardless of where the function is called from.
# src/p1/preprocessing.py -> go up two levels -> FoDS/ -> data/
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "thyroid_cancer_risk_data.csv")


def load_and_preprocess(data_path=DEFAULT_DATA_PATH, random_state=42, test_size=0.2, verbose=True):
    """
    Loads the thyroid cancer dataset and runs the full preprocessing pipeline.

    Steps performed:
        1.  Load CSV and drop missing values
        2.  Remove outliers using Z-score (|Z| > 3) on continuous columns
        3.  Drop Patient_ID, Country, Ethnicity, and Thyroid_Cancer_Risk
        4.  Encode binary Yes/No columns as 0/1
        5.  Encode Gender as 0/1
        6.  Encode target variable Diagnosis as 0 (Benign) / 1 (Malignant)
        7.  Stratified 80/20 train/test split
        8.  StandardScaler on continuous features (fit on training data only)

    Parameters
    ----------
    data_path    : str   — path to thyroid_cancer_risk_data.csv
                           defaults to ../../data/ relative to this file
    random_state : int   — random seed for reproducibility (default 42)
    test_size    : float — fraction of data used for test set (default 0.2)
    verbose      : bool  — if True, prints a summary of each step

    Returns
    -------
    X_train_sc   : pd.DataFrame — scaled training features
    X_test_sc    : pd.DataFrame — scaled test features
    y_train      : pd.Series    — training labels (0=Benign, 1=Malignant)
    y_test       : pd.Series    — test labels (0=Benign, 1=Malignant)
    feature_names: list         — names of the features in X_train_sc / X_test_sc
    """

    # ── Column definitions ────────────────────────────────────────────────────
    continuous_cols = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
    binary_cols     = [
        "Family_History", "Radiation_Exposure", "Iodine_Deficiency",
        "Smoking", "Obesity", "Diabetes"
    ]

    # ── Step 1: Load CSV and drop missing values ──────────────────────────────
    # Missing values are only 12 rows out of 212,691 (0.006%) so dropping
    # is safe and has no impact on class balance or dataset size.
    df               = pd.read_csv(data_path)
    rows_before_null = len(df)
    df.dropna(inplace=True)

    if verbose:
        print(f"[1] Loaded data:         {rows_before_null:,} rows")
        print(f"    After dropping nulls: {len(df):,} rows "
              f"({rows_before_null - len(df)} rows dropped)")

    # ── Step 2: Remove outliers using Z-score (|Z| > 3) ──────────────────────
    # IQR bounds are too wide at n=212k to catch extreme values like
    # TSH=2000 or T4=200. Z-score is more sensitive to individual extremes.
    # Any row where at least one continuous column has |Z| > 3 is removed.
    z          = df[continuous_cols].apply(zscore)
    mask_clean = (z.abs() <= 3).all(axis=1)
    df_clean   = df[mask_clean].copy()

    if verbose:
        print(f"\n[2] Outlier removal (Z-score |Z| > 3):")
        print(f"    Rows before: {len(df):,}")
        print(f"    Rows after:  {len(df_clean):,} "
              f"({len(df) - len(df_clean):,} rows removed, "
              f"{(1 - len(df_clean)/len(df))*100:.2f}%)")

    # ── Step 3: Drop columns not used in modelling ────────────────────────────
    # Patient_ID          → row number, no predictive value
    # Country & Ethnicity → weak Cramér's V (0.11 / 0.16), confounded signal
    # Thyroid_Cancer_Risk → correlation with Diagnosis r=0.37, too strong —
    #                       keeping it lets models copy it rather than learn
    df_model = df_clean.copy()
    df_model.drop(
        columns=["Patient_ID", "Country", "Ethnicity", "Thyroid_Cancer_Risk"],
        inplace=True
    )

    if verbose:
        print(f"\n[3] Dropped: Patient_ID, Country, Ethnicity, Thyroid_Cancer_Risk")

    # ── Step 4: Encode binary Yes/No columns as 0/1 ──────────────────────────
    for col in binary_cols:
        df_model[col] = (df_model[col] == "Yes").astype(int)

    # ── Step 5: Encode Gender as 0/1 ─────────────────────────────────────────
    # Male → 1, Female → 0
    df_model["Gender"] = (df_model["Gender"] == "Male").astype(int)

    if verbose:
        print(f"[4] Encoded binary columns and Gender as 0/1")

    # ── Step 6: Encode target variable ───────────────────────────────────────
    # Malignant → 1, Benign → 0
    df_model["Diagnosis"] = (df_model["Diagnosis"] == "Malignant").astype(int)

    if verbose:
        print(f"[5] Encoded Diagnosis: Benign=0, Malignant=1")

    # ── Step 7: Separate features (X) and target (y) ─────────────────────────
    X             = df_model.drop(columns=["Diagnosis"])
    y             = df_model["Diagnosis"]
    feature_names = X.columns.tolist()

    if verbose:
        print(f"\n[6] Features ({len(feature_names)}): {feature_names}")
        print(f"    Class balance — "
              f"Benign: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)  "
              f"Malignant: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    # ── Step 8: Stratified 80/20 train/test split ─────────────────────────────
    # stratify=y ensures both splits preserve the same 23% Malignant ratio
    # as the full dataset, preventing class imbalance from skewing results.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if verbose:
        print(f"\n[7] Train/test split (stratified {int((1-test_size)*100)}/{int(test_size*100)}):")
        print(f"    Train: {len(X_train):,} rows — "
              f"Malignant rate: {y_train.mean()*100:.1f}%")
        print(f"    Test:  {len(X_test):,}  rows — "
              f"Malignant rate: {y_test.mean()*100:.1f}%")

    # ── Step 9: StandardScaler on continuous features ─────────────────────────
    # Fit on training data ONLY — using test data to fit the scaler would
    # be data leakage, giving artificially good results that won't hold
    # in the real world.
    scaler    = StandardScaler()
    X_train_sc = X_train.copy()
    X_test_sc  = X_test.copy()

    # Cast to float first — pandas won't store scaled decimals in int columns
    X_train_sc[continuous_cols] = X_train_sc[continuous_cols].astype(float)
    X_test_sc[continuous_cols]  = X_test_sc[continuous_cols].astype(float)

    # fit_transform on train: learns mean/std AND scales in one step
    # transform on test:      uses the mean/std from training ONLY
    X_train_sc[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_sc[continuous_cols]  = scaler.transform(X_test[continuous_cols])

    if verbose:
        print(f"\n[8] StandardScaler applied to: {continuous_cols}")
        print(f"    Scaler fitted on training data only — no data leakage")
        print(f"\n✅ Preprocessing complete. Ready for modelling.")

    return X_train_sc, X_test_sc, y_train, y_test, feature_names