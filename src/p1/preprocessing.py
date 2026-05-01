"""
src/preprocessing.py
Thyroid Cancer Risk — Foundations of Data Science Group Project
 
This file contains the preprocessing pipeline as a clean, importable function.
It is used by all group members so everyone works with identical cleaned data.
 
Usage (in any notebook):
    from src.preprocessing import load_and_preprocess
 
    X_train_sc, X_test_sc, y_train, y_test, feature_names = load_and_preprocess(
        data_path="../data/thyroid_cancer_risk_data.csv"
    )
"""
 
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
 
def load_and_preprocess(data_path, random_state=42, test_size=0.2, verbose=True):
    """
    Loads the thyroid cancer dataset and runs the full preprocessing pipeline.
 
    Steps performed:
        1. Load CSV and drop missing values
        2. Remove outliers using Z-score (|Z| > 3) on continuous columns
        3. Encode binary Yes/No columns as 0/1
        4. Encode Gender as 0/1
        5. Drop Patient_ID, Country, Ethnicity, and Thyroid_Cancer_Risk
           (Thyroid_Cancer_Risk excluded: correlation with Diagnosis r=0.37,
            too strong — would let models cheat)
        6. Encode target variable Diagnosis as 0 (Benign) / 1 (Malignant)
        7. Stratified 80/20 train/test split
        8. StandardScaler on continuous features (fit on training data only)
 
    Parameters
    ----------
    data_path    : str   — path to thyroid_cancer_risk_data.csv
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
 
    # ── Columns we will work with ─────────────────────────────────────────────
    continuous_cols = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
    binary_cols     = [
        "Family_History", "Radiation_Exposure", "Iodine_Deficiency",
        "Smoking", "Obesity", "Diabetes"
    ]
 
    # ── Step 1: Load data & drop missing values ───────────────────────────────
    df = pd.read_csv(data_path)
    rows_before_null = len(df)
    df.dropna(inplace=True)
 
    if verbose:
        print(f"[1] Loaded data:         {rows_before_null:,} rows")
        print(f"    After dropping nulls: {len(df):,} rows "
              f"({rows_before_null - len(df)} rows dropped)")
 
    # ── Step 2: Remove outliers using Z-score (|Z| > 3) ──────────────────────
    # We use Z-score rather than IQR because the dataset is very large (212k
    # rows), which makes IQR bounds too wide to catch extreme values like
    # TSH=2000 or T4=200. Z-score is more sensitive to individual extremes.
    # Any row where at least one continuous column has |Z| > 3 is removed.
    z          = df[continuous_cols].apply(zscore)
    mask_clean = (z.abs() <= 3).all(axis=1)
    df_clean   = df[mask_clean].copy()
 
    if verbose:
        print(f"\n[2] Outlier removal (Z-score |Z|>3):")
        print(f"    Rows before: {len(df):,}")
        print(f"    Rows after:  {len(df_clean):,} "
              f"({len(df) - len(df_clean):,} rows removed, "
              f"{(1 - len(df_clean)/len(df))*100:.2f}%)")
 
    # ── Step 3: Drop columns not used in modelling ───────────────────────────
    # Patient_ID   → just a row number, no predictive value
    # Country      → high cardinality, no geographic signal in synthetic data
    # Ethnicity    → same as Country
    # Thyroid_Cancer_Risk → excluded because its correlation with Diagnosis
    #                       (r=0.37) is too strong; keeping it would let the
    #                       model essentially copy this column rather than
    #                       learning from genuine clinical features
    df_model = df_clean.copy()
    df_model.drop(
        columns=["Patient_ID", "Country", "Ethnicity", "Thyroid_Cancer_Risk"],
        inplace=True
    )
 
    if verbose:
        print(f"\n[3] Dropped columns: Patient_ID, Country, Ethnicity, "
              f"Thyroid_Cancer_Risk")
 
    # ── Step 4: Encode binary Yes/No columns as 0/1 ──────────────────────────
    for col in binary_cols:
        df_model[col] = (df_model[col] == "Yes").astype(int)
 
    # ── Step 5: Encode Gender as 0/1 ─────────────────────────────────────────
    df_model["Gender"] = (df_model["Gender"] == "Male").astype(int)
 
    if verbose:
        print(f"[4] Encoded binary columns and Gender as 0/1")
 
    # ── Step 6: Encode target variable ───────────────────────────────────────
    # Malignant → 1, Benign → 0
    df_model["Diagnosis"] = (df_model["Diagnosis"] == "Malignant").astype(int)
 
    if verbose:
        print(f"[5] Encoded target: Benign=0, Malignant=1")
 
    # ── Step 7: Separate features and target ─────────────────────────────────
    X             = df_model.drop(columns=["Diagnosis"])
    y             = df_model["Diagnosis"]
    feature_names = X.columns.tolist()
 
    if verbose:
        print(f"\n[6] Features ({len(feature_names)}): {feature_names}")
        print(f"    Class balance — "
              f"Benign: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)  "
              f"Malignant: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
 
    # ── Step 8: Stratified train/test split ───────────────────────────────────
    # stratify=y ensures both splits have the same Benign/Malignant ratio
    # as the full dataset, preventing imbalance from skewing results.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
 
    if verbose:
        print(f"\n[7] Train/test split (stratified, {int((1-test_size)*100)}/{int(test_size*100)}):")
        print(f"    Train: {len(X_train):,} rows — "
              f"Malignant rate: {y_train.mean()*100:.1f}%")
        print(f"    Test:  {len(X_test):,}  rows — "
              f"Malignant rate: {y_test.mean()*100:.1f}%")
 
    # ── Step 9: StandardScaler on continuous features ─────────────────────────
    # We fit the scaler ONLY on the training data, then use it to transform
    # both training and test sets. Fitting on test data or the full dataset
    # would be data leakage — the model would indirectly "see" test values
    # during training, giving artificially good results.
    scaler     = StandardScaler()
 
    X_train_sc = X_train.copy()
    X_test_sc  = X_test.copy()
 
    # Cast to float first — pandas won't store scaled decimals in int columns
    X_train_sc[continuous_cols] = X_train_sc[continuous_cols].astype(float)
    X_test_sc[continuous_cols]  = X_test_sc[continuous_cols].astype(float)
 
    X_train_sc[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_sc[continuous_cols]  = scaler.transform(X_test[continuous_cols])
 
    if verbose:
        print(f"\n[8] StandardScaler applied to: {continuous_cols}")
        print(f"    (Scaler fitted on training data only — no leakage)")
        print(f"\n✅ Preprocessing complete. Ready for modelling.")
 
    return X_train_sc, X_test_sc, y_train, y_test, feature_names
