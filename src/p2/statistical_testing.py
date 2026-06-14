"""
Statistical Testing

This script runs the statistical analysis for the project on the cleaned dataset
provided by the shared preprocessing pipeline from P1.

Tests included:
    1. Chi-square tests for categorical variables vs. Diagnosis
    2. Cramer's V as effect size for categorical associations
    3. Mann-Whitney U tests for continuous variables vs. Diagnosis
    4. Rank-biserial correlation as effect size for group differences
    5. Benjamini-Hochberg FDR correction for multiple testing
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels not installed — FDR correction will be skipped.")
    print("Install with: pip install statsmodels")

from src.p1.preprocessing import load_raw, split_then_filter_outliers


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = "data/thyroid_cancer_risk_data.csv"
OUTPUT_DIR = "outputs/p2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTINUOUS_COLS = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]

BINARY_COLS = [
    "Family_History",
    "Radiation_Exposure",
    "Iodine_Deficiency",
    "Smoking",
    "Obesity",
    "Diabetes"
]

CATEGORICAL_FEATURES = BINARY_COLS + ["Gender", "Thyroid_Cancer_Risk"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cramers_v_from_table(contingency_table: pd.DataFrame) -> float:
    """Compute Cramer's V from a contingency table."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.to_numpy().sum()
    min_dim = min(contingency_table.shape) - 1

    if n == 0 or min_dim <= 0:
        return np.nan

    return np.sqrt(chi2 / (n * min_dim))


def interpret_cramers_v(v: float) -> str:
    """Interpret Cramer's V magnitude."""
    if pd.isna(v):
        return "NA"
    if v < 0.10:
        return "negligible"
    if v < 0.30:
        return "small"
    if v < 0.50:
        return "medium"
    return "large"


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    """
    Compute rank-biserial correlation from Mann-Whitney U.
    Negative sign means the second group tends to have larger values
    if group order is Benign first, Malignant second.
    """
    if n1 == 0 or n2 == 0:
        return np.nan
    return 1 - (2 * u_stat) / (n1 * n2)


def interpret_rank_biserial(r: float) -> str:
    """Interpret rank-biserial effect size by absolute magnitude."""
    if pd.isna(r):
        return "NA"
    abs_r = abs(r)
    if abs_r < 0.10:
        return "negligible"
    if abs_r < 0.30:
        return "small"
    if abs_r < 0.50:
        return "medium"
    return "large"


def apply_fdr_correction(df_results: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction if statsmodels is available."""
    df_results = df_results.copy()
    df_results["p_adj"] = np.nan
    df_results["Significant_raw"] = df_results[p_col] < 0.05
    df_results["Significant_FDR"] = np.nan

    if STATSMODELS_AVAILABLE:
        valid_mask = df_results[p_col].notna()
        if valid_mask.sum() > 0:
            df_results.loc[valid_mask, "p_adj"] = multipletests(
                df_results.loc[valid_mask, p_col],
                method="fdr_bh"
            )[1]
            df_results["Significant_FDR"] = df_results["p_adj"] < 0.05

    return df_results


# =============================================================================
# LOAD CLEANED DATA
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3 — STATISTICAL TESTING")
print("=" * 70)


df_raw = load_raw(DATA_PATH)
split = split_then_filter_outliers(
    df_raw,
    random_state=42,
    test_size=0.2,
)

df_train = split["X_train"].copy()
df_train["Diagnosis"] = split["y_train"]

df_test = split["X_test"].copy()
df_test["Diagnosis"] = split["y_test"]

df_clean = pd.concat([df_train, df_test], axis=0)

# Encoding variables for statistical testing
for col in BINARY_COLS:
    df_clean[col] = (df_clean[col] == "Yes").astype(int)

df_clean["Gender"] = (df_clean["Gender"] == "Male").astype(int)


print("\nUsing P1 preprocessing basis:")
print(f"Train rows after outlier filtering: {len(split['X_train']):,}")
print(f"Test rows, unfiltered: {len(split['X_test']):,}")

print("\nCleaned dataset ready for statistics:")
print(f"Rows used for stats: {len(df_clean):,}")

benign = df_clean[df_clean["Diagnosis"] == 0]
malignant = df_clean[df_clean["Diagnosis"] == 1]

print(f"Benign cases:    {len(benign):,}")
print(f"Malignant cases: {len(malignant):,}")

print("\nCategorical features tested:")
print(CATEGORICAL_FEATURES)

print("\nContinuous features tested:")
print(CONTINUOUS_COLS)


# =============================================================================
# 3A. CHI-SQUARE TESTS + CRAMER'S V
# =============================================================================

print("\n" + "-" * 70)
print("3A. CHI-SQUARE TESTS + CRAMER'S V")
print("-" * 70)

chi_results = []

for col in CATEGORICAL_FEATURES:
    df_test = df_clean[[col, "Diagnosis"]].dropna().copy()
    contingency = pd.crosstab(df_test[col], df_test["Diagnosis"])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        chi_results.append({
            "Feature": col,
            "Chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "Cramers_V": np.nan,
            "Min_expected": np.nan,
            "Interpretation": "Test not possible"
        })
        continue

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    v = cramers_v_from_table(contingency)

    chi_results.append({
        "Feature": col,
        "Chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "Cramers_V": v,
        "Min_expected": expected.min(),
        "Interpretation": interpret_cramers_v(v)
    })

chi_results_df = pd.DataFrame(chi_results)
chi_results_df = apply_fdr_correction(chi_results_df, p_col="p_value")
chi_results_df = chi_results_df.sort_values("Cramers_V", ascending=False)

print("\nChi-square results (sorted by Cramer's V):")
print(
    chi_results_df[
        [
            "Feature", "Chi2", "p_value", "p_adj", "Cramers_V",
            "Interpretation", "Min_expected", "Significant_FDR"
        ]
    ].round(4).to_string(index=False)
)


# =============================================================================
# 3B. MANN-WHITNEY U TESTS + RANK-BISERIAL CORRELATION
# =============================================================================

print("\n" + "-" * 70)
print("3B. MANN-WHITNEY U TESTS + RANK-BISERIAL CORRELATION")
print("-" * 70)

mw_results = []

for col in CONTINUOUS_COLS:
    df_test = df_clean[[col, "Diagnosis"]].dropna().copy()

    benign_vals = df_test.loc[df_test["Diagnosis"] == 0, col]
    malignant_vals = df_test.loc[df_test["Diagnosis"] == 1, col]

    if len(benign_vals) == 0 or len(malignant_vals) == 0:
        mw_results.append({
            "Feature": col,
            "Benign_median": np.nan,
            "Malignant_median": np.nan,
            "U_stat": np.nan,
            "p_value": np.nan,
            "Rank_biserial_r": np.nan,
            "Interpretation": "Test not possible"
        })
        continue

    u_stat, p_value = stats.mannwhitneyu(
        benign_vals,
        malignant_vals,
        alternative="two-sided"
    )

    r_rb = rank_biserial_from_u(u_stat, len(benign_vals), len(malignant_vals))

    mw_results.append({
        "Feature": col,
        "Benign_median": benign_vals.median(),
        "Malignant_median": malignant_vals.median(),
        "Benign_mean": benign_vals.mean(),
        "Malignant_mean": malignant_vals.mean(),
        "U_stat": u_stat,
        "p_value": p_value,
        "Rank_biserial_r": r_rb,
        "Interpretation": interpret_rank_biserial(r_rb)
    })

mw_results_df = pd.DataFrame(mw_results)
mw_results_df = apply_fdr_correction(mw_results_df, p_col="p_value")
mw_results_df = mw_results_df.reindex(
    mw_results_df["Rank_biserial_r"].abs().sort_values(ascending=False).index
)

print("\nMann-Whitney U results (sorted by |rank-biserial r|):")
print(
    mw_results_df[
        [
            "Feature", "Benign_median", "Malignant_median",
            "Benign_mean", "Malignant_mean",
            "U_stat", "p_value", "p_adj",
            "Rank_biserial_r", "Interpretation", "Significant_FDR"
        ]
    ].round(4).to_string(index=False)
)


# =============================================================================
# 3C. QUICK SUMMARY
# =============================================================================

print("\n" + "-" * 70)
print("3C. QUICK SUMMARY FOR REPORT WRITING")
print("-" * 70)

if not chi_results_df.empty:
    print("\nTop categorical effects:")
    print(
        chi_results_df[
            ["Feature", "Cramers_V", "Interpretation", "Significant_FDR"]
        ].head(5).round(4).to_string(index=False)
    )

    near_zero_cat = chi_results_df[chi_results_df["Cramers_V"] < 0.10]
    if not near_zero_cat.empty:
        print("\nCategorical variables with negligible effect sizes:")
        print(near_zero_cat["Feature"].to_list())

if not mw_results_df.empty:
    print("\nTop continuous effects:")
    print(
        mw_results_df[
            ["Feature", "Rank_biserial_r", "Interpretation", "Significant_FDR"]
        ].head(5).round(4).to_string(index=False)
    )


# =============================================================================
# 3D. SAVE OUTPUT TABLES
# =============================================================================

chi_path = os.path.join(OUTPUT_DIR, "chi_square_results.csv")
mw_path = os.path.join(OUTPUT_DIR, "mann_whitney_results.csv")

chi_results_df.to_csv(chi_path, index=False)
mw_results_df.to_csv(mw_path, index=False)

print("\nSaved output files:")
print(f" - {chi_path}")
print(f" - {mw_path}")
print("\nDone.")
