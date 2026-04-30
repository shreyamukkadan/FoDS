import numpy as np
import pandas as pd
from scipy import stats

# Optional multiple-testing correction
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("statsmodels not installed — p-value adjustment skipped")
    print("Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
DATA_PATH = "data/thyroid_cancer_risk_data.csv"
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# FEATURE DEFINITIONS
# ---------------------------------------------------------------------------
continuous_cols = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]

binary_cols = [
    "Family_History",
    "Radiation_Exposure",
    "Iodine_Deficiency",
    "Smoking",
    "Obesity",
    "Diabetes"
]

# =============================================================================
# 3. STATISTICAL TESTING
# =============================================================================

print("\n" + "=" * 65)
print("STATISTICAL TESTING")

# Optional multiple-testing correction
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("statsmodels not installed — p-value adjustment skipped")
    print("Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Define feature groups
# ---------------------------------------------------------------------------
# Binary/categorical features from your earlier code
categorical_features = binary_cols + ["Gender", "Thyroid_Cancer_Risk"]

# Continuous features from your earlier code
continuous_features = continuous_cols

# Remove duplicates while preserving order
categorical_features = list(dict.fromkeys(categorical_features))
continuous_features = list(dict.fromkeys(continuous_features))

# Split original dataframe by diagnosis
benign = df[df["Diagnosis"] == "Benign"]
malignant = df[df["Diagnosis"] == "Malignant"]

print("\nCategorical features tested:")
print(categorical_features)

print("\nContinuous features tested:")
print(continuous_features)

# =============================================================================
# 3A. CHI-SQUARE TESTS FOR CATEGORICAL FEATURES
# =============================================================================

print("\n--- Chi-square tests + Cramér's V ---")

chi_results = []

for col in categorical_features:
    df_test = df[[col, "Diagnosis"]].dropna().copy()

    # contingency table
    ct = pd.crosstab(df_test[col], df_test["Diagnosis"])

    # skip if table is invalid
    if ct.shape[0] < 2 or ct.shape[1] < 2:
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

    chi2, p, dof, expected = stats.chi2_contingency(ct)

    # Cramér's V
    n = ct.to_numpy().sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

    # Effect size interpretation
    if pd.isna(cramers_v):
        interpretation = "NA"
    elif cramers_v < 0.10:
        interpretation = "negligible"
    elif cramers_v < 0.30:
        interpretation = "small"
    elif cramers_v < 0.50:
        interpretation = "medium"
    else:
        interpretation = "large"

    chi_results.append({
        "Feature": col,
        "Chi2": chi2,
        "p_value": p,
        "dof": dof,
        "Cramers_V": cramers_v,
        "Min_expected": expected.min(),
        "Interpretation": interpretation
    })

chi_results_df = pd.DataFrame(chi_results)

# Multiple testing correction
if STATSMODELS_AVAILABLE:
    valid_mask = chi_results_df["p_value"].notna()
    chi_results_df.loc[valid_mask, "p_adj"] = multipletests(
        chi_results_df.loc[valid_mask, "p_value"],
        method="fdr_bh"
    )[1]
else:
    chi_results_df["p_adj"] = np.nan

chi_results_df["Significant_raw"] = chi_results_df["p_value"] < 0.05
chi_results_df["Significant_FDR"] = chi_results_df["p_adj"] < 0.05 if STATSMODELS_AVAILABLE else np.nan

# Sort by strongest effect size
chi_results_df = chi_results_df.sort_values("Cramers_V", ascending=False)

print("\nChi-square results (sorted by Cramér's V):")
print(
    chi_results_df[
        ["Feature", "Chi2", "p_value", "p_adj", "Cramers_V",
         "Interpretation", "Min_expected", "Significant_FDR"]
    ].round(4).to_string(index=False)
)

# =============================================================================
# 3B. MANN-WHITNEY U TESTS FOR CONTINUOUS FEATURES
# =============================================================================

print("\n--- Mann-Whitney U tests + rank-biserial correlation ---")

mw_results = []

for col in continuous_features:
    df_test = df[[col, "Diagnosis"]].dropna().copy()

    benign_vals = df_test[df_test["Diagnosis"] == "Benign"][col]
    malignant_vals = df_test[df_test["Diagnosis"] == "Malignant"][col]

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

    # two-sided Mann-Whitney U test
    u_stat, p = stats.mannwhitneyu(
        benign_vals,
        malignant_vals,
        alternative="two-sided"
    )

    n1 = len(benign_vals)
    n2 = len(malignant_vals)

    # Rank-biserial correlation
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    # Interpret using absolute effect size
    abs_r = abs(rank_biserial)
    if abs_r < 0.10:
        interpretation = "negligible"
    elif abs_r < 0.30:
        interpretation = "small"
    elif abs_r < 0.50:
        interpretation = "medium"
    else:
        interpretation = "large"

    mw_results.append({
        "Feature": col,
        "Benign_median": benign_vals.median(),
        "Malignant_median": malignant_vals.median(),
        "U_stat": u_stat,
        "p_value": p,
        "Rank_biserial_r": rank_biserial,
        "Interpretation": interpretation
    })

mw_results_df = pd.DataFrame(mw_results)

# Multiple testing correction
if STATSMODELS_AVAILABLE:
    valid_mask = mw_results_df["p_value"].notna()
    mw_results_df.loc[valid_mask, "p_adj"] = multipletests(
        mw_results_df.loc[valid_mask, "p_value"],
        method="fdr_bh"
    )[1]
else:
    mw_results_df["p_adj"] = np.nan

mw_results_df["Significant_raw"] = mw_results_df["p_value"] < 0.05
mw_results_df["Significant_FDR"] = mw_results_df["p_adj"] < 0.05 if STATSMODELS_AVAILABLE else np.nan

# Sort by absolute effect size
mw_results_df = mw_results_df.reindex(
    mw_results_df["Rank_biserial_r"].abs().sort_values(ascending=False).index
)

print("\nMann-Whitney U results (sorted by |rank-biserial r|):")
print(
    mw_results_df[
        ["Feature", "Benign_median", "Malignant_median", "U_stat", "p_value",
         "p_adj", "Rank_biserial_r", "Interpretation", "Significant_FDR"]
    ].round(4).to_string(index=False)
)

# =============================================================================
# 3C. OPTIONAL SUMMARY FOR REPORT WRITING
# =============================================================================

print("\n--- Quick summary for interpretation ---")

if not chi_results_df.empty:
    top_chi = chi_results_df[["Feature", "Cramers_V", "Interpretation"]].head(5)
    print("\nTop categorical effects:")
    print(top_chi.round(4).to_string(index=False))

if not mw_results_df.empty:
    top_mw = mw_results_df[["Feature", "Rank_biserial_r", "Interpretation"]].head(5)
    print("\nTop continuous effects:")
    print(top_mw.round(4).to_string(index=False))

# =============================================================================
# 3D. SAVE OUTPUT TABLES
# =============================================================================

chi_results_df.to_csv("chi_square_results.csv", index=False)
mw_results_df.to_csv("mann_whitney_results.csv", index=False)

print("\nSaved:")
print(" - chi_square_results.csv")
print(" - mann_whitney_results.csv")