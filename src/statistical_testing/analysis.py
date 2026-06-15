"""Run statistical association tests on the preprocessed dataset."""

from typing import Optional

import os

from config import DATA_PATH, OUT_STATISTICAL_TESTING, RANDOM_STATE, TEST_SIZE

os.environ.setdefault("MPLCONFIGDIR", str(OUT_STATISTICAL_TESTING.parent.parent / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing.pipeline import load_raw, split_then_filter_outliers

try:
    from statsmodels.stats.multitest import multipletests

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


OUTPUT_DIR = OUT_STATISTICAL_TESTING
PLOT_DIR = OUTPUT_DIR / "plots"

CONTINUOUS_COLS = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
BINARY_COLS = [
    "Family_History",
    "Radiation_Exposure",
    "Iodine_Deficiency",
    "Smoking",
    "Obesity",
    "Diabetes",
]
CATEGORICAL_FEATURES = BINARY_COLS + ["Gender", "Thyroid_Cancer_Risk", "Country", "Ethnicity"]


def cramers_v_from_table(contingency_table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.to_numpy().sum()
    min_dim = min(contingency_table.shape) - 1
    if n == 0 or min_dim <= 0:
        return np.nan
    return np.sqrt(chi2 / (n * min_dim))


def interpret_cramers_v(v: float) -> str:
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
    if n1 == 0 or n2 == 0:
        return np.nan
    return 1 - (2 * u_stat) / (n1 * n2)


def interpret_rank_biserial(r: float) -> str:
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
    df_results = df_results.copy()
    df_results["p_adj"] = np.nan
    df_results["Significant_raw"] = df_results[p_col] < 0.05
    df_results["Significant_FDR"] = np.nan

    if STATSMODELS_AVAILABLE:
        valid_mask = df_results[p_col].notna()
        if valid_mask.sum() > 0:
            df_results.loc[valid_mask, "p_adj"] = multipletests(
                df_results.loc[valid_mask, p_col],
                method="fdr_bh",
            )[1]
            df_results["Significant_FDR"] = df_results["p_adj"] < 0.05

    return df_results


def build_analysis_table() -> tuple[pd.DataFrame, dict]:
    df_raw = load_raw(DATA_PATH)
    split = split_then_filter_outliers(
        df_raw,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
    )

    df_train = split["X_train"].copy()
    df_train["Diagnosis"] = split["y_train"]
    df_test = split["X_test"].copy()
    df_test["Diagnosis"] = split["y_test"]
    df_clean = pd.concat([df_train, df_test], axis=0)

    for col in BINARY_COLS:
        df_clean[col] = (df_clean[col] == "Yes").astype(int)
    df_clean["Gender"] = (df_clean["Gender"] == "Male").astype(int)

    return df_clean, split


def run_chi_square_tests(df_clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_FEATURES:
        test_df = df_clean[[col, "Diagnosis"]].dropna().copy()
        contingency = pd.crosstab(test_df[col], test_df["Diagnosis"])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            rows.append({
                "Feature": col,
                "Chi2": np.nan,
                "p_value": np.nan,
                "dof": np.nan,
                "Cramers_V": np.nan,
                "Min_expected": np.nan,
                "Interpretation": "Test not possible",
            })
            continue

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        v = cramers_v_from_table(contingency)
        rows.append({
            "Feature": col,
            "Chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "Cramers_V": v,
            "Min_expected": expected.min(),
            "Interpretation": interpret_cramers_v(v),
        })

    results = pd.DataFrame(rows)
    results = apply_fdr_correction(results, p_col="p_value")
    return results.sort_values("Cramers_V", ascending=False)


def run_mann_whitney_tests(df_clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CONTINUOUS_COLS:
        test_df = df_clean[[col, "Diagnosis"]].dropna().copy()
        benign_vals = test_df.loc[test_df["Diagnosis"] == 0, col]
        malignant_vals = test_df.loc[test_df["Diagnosis"] == 1, col]

        if len(benign_vals) == 0 or len(malignant_vals) == 0:
            rows.append({
                "Feature": col,
                "Benign_median": np.nan,
                "Malignant_median": np.nan,
                "U_stat": np.nan,
                "p_value": np.nan,
                "Rank_biserial_r": np.nan,
                "Interpretation": "Test not possible",
            })
            continue

        u_stat, p_value = stats.mannwhitneyu(benign_vals, malignant_vals, alternative="two-sided")
        r_rb = rank_biserial_from_u(u_stat, len(benign_vals), len(malignant_vals))
        rows.append({
            "Feature": col,
            "Benign_median": benign_vals.median(),
            "Malignant_median": malignant_vals.median(),
            "Benign_mean": benign_vals.mean(),
            "Malignant_mean": malignant_vals.mean(),
            "U_stat": u_stat,
            "p_value": p_value,
            "Rank_biserial_r": r_rb,
            "Interpretation": interpret_rank_biserial(r_rb),
        })

    results = pd.DataFrame(rows)
    results = apply_fdr_correction(results, p_col="p_value")
    return results.reindex(results["Rank_biserial_r"].abs().sort_values(ascending=False).index)


def plot_malignancy_rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    filename: str,
    top_n: Optional[int] = None,
) -> None:
    plot_df = df.copy()
    if top_n is not None:
        largest_groups = plot_df[group_col].value_counts().head(top_n).index
        plot_df = plot_df[plot_df[group_col].isin(largest_groups)]

    rates = plot_df.groupby(group_col)["Diagnosis"].mean().sort_values(ascending=True)
    overall_rate = plot_df["Diagnosis"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(rates.index.astype(str), rates.values)
    ax.axvline(overall_rate, linestyle="--", linewidth=1)
    ax.set_xlabel("Malignancy rate")
    ax.set_title(title)
    fig.tight_layout()

    save_path = PLOT_DIR / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def save_plots(df_clean: pd.DataFrame) -> None:
    plot_malignancy_rate_by_group(
        df_clean,
        "Thyroid_Cancer_Risk",
        "Malignancy rate by Thyroid Cancer Risk",
        "malignancy_by_thyroid_cancer_risk.png",
    )
    plot_malignancy_rate_by_group(
        df_clean,
        "Country",
        "Malignancy rate by Country (largest groups)",
        "malignancy_by_country.png",
        top_n=10,
    )
    plot_malignancy_rate_by_group(
        df_clean,
        "Ethnicity",
        "Malignancy rate by Ethnicity",
        "malignancy_by_ethnicity.png",
    )
    plot_malignancy_rate_by_group(
        df_clean,
        "Family_History",
        "Malignancy rate by Family History",
        "malignancy_by_family_history.png",
    )
    plot_malignancy_rate_by_group(
        df_clean,
        "Radiation_Exposure",
        "Malignancy rate by Radiation Exposure",
        "malignancy_by_radiation_exposure.png",
    )
    plot_malignancy_rate_by_group(
        df_clean,
        "Iodine_Deficiency",
        "Malignancy rate by Iodine Deficiency",
        "malignancy_by_iodine_deficiency.png",
    )


def print_summary(df_clean: pd.DataFrame, split: dict, chi_results_df: pd.DataFrame, mw_results_df: pd.DataFrame) -> None:
    benign = df_clean[df_clean["Diagnosis"] == 0]
    malignant = df_clean[df_clean["Diagnosis"] == 1]

    print("\nUsing preprocessing pipeline basis:")
    print(f"Train rows after outlier filtering: {len(split['X_train']):,}")
    print(f"Test rows, unfiltered: {len(split['X_test']):,}")
    print("\nCleaned dataset ready for statistics:")
    print(f"Rows used for stats: {len(df_clean):,}")
    print(f"Benign cases:    {len(benign):,}")
    print(f"Malignant cases: {len(malignant):,}")
    print("\nCategorical features tested:")
    print(CATEGORICAL_FEATURES)
    print("\nContinuous features tested:")
    print(CONTINUOUS_COLS)

    print("\nChi-square results (sorted by Cramer's V):")
    print(
        chi_results_df[
            [
                "Feature", "Chi2", "p_value", "p_adj", "Cramers_V",
                "Interpretation", "Min_expected", "Significant_FDR",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nMann-Whitney U results (sorted by |rank-biserial r|):")
    print(
        mw_results_df[
            [
                "Feature", "Benign_median", "Malignant_median",
                "Benign_mean", "Malignant_mean", "U_stat", "p_value", "p_adj",
                "Rank_biserial_r", "Interpretation", "Significant_FDR",
            ]
        ].round(4).to_string(index=False)
    )

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

    print("\nTop continuous effects:")
    print(
        mw_results_df[
            ["Feature", "Rank_biserial_r", "Interpretation", "Significant_FDR"]
        ].head(5).round(4).to_string(index=False)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not STATSMODELS_AVAILABLE:
        print("statsmodels not installed; FDR correction will be skipped.")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTING")
    print("=" * 70)

    df_clean, split = build_analysis_table()
    chi_results_df = run_chi_square_tests(df_clean)
    mw_results_df = run_mann_whitney_tests(df_clean)

    print_summary(df_clean, split, chi_results_df, mw_results_df)

    chi_path = OUTPUT_DIR / "chi_square_results.csv"
    mw_path = OUTPUT_DIR / "mann_whitney_results.csv"
    chi_results_df.to_csv(chi_path, index=False)
    mw_results_df.to_csv(mw_path, index=False)
    save_plots(df_clean)

    print("\nSaved output files:")
    print(f" - {chi_path}")
    print(f" - {mw_path}")
    print(f" - {PLOT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
