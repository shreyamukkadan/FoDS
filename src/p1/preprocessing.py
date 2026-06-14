"""
Leakage-safe preprocessing and EDA, revised for the "use all the information"
project direction.

What changed relative to the previous version, and why
------------------------------------------------------
1. The "full" feature set (Country, Ethnicity, Thyroid_Cancer_Risk + everything
   else) is now the primary modelling target. The leakage-safe split that was
   already here is preserved unchanged - it is the part that makes the project
   defensible, so it stays.

2. Thyroid_Cancer_Risk is now ONE-HOT encoded by default instead of ordinal.
   The training-set malignancy rates are Low~=0.15, Medium~=0.15, High~=0.70.
   Ordinal encoding (Low=0, Medium=1, High=2) forces a single monotonic slope
   and tells the model that "Medium" is halfway between "Low" and "High", which
   is false - Medium behaves exactly like Low. One-hot lets the model give High
   its own coefficient and leave Medium near zero, which matches the data.
   Set RISK_ENCODING = "ordinal" in config.py to revert.

3. Subgroups (Country / Ethnicity) are now first-class. There are helper
   functions to list usable groups and to carve a single group out of the EXISTING
   global split (no re-splitting), so a per-group model is trained and tested on
   the same patients the global model never saw. This supports the "separate
   models by country or ethnicity" recommendation without introducing leakage.

4. A per-group diagnostics table (n and malignancy rate per Country/Ethnicity in
   train and test) is written out to support interpretation.

Run from chatgbt_code_final:
    python src/preprocessing.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a Jupyter / VS Code notebook cell, where __file__ is undefined
    _PROJECT_ROOT = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import (
    ABLATION_FEATURE_SETS,
    BINARY,
    CONTINUOUS,
    COUNTRY,
    DATA_PATH,
    ETHNICITY,
    FEATURE_SET_DESCRIPTIONS,
    GENDER,
    ID_COL,
    OUT_P1,
    RANDOM_STATE,
    RISK,
    TARGET,
    TEST_SIZE,
    Z_THRESHOLD,
)

# Optional config overrides; sensible defaults if they are not present in config.py.
try:
    from config import RISK_ENCODING  # "onehot" (default) or "ordinal"
except ImportError:
    RISK_ENCODING = "onehot"

GROUP_COLS = [COUNTRY, ETHNICITY]


def load_raw(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    before = len(df)
    df = df.dropna().copy()
    df.attrs["rows_raw"] = before
    df.attrs["rows_after_dropna"] = len(df)
    return df


def encode_target(df: pd.DataFrame) -> pd.Series:
    return (df[TARGET] == "Malignant").astype(int)


def all_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in [ID_COL, TARGET]]


def select_features(df: pd.DataFrame, feature_set: str) -> List[str]:
    cols = all_features(df)
    if feature_set == "full":
        return cols
    if feature_set == "restricted_original":
        return [c for c in cols if c not in [COUNTRY, ETHNICITY, RISK]]
    if feature_set == "full_without_risk":
        return [c for c in cols if c != RISK]
    if feature_set == "risk_only":
        return [RISK]
    if feature_set == "groups_only":
        return [COUNTRY, ETHNICITY]
    if feature_set == "risk_plus_groups":
        return [RISK, COUNTRY, ETHNICITY]
    if feature_set == "continuous_only":
        return [c for c in CONTINUOUS if c in cols]
    if feature_set == "binary_clinical":
        return [c for c in BINARY + [GENDER] if c in cols]
    raise ValueError(f"Unknown feature set: {feature_set}")


# ---------------------------------------------------------------------------
# Split first, then learn outlier bounds on the training split only.
# (Unchanged: this is the core leakage control.)
# ---------------------------------------------------------------------------
def split_then_filter_outliers(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    z_threshold: float = Z_THRESHOLD,
) -> Dict:
    y = encode_target(df)
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    means = X_train[CONTINUOUS].mean()
    stds = X_train[CONTINUOUS].std(ddof=0).replace(0, np.nan)
    z = (X_train[CONTINUOUS] - means) / stds
    keep = (z.abs() <= z_threshold).all(axis=1).fillna(True)

    bounds = pd.DataFrame(
        {
            "Feature": CONTINUOUS,
            "Train mean": [means[c] for c in CONTINUOUS],
            "Train std": [stds[c] for c in CONTINUOUS],
            "Lower bound": [means[c] - z_threshold * stds[c] for c in CONTINUOUS],
            "Upper bound": [means[c] + z_threshold * stds[c] for c in CONTINUOUS],
        }
    )

    return {
        "X_train": X_train.loc[keep].copy(),
        "X_test": X_test.copy(),
        "y_train": y_train.loc[keep].copy(),
        "y_test": y_test.copy(),
        "n_train_before_outlier_filter": int(len(X_train)),
        "n_train_after_outlier_filter": int(keep.sum()),
        "n_train_outliers_removed": int((~keep).sum()),
        "outlier_bounds": bounds,
    }


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _one_hot_categories(categories: List[List[str]]) -> OneHotEncoder:
    """One-hot with a fixed category order so column names are stable/readable."""
    try:
        return OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(categories=categories, handle_unknown="ignore", sparse=False)


def _ordinal(categories: List[List[str]]) -> OrdinalEncoder:
    try:
        return OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
    except TypeError:
        return OrdinalEncoder(categories=categories)


def build_preprocessor(feature_cols: List[str], risk_encoding: str = RISK_ENCODING) -> ColumnTransformer:
    transformers = []
    cont = [c for c in CONTINUOUS if c in feature_cols]
    binary = [c for c in BINARY if c in feature_cols]
    gender = [GENDER] if GENDER in feature_cols else []
    risk = [RISK] if RISK in feature_cols else []
    groups = [c for c in [COUNTRY, ETHNICITY] if c in feature_cols]

    if cont:
        transformers.append(("continuous", StandardScaler(), cont))
    if binary:
        transformers.append(("binary", _ordinal([["No", "Yes"]] * len(binary)), binary))
    if gender:
        transformers.append(("gender", _ordinal([["Female", "Male"]]), gender))
    if risk:
        if risk_encoding == "ordinal":
            transformers.append(("risk", _ordinal([["Low", "Medium", "High"]]), risk))
        else:
            # Default: one-hot. Low and Medium have near-identical malignancy rates,
            # so a single ordinal slope is misspecified.
            transformers.append(("risk", _one_hot_categories([["Low", "Medium", "High"]]), risk))
    if groups:
        transformers.append(("groups", _one_hot(), groups))
    return ColumnTransformer(transformers, remainder="drop")


def transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Expanded output feature names. Works for ordinal/scaler (pass-through names)
    and one-hot (expanded names) alike."""
    names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names.extend(transformer.get_feature_names_out(cols).tolist())
                continue
            except Exception:
                pass
        names.extend(list(cols))
    return names


# ---------------------------------------------------------------------------
# Subgroup helpers (Country / Ethnicity), reusing the SAME global split.
# ---------------------------------------------------------------------------
def subgroup_feature_cols(df: pd.DataFrame, group_col: str) -> List[str]:
    """Full feature set minus the grouping column itself (it is constant within a group)."""
    return [c for c in select_features(df, "full") if c != group_col]


def group_values(split: Dict, group_col: str, min_train: int = 1, min_test: int = 1) -> List[str]:
    """Group values that have enough rows in BOTH the train and test split to model."""
    train_counts = split["X_train"][group_col].value_counts()
    test_counts = split["X_test"][group_col].value_counts()
    vals = []
    for value in train_counts.index:
        if train_counts.get(value, 0) >= min_train and test_counts.get(value, 0) >= min_test:
            vals.append(value)
    return vals


def subgroup_view(df: pd.DataFrame, split: Dict, group_col: str, value, drop_group: bool = True) -> Dict:
    """
    Carve one group out of the existing global split. The returned test rows are a
    strict subset of the global test set, so a per-group model is comparable to the
    global model and no patient ever moves between train and test.
    """
    cols = subgroup_feature_cols(df, group_col) if drop_group else select_features(df, "full")
    tr_mask = split["X_train"][group_col] == value
    te_mask = split["X_test"][group_col] == value
    return {
        "X_train": split["X_train"].loc[tr_mask, cols].copy(),
        "X_test": split["X_test"].loc[te_mask, cols].copy(),
        "y_train": split["y_train"].loc[tr_mask].copy(),
        "y_test": split["y_test"].loc[te_mask].copy(),
        "feature_cols": cols,
        "group_col": group_col,
        "group_value": value,
    }


def group_diagnostics(split: Dict) -> pd.DataFrame:
    """n and malignancy rate per group value, in train and test, for each group column."""
    rows = []
    for gc in GROUP_COLS:
        for split_name, X, y in [
            ("train", split["X_train"], split["y_train"]),
            ("test", split["X_test"], split["y_test"]),
        ]:
            tmp = X[[gc]].copy()
            tmp["y"] = y  # aligns by index
            agg = tmp.groupby(gc)["y"].agg(n="size", malignant_n="sum").reset_index()
            agg["malignancy_rate"] = agg["malignant_n"] / agg["n"]
            agg = agg.rename(columns={gc: "Value"})
            agg.insert(0, "Split", split_name)
            agg.insert(0, "Group", gc)
            rows.append(agg)
    return pd.concat(rows, ignore_index=True).sort_values(["Group", "Value", "Split"])


# ---------------------------------------------------------------------------
# EDA helpers (unchanged behaviour)
# ---------------------------------------------------------------------------
def malignancy_rates(df: pd.DataFrame, col: str, min_n: int = 1) -> pd.DataFrame:
    tmp = df[[col, TARGET]].copy()
    tmp["Diagnosis_binary"] = encode_target(tmp.rename(columns={TARGET: TARGET}))
    out = (
        tmp.groupby(col, dropna=False)["Diagnosis_binary"]
        .agg(n="size", malignant_n="sum")
        .reset_index()
    )
    out["malignancy_rate"] = out["malignant_n"] / out["n"]
    return out[out["n"] >= min_n].sort_values(["malignancy_rate", "n"], ascending=[False, False])


def _save_bar(rate_df: pd.DataFrame, col: str, path: Path, title: str, max_bars: int = 25) -> None:
    d = rate_df.sort_values("n", ascending=False).head(max_bars).sort_values("malignancy_rate")
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.32 * len(d))))
    ax.barh(d[col].astype(str), d["malignancy_rate"], alpha=0.85)
    ax.axvline(rate_df["malignancy_rate"].mean(), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Malignancy rate")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_eda_outputs(df: pd.DataFrame, out_dir: Path = OUT_P1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = {"Benign": "#4C72B0", "Malignant": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df[TARGET].value_counts().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Diagnosis distribution")
    df[RISK].value_counts().reindex(["Low", "Medium", "High"]).plot(
        kind="bar", ax=axes[1], color=["#55A868", "#E8A838", "#C44E52"]
    )
    axes[1].set_title("Thyroid_Cancer_Risk distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnosis_and_risk_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(CONTINUOUS), figsize=(3.2 * len(CONTINUOUS), 4))
    for ax, col in zip(axes, CONTINUOUS):
        ax.boxplot(df[col].dropna(), showfliers=True)
        ax.set_title(col)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_boxplots_raw.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(CONTINUOUS), figsize=(3.7 * len(CONTINUOUS), 4))
    for ax, col in zip(axes, CONTINUOUS):
        for label, group in df.groupby(TARGET):
            ax.hist(group[col], bins=45, alpha=0.55, density=True, label=label, color=palette[label])
        ax.set_title(col)
        ax.grid(alpha=0.2)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_distributions_by_diagnosis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), BINARY):
        rates = malignancy_rates(df, col)
        rates = rates.sort_values(col)
        ax.bar(rates[col].astype(str), rates["malignancy_rate"], alpha=0.85)
        ax.set_title(col)
        ax.set_ylim(0, max(0.45, rates["malignancy_rate"].max() * 1.15))
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "binary_malignancy_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for col, filename in [
        (RISK, "risk_malignancy_rates.png"),
        (COUNTRY, "country_malignancy_rates.png"),
        (ETHNICITY, "ethnicity_malignancy_rates.png"),
    ]:
        _save_bar(malignancy_rates(df, col), col, out_dir / filename, f"Malignancy rate by {col}")

    numeric = df[CONTINUOUS].copy()
    numeric["Diagnosis_binary"] = encode_target(df)
    numeric["Risk_ordinal"] = df[RISK].map({"Low": 0, "Medium": 1, "High": 2})
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_preprocessing_outputs(df: pd.DataFrame, split: Dict, out_dir: Path = OUT_P1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split["outlier_bounds"].to_csv(out_dir / "train_outlier_bounds.csv", index=False)
    pd.DataFrame(
        [
            {"Split": "full_after_missing_drop", "n": len(df), "malignant_rate": encode_target(df).mean()},
            {
                "Split": "train_before_outlier_filter",
                "n": split["n_train_before_outlier_filter"],
                "malignant_rate": np.nan,
            },
            {"Split": "train_after_outlier_filter", "n": len(split["X_train"]), "malignant_rate": split["y_train"].mean()},
            {"Split": "test_unfiltered", "n": len(split["X_test"]), "malignant_rate": split["y_test"].mean()},
        ]
    ).to_csv(out_dir / "train_test_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "Feature set": fs,
                "n_raw_features": len(select_features(df, fs)),
                "Description": FEATURE_SET_DESCRIPTIONS[fs],
                "Columns": ", ".join(select_features(df, fs)),
            }
            for fs in ABLATION_FEATURE_SETS
        ]
    ).to_csv(out_dir / "feature_sets_summary.csv", index=False)

    # New: per-group counts / malignancy rates in train and test, for interpretation.
    group_diagnostics(split).to_csv(out_dir / "group_malignancy_diagnostics.csv", index=False)

    summary = {
        "rows_raw": int(df.attrs.get("rows_raw", len(df))),
        "rows_after_dropna": int(len(df)),
        "train_before_outlier_filter": split["n_train_before_outlier_filter"],
        "train_after_outlier_filter": split["n_train_after_outlier_filter"],
        "train_outliers_removed": split["n_train_outliers_removed"],
        "test_unfiltered": int(len(split["X_test"])),
        "risk_encoding": RISK_ENCODING,
        "primary_feature_set": "full",
        "group_columns": GROUP_COLS,
        "main_leakage_rule": "Split first; outlier bounds learned on train only; test set not filtered.",
        "subgroup_rule": "Per-group models reuse the global split (same patients), never re-split.",
    }
    (out_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    df = load_raw()
    split = split_then_filter_outliers(df)
    write_preprocessing_outputs(df, split)
    write_eda_outputs(df)
    print(f"Preprocessing outputs written to {OUT_P1}")
    print(f"Risk encoding: {RISK_ENCODING}; primary feature set: full; groups: {GROUP_COLS}")


if __name__ == "__main__":
    main()