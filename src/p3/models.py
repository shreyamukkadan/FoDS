
"""
src/p3/models.py
Thyroid Cancer Risk — Foundations of Data Science Group Project

P3: Model Training and Evaluation — final TA-aligned version

Main purpose
------------
Answer the project question honestly:

    Can ML predict whether a thyroid cancer case is malignant or benign?

This script keeps the existing modelling logic but adds the missing feature-selection
piece requested by the TA:

    1. Forward feature selection
    2. Backward feature elimination
    3. Ablation study as a smaller supporting comparison, not the main evidence

Models kept
-----------
We keep the same model family as before:

    1. Logistic Regression
       - interpretable linear baseline
       - good for checking whether the signal is mostly additive/linear

    2. Random Forest
       - nonlinear tree-based model
       - can capture interactions without assuming linear effects

    3. XGBoost, if installed
       - stronger boosted tree model for tabular data
       - useful check whether more model capacity improves results

We do NOT add a neural network here because this is tabular data with limited
feature signal. A neural net would add complexity without improving the logic
of the report.

Important design decisions
--------------------------
1. Patient_ID is dropped because it is only an identifier.
2. Thyroid_Cancer_Risk is dropped because it appears to be a pre-computed risk
   category and may cause shortcut learning / target leakage.
3. Country and Ethnicity are dropped from the main model because they are
   demographic/proxy variables. In a real medical setting, they may reflect
   healthcare access, screening patterns, population structure or dataset bias
   rather than directly actionable clinical evidence. Therefore they are not
   used for the main predictive model.
4. Feature selection is done using cross-validation on the TRAINING set only.
   The TEST set is touched only once for final evaluation.
5. Model selection uses threshold-independent metrics by default
   (average precision / ROC-AUC). Threshold tuning is done afterwards.

Usage
-----
From the project root:

    python -m src.p3.models

Expected folder structure:

    project/
    ├── data/
    │   └── thyroid_cancer_risk_data.csv
    └── src/
        └── p3/
            └── models.py

Outputs are saved to:

    outputs/p3/
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — XGBoost model will be skipped.")
    print("Install with: pip install xgboost")


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = "data/thyroid_cancer_risk_data.csv"
OUTPUT_DIR = "outputs/p3"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Average precision is useful for imbalanced binary classification because it
# focuses on positive-class ranking. ROC-AUC is reported too.
FEATURE_SELECTION_SCORING = "average_precision"
GRIDSEARCH_SCORING = "average_precision"

# Forward/backward stopping rule:
# a new feature/removal must improve CV score by at least this amount.
MIN_SELECTION_IMPROVEMENT = 0.001

# Main threshold strategy used for final model comparison.
# Options: "f1", "f05", "min_precision"
MAIN_THRESHOLD_MODE = "f1"
MIN_PRECISION_TARGET = 0.50

# Which metric picks the canonical best model for later parts.
BEST_SELECTION_METRIC = "F1"

# Keep runtime controlled. RF/XGB grids are intentionally not huge.
RUN_XGBOOST = True
RUN_BACKWARD_SELECTION = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

CONTINUOUS_COLS = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]

BINARY_COLS = [
    "Family_History",
    "Radiation_Exposure",
    "Iodine_Deficiency",
    "Smoking",
    "Obesity",
    "Diabetes",
]

BASE_FEATURES = CONTINUOUS_COLS + BINARY_COLS + ["Gender"]

DROP_COLS = [
    "Patient_ID",
    "Country",
    "Ethnicity",
    "Thyroid_Cancer_Risk",
]

TARGET_COL = "Diagnosis"


# =============================================================================
# SMALL PRINTING HELPERS
# =============================================================================

def print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subtitle(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# =============================================================================
# PREPROCESSING FOR P3
# =============================================================================

@dataclass
class PreprocessOutput:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler: StandardScaler
    train_removed_outliers: int
    raw_rows: int
    rows_after_null: int


def load_and_preprocess_p3(
    data_path: str,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    verbose: bool = True,
) -> PreprocessOutput:
    """
    P3-specific preprocessing.

    This is kept inside P3 because the existing shared preprocessing/statistics
    files are already fixed by the group and are not being changed now.

    Leakage prevention:
        - train/test split happens before outlier handling
        - outlier statistics are fitted on training data only
        - StandardScaler is fitted on training data only
        - test set is not filtered after splitting
    """
    df = pd.read_csv(data_path)
    raw_rows = len(df)
    df = df.dropna().copy()
    rows_after_null = len(df)

    if verbose:
        print_title("P3 PREPROCESSING")
        print(f"[1] Loaded raw data: {raw_rows:,} rows")
        print(f"[2] After dropna:    {rows_after_null:,} rows ({raw_rows - rows_after_null:,} dropped)")

    missing = [c for c in DROP_COLS + BASE_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    if verbose:
        print("\n[3] Columns excluded from the main model:")
        print(f"    {DROP_COLS}")
        print("    Note: Country/Ethnicity are excluded because they are demographic/proxy variables,")
        print("    not directly actionable clinical measurements for the main prediction claim.")
        print("    Thyroid_Cancer_Risk is excluded because it may be a pre-computed shortcut variable.")

    # Binary Yes/No features.
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # Gender as binary for this dataset.
    df["Gender"] = (df["Gender"] == "Male").astype(int)

    # Target.
    df[TARGET_COL] = (df[TARGET_COL] == "Malignant").astype(int)

    X = df[BASE_FEATURES].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Outlier removal on train only.
    train_mean = X_train[CONTINUOUS_COLS].mean()
    train_std = X_train[CONTINUOUS_COLS].std(ddof=0).replace(0, 1)

    z_train = (X_train[CONTINUOUS_COLS] - train_mean) / train_std
    mask_train = (z_train.abs() <= 3).all(axis=1)

    n_train_before = len(X_train)
    X_train = X_train.loc[mask_train].copy()
    y_train = y_train.loc[mask_train].copy()
    train_removed_outliers = n_train_before - len(X_train)

    # Scaling continuous features only.
    scaler = StandardScaler()
    X_train_sc = X_train.copy()
    X_test_sc = X_test.copy()

    X_train_sc[CONTINUOUS_COLS] = X_train_sc[CONTINUOUS_COLS].astype(float)
    X_test_sc[CONTINUOUS_COLS] = X_test_sc[CONTINUOUS_COLS].astype(float)

    X_train_sc[CONTINUOUS_COLS] = scaler.fit_transform(X_train_sc[CONTINUOUS_COLS])
    X_test_sc[CONTINUOUS_COLS] = scaler.transform(X_test_sc[CONTINUOUS_COLS])

    if verbose:
        print("\n[4] Encoded binary variables, Gender and Diagnosis.")
        print(f"[5] Features used in main model ({len(BASE_FEATURES)}): {BASE_FEATURES}")
        print(f"[6] Full-data class balance:")
        print(f"    Benign:    {(y == 0).sum():,} ({(y == 0).mean() * 100:.1f}%)")
        print(f"    Malignant: {(y == 1).sum():,} ({(y == 1).mean() * 100:.1f}%)")
        print(f"[7] Stratified train/test split:")
        print(f"    Train: {len(X_train):,} rows — malignant rate: {y_train.mean() * 100:.1f}%")
        print(f"    Test:  {len(X_test):,} rows — malignant rate: {y_test.mean() * 100:.1f}%")
        print(f"[8] Train-only outlier removal:")
        print(f"    Removed from train: {train_removed_outliers:,}")
        print(f"    Removed from test:  0")
        print(f"[9] StandardScaler fitted on train only for: {CONTINUOUS_COLS}")
        print("\nPreprocessing complete.")

    return PreprocessOutput(
        X_train=X_train_sc,
        X_test=X_test_sc,
        y_train=y_train,
        y_test=y_test,
        feature_names=BASE_FEATURES,
        scaler=scaler,
        train_removed_outliers=train_removed_outliers,
        raw_rows=raw_rows,
        rows_after_null=rows_after_null,
    )


# =============================================================================
# FEATURE SIGNAL DIAGNOSTICS
# =============================================================================

def single_feature_signal_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring: str = FEATURE_SELECTION_SCORING,
) -> pd.DataFrame:
    """
    Quick train-only single-feature report.

    This is not used to automatically delete features from the final model.
    It is used to document which features look random-like on their own.
    """
    rows = []

    for col in X_train.columns:
        X_col = X_train[[col]]

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

        scores = cross_val_score(
            model,
            X_col,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        # Directionless single-feature ROC-AUC, useful for interpretability.
        try:
            auc_scores = cross_val_score(
                model,
                X_col,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            auc_mean = float(np.mean(auc_scores))
        except Exception:
            auc_mean = np.nan

        rows.append({
            "Feature": col,
            f"CV {scoring} mean": round(float(np.mean(scores)), 5),
            f"CV {scoring} std": round(float(np.std(scores)), 5),
            "CV ROC-AUC mean": round(auc_mean, 5) if not pd.isna(auc_mean) else np.nan,
            "Random-like flag": bool((not pd.isna(auc_mean)) and abs(auc_mean - 0.5) < 0.03),
        })

    return pd.DataFrame(rows).sort_values(f"CV {scoring} mean", ascending=False)


# =============================================================================
# FORWARD / BACKWARD FEATURE SELECTION
# =============================================================================

def cv_score_for_features(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    cv: StratifiedKFold,
    scoring: str,
) -> Tuple[float, float]:
    """Return mean/std CV score for a selected feature list."""
    if len(features) == 0:
        return np.nan, np.nan

    scores = cross_val_score(
        clone(estimator),
        X[features],
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    return float(np.mean(scores)), float(np.std(scores))


def forward_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_features: List[str],
    estimator,
    cv: StratifiedKFold,
    scoring: str = FEATURE_SELECTION_SCORING,
    min_improvement: float = MIN_SELECTION_IMPROVEMENT,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Greedy forward selection.

    Starts with no features and adds the feature that gives the largest CV
    improvement. Stops when the best additional feature improves the score by
    less than min_improvement.
    """
    selected: List[str] = []
    remaining = candidate_features.copy()
    current_score = -np.inf
    rows = []
    step = 0

    while remaining:
        step += 1
        trial_rows = []

        for feature in remaining:
            trial_features = selected + [feature]
            mean_score, std_score = cv_score_for_features(
                estimator,
                X,
                y,
                trial_features,
                cv,
                scoring,
            )
            trial_rows.append((feature, mean_score, std_score))

        best_feature, best_score, best_std = max(trial_rows, key=lambda x: x[1])
        improvement = best_score - current_score if np.isfinite(current_score) else best_score

        accept = bool((not np.isfinite(current_score)) or improvement >= min_improvement)

        rows.append({
            "Step": step,
            "Action": "add" if accept else "stop",
            "Feature": best_feature,
            "Selected features after step": selected + [best_feature] if accept else selected,
            f"CV {scoring} mean": round(best_score, 5),
            f"CV {scoring} std": round(best_std, 5),
            "Improvement": round(float(improvement), 5),
            "Accepted": accept,
        })

        if not accept:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        current_score = best_score

    return selected, pd.DataFrame(rows)


def backward_feature_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_features: List[str],
    estimator,
    cv: StratifiedKFold,
    scoring: str = FEATURE_SELECTION_SCORING,
    min_improvement: float = MIN_SELECTION_IMPROVEMENT,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Greedy backward elimination.

    Starts with all features and removes the feature whose removal gives the
    best CV score. The removal is accepted only if score does not decrease by
    more than min_improvement. This keeps the final feature set compact without
    aggressively deleting clinically plausible variables.
    """
    selected = candidate_features.copy()
    current_score, current_std = cv_score_for_features(
        estimator,
        X,
        y,
        selected,
        cv,
        scoring,
    )

    rows = [{
        "Step": 0,
        "Action": "start",
        "Removed feature": "",
        "Remaining features after step": selected.copy(),
        f"CV {scoring} mean": round(current_score, 5),
        f"CV {scoring} std": round(current_std, 5),
        "Score change": 0.0,
        "Accepted": True,
    }]

    step = 0

    while len(selected) > 1:
        step += 1
        trial_rows = []

        for feature in selected:
            trial_features = [f for f in selected if f != feature]
            mean_score, std_score = cv_score_for_features(
                estimator,
                X,
                y,
                trial_features,
                cv,
                scoring,
            )
            trial_rows.append((feature, mean_score, std_score, trial_features))

        removed_feature, best_score, best_std, best_features = max(trial_rows, key=lambda x: x[1])
        score_change = best_score - current_score

        # Accept if removal improves score, or if the loss is tiny.
        accept = bool(score_change >= -min_improvement)

        rows.append({
            "Step": step,
            "Action": "remove" if accept else "stop",
            "Removed feature": removed_feature,
            "Remaining features after step": best_features if accept else selected.copy(),
            f"CV {scoring} mean": round(best_score, 5),
            f"CV {scoring} std": round(best_std, 5),
            "Score change": round(float(score_change), 5),
            "Accepted": accept,
        })

        if not accept:
            break

        selected = best_features
        current_score = best_score

    return selected, pd.DataFrame(rows)


def choose_feature_set(
    forward_features: List[str],
    backward_features: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selector_estimator,
    cv: StratifiedKFold,
    scoring: str,
) -> Tuple[str, List[str], pd.DataFrame]:
    """
    Choose the feature set used for final models.

    We compare:
        - Full retained features
        - Forward-selected features
        - Backward-selected features

    The selected set is whichever has the best CV score using the same
    lightweight selection estimator.
    """
    candidates = {
        "Full retained features": X_train.columns.tolist(),
        "Forward-selected features": forward_features,
        "Backward-selected features": backward_features,
    }

    rows = []
    for name, features in candidates.items():
        mean_score, std_score = cv_score_for_features(
            selector_estimator,
            X_train,
            y_train,
            features,
            cv,
            scoring,
        )
        rows.append({
            "Feature set": name,
            "n_features": len(features),
            "Features": features,
            f"CV {scoring} mean": round(mean_score, 5),
            f"CV {scoring} std": round(std_score, 5),
        })

    df = pd.DataFrame(rows).sort_values(f"CV {scoring} mean", ascending=False)
    best_row = df.iloc[0]
    return str(best_row["Feature set"]), list(best_row["Features"]), df


# =============================================================================
# THRESHOLD UTILITIES
# =============================================================================

def cv_probs(model, X, y, cv) -> np.ndarray:
    """Out-of-fold positive-class probabilities for threshold tuning."""
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def threshold_for_max_f1(probs, y):
    prec, rec, thr = precision_recall_curve(y, probs)
    prec, rec = prec[:-1], rec[:-1]
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    idx = int(np.argmax(f1s))
    return float(thr[idx]), float(f1s[idx]), float(prec[idx]), float(rec[idx])


def threshold_for_max_fbeta(probs, y, beta=0.5):
    prec, rec, thr = precision_recall_curve(y, probs)
    prec, rec = prec[:-1], rec[:-1]
    beta2 = beta ** 2
    fbeta = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-9)
    idx = int(np.argmax(fbeta))
    return float(thr[idx]), float(fbeta[idx]), float(prec[idx]), float(rec[idx])


def threshold_for_min_precision(probs, y, min_precision=0.50):
    prec, rec, thr = precision_recall_curve(y, probs)
    prec, rec = prec[:-1], rec[:-1]

    valid = np.where(prec >= min_precision)[0]
    if len(valid) == 0:
        return None, None, None, None

    idx = int(valid[np.argmax(rec[valid])])
    f1 = 2 * prec[idx] * rec[idx] / (prec[idx] + rec[idx] + 1e-9)
    return float(thr[idx]), float(f1), float(prec[idx]), float(rec[idx])


def choose_threshold_from_probs(probs, y, mode="f1", min_precision=0.50):
    if mode == "f1":
        thr, score, prec, rec = threshold_for_max_f1(probs, y)
        return {
            "mode": "f1",
            "label": "F1-optimal (screening)",
            "threshold": thr,
            "score": score,
            "cv_precision": prec,
            "cv_recall": rec,
            "note": "",
        }

    if mode == "f05":
        thr, score, prec, rec = threshold_for_max_fbeta(probs, y, beta=0.5)
        return {
            "mode": "f05",
            "label": "F0.5-optimal (precision-leaning)",
            "threshold": thr,
            "score": score,
            "cv_precision": prec,
            "cv_recall": rec,
            "note": "",
        }

    if mode == "min_precision":
        thr, score, prec, rec = threshold_for_min_precision(
            probs,
            y,
            min_precision=min_precision,
        )
        if thr is None:
            thr, score, prec, rec = threshold_for_max_fbeta(probs, y, beta=0.5)
            return {
                "mode": "min_precision_fallback_f05",
                "label": f"Min-precision>={min_precision:.2f} unreachable; fallback F0.5",
                "threshold": thr,
                "score": score,
                "cv_precision": prec,
                "cv_recall": rec,
                "note": f"Precision target {min_precision:.2f} unreachable on CV; used F0.5 fallback.",
            }

        return {
            "mode": "min_precision",
            "label": f"Min-precision>={min_precision:.2f}",
            "threshold": thr,
            "score": score,
            "cv_precision": prec,
            "cv_recall": rec,
            "note": "",
        }

    raise ValueError("mode must be one of: 'f1', 'f05', 'min_precision'")


def get_all_threshold_strategies(model, X_train, y_train, cv, min_precision=0.50):
    probs = cv_probs(model, X_train, y_train, cv)
    return [
        choose_threshold_from_probs(probs, y_train, mode="f1", min_precision=min_precision),
        choose_threshold_from_probs(probs, y_train, mode="f05", min_precision=min_precision),
        choose_threshold_from_probs(probs, y_train, mode="min_precision", min_precision=min_precision),
    ]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_at_threshold(name, strategy_info, model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = strategy_info["threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Model": name,
        "Strategy": strategy_info["label"],
        "Mode": strategy_info["mode"],
        "Threshold": round(float(threshold), 3),
        "CV Precision": round(float(strategy_info["cv_precision"]), 4),
        "CV Recall": round(float(strategy_info["cv_recall"]), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Average Precision": round(average_precision_score(y_test, y_proba), 4),
        "Note": strategy_info["note"],
        "_y_pred": y_pred,
        "_y_proba": y_proba,
    }


# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring: str = FEATURE_SELECTION_SCORING,
) -> pd.DataFrame:
    """
    Supporting comparison only.

    This is not the main feature-selection method. It helps interpret where
    the signal sits after forward/backward selection.
    """
    ablation_sets = {
        "Clinical continuous only": [c for c in CONTINUOUS_COLS if c in X_train.columns],
        "Binary risk factors + Gender": [c for c in BINARY_COLS + ["Gender"] if c in X_train.columns],
        "All retained features": X_train.columns.tolist(),
    }

    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    rows = []
    for name, cols in ablation_sets.items():
        mean_score, std_score = cv_score_for_features(
            estimator,
            X_train,
            y_train,
            cols,
            cv,
            scoring,
        )
        mean_auc, std_auc = cv_score_for_features(
            estimator,
            X_train,
            y_train,
            cols,
            cv,
            "roc_auc",
        )
        rows.append({
            "Feature set": name,
            "n_features": len(cols),
            "Features": cols,
            f"CV {scoring} mean": round(mean_score, 5),
            f"CV {scoring} std": round(std_score, 5),
            "CV ROC-AUC mean": round(mean_auc, 5),
            "CV ROC-AUC std": round(std_auc, 5),
        })

    return pd.DataFrame(rows).sort_values(f"CV {scoring} mean", ascending=False)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def save_feature_selection_plot(selection_compare_df: pd.DataFrame, scoring: str, path: str) -> None:
    score_col = f"CV {scoring} mean"
    std_col = f"CV {scoring} std"

    df = selection_compare_df.copy().sort_values(score_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df[score_col], xerr=df[std_col], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Feature set"])
    ax.set_xlabel(f"CV {scoring}")
    ax.set_title("Feature selection comparison")
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(df[score_col]):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ablation_plot(ablation_df: pd.DataFrame, scoring: str, path: str) -> None:
    score_col = f"CV {scoring} mean"
    std_col = f"CV {scoring} std"

    df = ablation_df.copy().sort_values(score_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df[score_col], xerr=df[std_col], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Feature set"])
    ax.set_xlabel(f"CV {scoring}")
    ax.set_title("Ablation study — supporting comparison only")
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(df[score_col]):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_probability_overlap_plot(best_name, best_model, X_test, y_test, path: str) -> None:
    y_proba = best_model.predict_proba(X_test)[:, 1]
    benign_probs = y_proba[y_test == 0]
    malignant_probs = y_proba[y_test == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(benign_probs, bins=40, alpha=0.55, density=True, label="Benign")
    ax.hist(malignant_probs, bins=40, alpha=0.55, density=True, label="Malignant")

    ax.set_xlabel("Predicted probability of malignant")
    ax.set_ylabel("Density")
    ax.set_title(f"Probability overlap — {best_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_importance_plot(best_name, best_model, feature_names: List[str], path: str) -> None:
    importance = None

    if hasattr(best_model, "feature_importances_"):
        importance = np.array(best_model.feature_importances_)
    elif hasattr(best_model, "coef_"):
        importance = np.abs(best_model.coef_[0])

    if importance is None:
        return

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(imp_df))))
    ax.barh(imp_df["Feature"], imp_df["Importance"], alpha=0.85)
    ax.set_xlabel("Importance" if hasattr(best_model, "feature_importances_") else "|Coefficient|")
    ax.set_title(f"Feature importance — {best_name}")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_title("P3 — MODEL TRAINING AND EVALUATION")

    prep = load_and_preprocess_p3(
        data_path=DATA_PATH,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        verbose=True,
    )

    X_train = prep.X_train
    X_test = prep.X_test
    y_train = prep.y_train
    y_test = prep.y_test
    feature_names = prep.feature_names
    scaler = prep.scaler

    n_benign = int((y_train == 0).sum())
    n_malignant = int((y_train == 1).sum())
    imbal_ratio = n_benign / max(n_malignant, 1)

    print("\nClass imbalance after train-only outlier removal:")
    print(f"  Benign:    {n_benign:,}")
    print(f"  Malignant: {n_malignant:,}")
    print(f"  Ratio neg/pos: {imbal_ratio:.2f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # -------------------------------------------------------------------------
    # 1. SINGLE-FEATURE DIAGNOSTICS
    # -------------------------------------------------------------------------
    print_subtitle("Single-feature signal report")
    print("Purpose: document weak/random-like features. This does not automatically delete features.")

    signal_df = single_feature_signal_report(
        X_train,
        y_train,
        cv,
        scoring=FEATURE_SELECTION_SCORING,
    )
    print(signal_df.to_string(index=False))

    signal_path = os.path.join(OUTPUT_DIR, "single_feature_signal_report.csv")
    signal_df.to_csv(signal_path, index=False)
    print(f"\nSaved: {signal_path}")

    # -------------------------------------------------------------------------
    # 2. FORWARD / BACKWARD FEATURE SELECTION
    # -------------------------------------------------------------------------
    print_subtitle("Forward and backward feature selection")
    print("Main TA-aligned feature-selection step.")
    print(f"Scoring: {FEATURE_SELECTION_SCORING}")
    print(f"Minimum improvement threshold: {MIN_SELECTION_IMPROVEMENT}")

    selector_estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    forward_features, forward_df = forward_feature_selection(
        X_train,
        y_train,
        candidate_features=feature_names,
        estimator=selector_estimator,
        cv=cv,
        scoring=FEATURE_SELECTION_SCORING,
        min_improvement=MIN_SELECTION_IMPROVEMENT,
    )

    print("\nForward selection result:")
    print(f"Selected features ({len(forward_features)}): {forward_features}")
    print(forward_df.to_string(index=False))

    forward_path = os.path.join(OUTPUT_DIR, "forward_feature_selection.csv")
    forward_df.to_csv(forward_path, index=False)

    if RUN_BACKWARD_SELECTION:
        backward_features, backward_df = backward_feature_elimination(
            X_train,
            y_train,
            candidate_features=feature_names,
            estimator=selector_estimator,
            cv=cv,
            scoring=FEATURE_SELECTION_SCORING,
            min_improvement=MIN_SELECTION_IMPROVEMENT,
        )
        print("\nBackward elimination result:")
        print(f"Selected features ({len(backward_features)}): {backward_features}")
        print(backward_df.to_string(index=False))
    else:
        backward_features = feature_names.copy()
        backward_df = pd.DataFrame()

    backward_path = os.path.join(OUTPUT_DIR, "backward_feature_elimination.csv")
    backward_df.to_csv(backward_path, index=False)

    selected_set_name, selected_features, feature_set_compare_df = choose_feature_set(
        forward_features=forward_features,
        backward_features=backward_features,
        X_train=X_train,
        y_train=y_train,
        selector_estimator=selector_estimator,
        cv=cv,
        scoring=FEATURE_SELECTION_SCORING,
    )

    print_title("FEATURE SET DECISION")
    print(feature_set_compare_df.to_string(index=False))
    print(f"\nChosen feature set for final models: {selected_set_name}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")

    feature_set_path = os.path.join(OUTPUT_DIR, "feature_set_comparison.csv")
    feature_set_compare_df.to_csv(feature_set_path, index=False)

    fs_plot_path = os.path.join(OUTPUT_DIR, "feature_selection_comparison.png")
    save_feature_selection_plot(feature_set_compare_df, FEATURE_SELECTION_SCORING, fs_plot_path)
    print(f"\nSaved: {forward_path}")
    print(f"Saved: {backward_path}")
    print(f"Saved: {feature_set_path}")
    print(f"Saved: {fs_plot_path}")

    # Restrict final model training/evaluation to chosen feature set.
    X_train_sel = X_train[selected_features].copy()
    X_test_sel = X_test[selected_features].copy()

    # -------------------------------------------------------------------------
    # 3. ABLATION STUDY — SUPPORTING COMPARISON ONLY
    # -------------------------------------------------------------------------
    print_subtitle("Ablation study — supporting comparison only")
    print("This is kept for interpretation. It is NOT the main TA-requested feature-selection method.")

    ablation_df = run_ablation_study(
        X_train,
        y_train,
        cv,
        scoring=FEATURE_SELECTION_SCORING,
    )
    print(ablation_df.to_string(index=False))

    ablation_path = os.path.join(OUTPUT_DIR, "ablation_study_supporting.csv")
    ablation_df.to_csv(ablation_path, index=False)

    ablation_plot_path = os.path.join(OUTPUT_DIR, "ablation_study_supporting.png")
    save_ablation_plot(ablation_df, FEATURE_SELECTION_SCORING, ablation_plot_path)

    print(f"\nSaved: {ablation_path}")
    print(f"Saved: {ablation_plot_path}")

    # -------------------------------------------------------------------------
    # 4. MODEL DEFINITIONS
    # -------------------------------------------------------------------------
    print_subtitle("Model choices")
    print("Logistic Regression: interpretable linear baseline.")
    print("Random Forest: nonlinear tree model that can capture interactions.")
    print("XGBoost: stronger boosted tree model for tabular data, if installed.")
    print("No neural network: unnecessary complexity for this tabular dataset and report scope.")

    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf_search = GridSearchCV(
        estimator=rf_base,
        param_grid=rf_param_grid,
        scoring=GRIDSEARCH_SCORING,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    xgb_search = None
    if XGBOOST_AVAILABLE and RUN_XGBOOST:
        xgb_base = XGBClassifier(
            scale_pos_weight=imbal_ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

        xgb_param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        xgb_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=xgb_param_grid,
            scoring=GRIDSEARCH_SCORING,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

    # -------------------------------------------------------------------------
    # 5. TRAIN MODELS
    # -------------------------------------------------------------------------
    print_subtitle("Training Logistic Regression")
    lr_model.fit(X_train_sel, y_train)
    print("Done.")

    print_subtitle("Training Random Forest with GridSearchCV")
    rf_search.fit(X_train_sel, y_train)
    print(f"Best params: {rf_search.best_params_}")
    print(f"Best CV {GRIDSEARCH_SCORING}: {rf_search.best_score_:.4f}")

    if xgb_search is not None:
        print_subtitle("Training XGBoost with GridSearchCV")
        xgb_search.fit(X_train_sel, y_train)
        print(f"Best params: {xgb_search.best_params_}")
        print(f"Best CV {GRIDSEARCH_SCORING}: {xgb_search.best_score_:.4f}")

    models_to_eval = [
        ("Logistic Regression", lr_model),
        ("Random Forest", rf_search.best_estimator_),
    ]
    if xgb_search is not None:
        models_to_eval.append(("XGBoost", xgb_search.best_estimator_))

    model_map = {name: model for name, model in models_to_eval}

    # -------------------------------------------------------------------------
    # 6. CLASS-WEIGHT SENSITIVITY — TWO-PANEL CORE PLOT
    # -------------------------------------------------------------------------
    print_subtitle("Class-weight sensitivity — fixed threshold vs F1-tuned threshold")

    weight_settings = [
        ("None (1:1)", {0: 1, 1: 1}),
        ("Mild (1:2)", {0: 1, 1: 2}),
        ("Moderate (1:3)", {0: 1, 1: 3}),
        ("Balanced (auto)", "balanced"),
        ("Strong (1:5)", {0: 1, 1: 5}),
        ("Aggressive (1:7)", {0: 1, 1: 7}),
    ]

    cw_fixed_rows = []
    cw_tuned_rows = []
    cw_all_modes_rows = []

    for label, weight in weight_settings:
        lr_cw = LogisticRegression(
            class_weight=weight,
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
        lr_cw.fit(X_train_sel, y_train)

        y_proba_test = lr_cw.predict_proba(X_test_sel)[:, 1]

        # Fixed threshold = 0.5.
        y_pred_fixed = (y_proba_test >= 0.5).astype(int)
        cw_fixed_rows.append({
            "Weight setting": label,
            "Threshold": 0.50,
            "Precision": round(precision_score(y_test, y_pred_fixed, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_fixed, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred_fixed, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred_fixed), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba_test), 4),
            "Average Precision": round(average_precision_score(y_test, y_proba_test), 4),
        })

        # F1-tuned threshold.
        probs_cv = cv_probs(lr_cw, X_train_sel, y_train, cv)
        f1_info = choose_threshold_from_probs(
            probs_cv,
            y_train,
            mode="f1",
            min_precision=MIN_PRECISION_TARGET,
        )
        y_pred_tuned = (y_proba_test >= f1_info["threshold"]).astype(int)
        cw_tuned_rows.append({
            "Weight setting": label,
            "Threshold": round(f1_info["threshold"], 3),
            "Precision": round(precision_score(y_test, y_pred_tuned, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_tuned, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred_tuned, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred_tuned), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba_test), 4),
            "Average Precision": round(average_precision_score(y_test, y_proba_test), 4),
        })

        # All threshold modes for appendix/debugging.
        for mode in ["f1", "f05", "min_precision"]:
            info = choose_threshold_from_probs(
                probs_cv,
                y_train,
                mode=mode,
                min_precision=MIN_PRECISION_TARGET,
            )
            y_pred_mode = (y_proba_test >= info["threshold"]).astype(int)
            cw_all_modes_rows.append({
                "Weight setting": label,
                "Mode": mode,
                "Strategy": info["label"],
                "Threshold": round(info["threshold"], 3),
                "CV Precision": round(info["cv_precision"], 4),
                "CV Recall": round(info["cv_recall"], 4),
                "Precision": round(precision_score(y_test, y_pred_mode, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred_mode, zero_division=0), 4),
                "F1": round(f1_score(y_test, y_pred_mode, zero_division=0), 4),
                "Accuracy": round(accuracy_score(y_test, y_pred_mode), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_proba_test), 4),
                "Average Precision": round(average_precision_score(y_test, y_proba_test), 4),
                "Note": info["note"],
            })

    cw_fixed_df = pd.DataFrame(cw_fixed_rows).set_index("Weight setting")
    cw_tuned_df = pd.DataFrame(cw_tuned_rows).set_index("Weight setting")
    cw_all_modes_df = pd.DataFrame(cw_all_modes_rows)

    print("\n(a) Fixed threshold = 0.5 — class_weight effect visible:")
    print(cw_fixed_df.to_string())

    print("\n(b) F1-tuned threshold — class_weight effect mostly neutralised:")
    print(cw_tuned_df.to_string())

    cw_fixed_df.to_csv(os.path.join(OUTPUT_DIR, "class_weight_fixed_threshold.csv"))
    cw_tuned_df.to_csv(os.path.join(OUTPUT_DIR, "class_weight_f1_tuned_threshold.csv"))
    cw_all_modes_df.to_csv(os.path.join(OUTPUT_DIR, "class_weight_all_threshold_modes.csv"), index=False)

    # Two-panel class-weight plot.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for ax, df_plot, title in [
        (axes[0], cw_fixed_df, "Fixed threshold = 0.5"),
        (axes[1], cw_tuned_df, "F1-tuned threshold"),
    ]:
        labels = df_plot.index.tolist()
        x = np.arange(len(labels))
        w = 0.25

        ax.bar(x - w, df_plot["Precision"], w, label="Precision", alpha=0.85)
        ax.bar(x, df_plot["Recall"], w, label="Recall", alpha=0.85)
        ax.bar(x + w, df_plot["F1"], w, label="F1", alpha=0.85)

        for i, val in enumerate(df_plot["F1"]):
            ax.text(i + w, val + 0.01, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Score", fontsize=12)
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Class-weight sensitivity: threshold retuning can neutralise class_weight effects",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    cw_plot_path = os.path.join(OUTPUT_DIR, "class_weight_sensitivity.png")
    fig.savefig(cw_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {cw_plot_path}")

    # -------------------------------------------------------------------------
    # 7. NAIVE BASELINES
    # -------------------------------------------------------------------------
    print_subtitle("Naive baselines")

    baseline_results = []
    baselines = [
        ("Dummy (Majority)", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        ("Dummy (Stratified)", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
    ]

    for name, baseline in baselines:
        baseline.fit(X_train_sel, y_train)
        y_pred = baseline.predict(X_test_sel)
        y_proba = baseline.predict_proba(X_test_sel)[:, 1]

        baseline_results.append({
            "Model": name,
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
            "Average Precision": round(average_precision_score(y_test, y_proba), 4),
        })

    baseline_df = pd.DataFrame(baseline_results).set_index("Model")
    print("\nBaseline performance:")
    print(baseline_df.to_string())

    baseline_path = os.path.join(OUTPUT_DIR, "baseline_comparison.csv")
    baseline_df.to_csv(baseline_path)
    print(f"\nSaved: {baseline_path}")

    # -------------------------------------------------------------------------
    # 8. THRESHOLD STRATEGIES PER MODEL
    # -------------------------------------------------------------------------
    print_subtitle("Threshold strategies per model")

    all_results = []
    strategy_info_by_model: Dict[str, List[dict]] = {}

    for name, model in models_to_eval:
        strategies = get_all_threshold_strategies(
            model,
            X_train_sel,
            y_train,
            cv,
            min_precision=MIN_PRECISION_TARGET,
        )
        strategy_info_by_model[name] = strategies

        print(f"\n{name}:")
        for info in strategies:
            print(
                f"  {info['label']}: threshold={info['threshold']:.3f}, "
                f"CV precision={info['cv_precision']:.3f}, "
                f"CV recall={info['cv_recall']:.3f}"
                + (f" | {info['note']}" if info["note"] else "")
            )
            all_results.append(evaluate_at_threshold(name, info, model, X_test_sel, y_test))

    strategies_df = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    )
    strategies_path = os.path.join(OUTPUT_DIR, "threshold_strategies.csv")
    strategies_df.to_csv(strategies_path, index=False)

    print_title("THRESHOLD STRATEGIES — TEST SET RESULTS")
    print(strategies_df.to_string(index=False))
    print(f"\nSaved: {strategies_path}")

    # -------------------------------------------------------------------------
    # 9. MAIN MODEL COMPARISON
    # -------------------------------------------------------------------------
    main_results = [
        r for r in all_results
        if (
            (MAIN_THRESHOLD_MODE == "f1" and r["Mode"] == "f1")
            or (MAIN_THRESHOLD_MODE == "f05" and r["Mode"] == "f05")
            or (MAIN_THRESHOLD_MODE == "min_precision" and r["Mode"].startswith("min_precision"))
        )
    ]

    metrics_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if not k.startswith("_") and k not in ["Strategy", "Mode", "Note"]
            }
            for r in main_results
        ]
    ).set_index("Model")

    print_title(f"MAIN COMPARISON — threshold mode: {MAIN_THRESHOLD_MODE}")
    print(metrics_df.round(4).to_string())

    combined_df = pd.concat([
        metrics_df[["F1", "Precision", "Recall", "Accuracy", "ROC-AUC", "Average Precision"]],
        baseline_df[["F1", "Precision", "Recall", "Accuracy", "ROC-AUC", "Average Precision"]],
    ])

    combined_path = os.path.join(OUTPUT_DIR, "models_vs_baselines.csv")
    combined_df.to_csv(combined_path)

    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    metrics_df.round(4).to_csv(comparison_path)

    print(f"\nSaved: {combined_path}")
    print(f"Saved: {comparison_path}")

    # -------------------------------------------------------------------------
    # 10. BEST MODEL
    # -------------------------------------------------------------------------
    best_name = metrics_df[BEST_SELECTION_METRIC].idxmax()
    best_model = model_map[best_name]

    best_f1 = metrics_df.loc[best_name, "F1"]
    best_precision = metrics_df.loc[best_name, "Precision"]
    best_recall = metrics_df.loc[best_name, "Recall"]
    best_auc = metrics_df.loc[best_name, "ROC-AUC"]
    best_ap = metrics_df.loc[best_name, "Average Precision"]
    best_thr = metrics_df.loc[best_name, "Threshold"]

    best_model_results = [r for r in all_results if r["Model"] == best_name]

    print_title("BEST MODEL")
    print(f"Best model by {BEST_SELECTION_METRIC}: {best_name}")
    print(f"  Feature set = {selected_set_name}")
    print(f"  n_features  = {len(selected_features)}")
    print(f"  F1          = {best_f1:.4f}")
    print(f"  Precision   = {best_precision:.4f}")
    print(f"  Recall      = {best_recall:.4f}")
    print(f"  ROC-AUC     = {best_auc:.4f}")
    print(f"  Avg Precision = {best_ap:.4f}")
    print(f"  Threshold   = {best_thr:.3f}")

    # -------------------------------------------------------------------------
    # 11. PLOTS
    # -------------------------------------------------------------------------
    print_subtitle("Saving plots")

    # ROC curves.
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in main_results:
        fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{r['Model']} (AUC = {r['ROC-AUC']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Thyroid Cancer Diagnosis", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {roc_path}")

    # Confusion matrices for model comparison.
    ncols = len(main_results)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for i, r in enumerate(main_results):
        cm = confusion_matrix(y_test, r["_y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(
            f"{r['Model']}\n(thr={r['Threshold']:.2f}, P={r['Precision']:.2f}, R={r['Recall']:.2f})",
            fontsize=10,
        )

    fig.suptitle(
        f"Confusion Matrices — model comparison, threshold mode: {MAIN_THRESHOLD_MODE}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    cm_model_path = os.path.join(OUTPUT_DIR, "confusion_matrices_models.png")
    fig.savefig(cm_model_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cm_model_path}")

    # Confusion matrices for threshold strategies of best model.
    best_strategy_order = ["f1", "f05", "min_precision"]
    best_strategy_results = []
    for mode in best_strategy_order:
        match = next((r for r in best_model_results if r["Mode"].startswith(mode)), None)
        if match is not None:
            best_strategy_results.append(match)

    fig, axes = plt.subplots(1, len(best_strategy_results), figsize=(5 * len(best_strategy_results), 4))
    if len(best_strategy_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, best_strategy_results):
        cm = confusion_matrix(y_test, r["_y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{best_name}\n{r['Strategy']}\n"
            f"thr={r['Threshold']:.2f}, P={r['Precision']:.2f}, R={r['Recall']:.2f}, F1={r['F1']:.2f}",
            fontsize=9,
        )

    fig.suptitle(f"Confusion Matrices — {best_name} under threshold strategies", fontsize=13, y=1.05)
    fig.tight_layout()
    cm_strategy_path = os.path.join(OUTPUT_DIR, "confusion_matrices_best_model_strategies.png")
    fig.savefig(cm_strategy_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cm_strategy_path}")

    # Metric bar chart.
    metric_cols = ["ROC-AUC", "Average Precision", "F1", "Precision", "Recall", "Accuracy"]
    all_labels = [r["Model"] for r in main_results] + [b["Model"] for b in baseline_results]
    all_values = [[r[m] for m in metric_cols] for r in main_results] + [
        [b[m] for m in metric_cols] for b in baseline_results
    ]

    x = np.arange(len(metric_cols))
    width = 0.8 / len(all_labels)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (label, vals) in enumerate(zip(all_labels, all_values)):
        offset = (i - len(all_labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.2f}",
                ha="center",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Model Performance vs Naive Baselines — threshold mode: {MAIN_THRESHOLD_MODE}",
        fontsize=13,
    )
    ax.legend(fontsize=9, ncol=2)
    ax.axhline(0.5, color="grey", lw=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "metric_comparison.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # Precision-recall curves.
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_to_eval:
        y_proba = model.predict_proba(X_test_sel)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(rec, prec, lw=2, label=name)

        if name == best_name:
            marker_map = {
                "f1": "o",
                "f05": "s",
                "min_precision": "^",
                "min_precision_fallback_f05": "D",
            }
            for r in best_model_results:
                ax.scatter(
                    r["Recall"],
                    r["Precision"],
                    marker=marker_map.get(r["Mode"], "o"),
                    s=120,
                    zorder=5,
                    edgecolor="black",
                    linewidth=1.5,
                    label=f"{best_name} — {r['Strategy']}",
                )

    baseline_prec = y_test.mean()
    ax.axhline(
        baseline_prec,
        color="grey",
        lw=0.8,
        linestyle="--",
        alpha=0.5,
        label=f"Random baseline precision = {baseline_prec:.2f}",
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Trade-off and Threshold Strategies", fontsize=13)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    pr_path = os.path.join(OUTPUT_DIR, "precision_recall_curves.png")
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {pr_path}")

    # Threshold diagnostics.
    best_y_proba = best_model.predict_proba(X_test_sel)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, best_y_proba)
    prec = prec[:-1]
    rec = rec[:-1]
    f1s = 2 * prec * rec / (prec + rec + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thr, prec, lw=2, label="Precision")
    ax.plot(thr, rec, lw=2, label="Recall")
    ax.plot(thr, f1s, lw=2, label="F1")

    for r in best_model_results:
        ax.axvline(
            r["Threshold"],
            linestyle="--",
            lw=1,
            alpha=0.7,
            label=f"{r['Strategy']} thr={r['Threshold']:.2f}",
        )

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Diagnostics — {best_name}", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    threshold_diag_path = os.path.join(OUTPUT_DIR, "threshold_diagnostics_best_model.png")
    fig.savefig(threshold_diag_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {threshold_diag_path}")

    # Probability overlap.
    prob_overlap_path = os.path.join(OUTPUT_DIR, "probability_overlap_best_model.png")
    save_probability_overlap_plot(best_name, best_model, X_test_sel, y_test, prob_overlap_path)
    print(f"Saved: {prob_overlap_path}")

    # Feature importance.
    feature_importance_path = os.path.join(OUTPUT_DIR, "feature_importance_best_model.png")
    save_feature_importance_plot(best_name, best_model, selected_features, feature_importance_path)
    if os.path.exists(feature_importance_path):
        print(f"Saved: {feature_importance_path}")

    # -------------------------------------------------------------------------
    # 12. SAVE MODEL + METADATA FOR P4
    # -------------------------------------------------------------------------
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    meta_path = os.path.join(OUTPUT_DIR, "best_model_meta.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    threshold_strategies_meta = {}
    for r in best_model_results:
        threshold_strategies_meta[r["Strategy"]] = {
            "mode": r["Mode"],
            "threshold": None if pd.isna(r["Threshold"]) else float(r["Threshold"]),
            "precision": None if pd.isna(r["Precision"]) else float(r["Precision"]),
            "recall": None if pd.isna(r["Recall"]) else float(r["Recall"]),
            "f1": None if pd.isna(r["F1"]) else float(r["F1"]),
            "note": r["Note"],
        }

    meta = {
        "name": best_name,
        "selection_metric": BEST_SELECTION_METRIC,
        "main_threshold_mode": MAIN_THRESHOLD_MODE,
        "min_precision_target": MIN_PRECISION_TARGET,
        "feature_selection_scoring": FEATURE_SELECTION_SCORING,
        "gridsearch_scoring": GRIDSEARCH_SCORING,
        "chosen_feature_set": selected_set_name,
        "selected_features": selected_features,
        "excluded_columns": DROP_COLS,
        "exclusion_note": (
            "Patient_ID is an identifier. Thyroid_Cancer_Risk may be a pre-computed shortcut. "
            "Country and Ethnicity are demographic/proxy variables and are excluded from the main "
            "clinical prediction model for interpretability and fairness reasons."
        ),
        "f1": round(float(best_f1), 4),
        "precision": round(float(best_precision), 4),
        "recall": round(float(best_recall), 4),
        "roc_auc": round(float(best_auc), 4),
        "average_precision": round(float(best_ap), 4),
        "threshold": round(float(best_thr), 3),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "feature_names": selected_features,
        "threshold_strategies": threshold_strategies_meta,
    }

    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\nSaved artifacts:")
    print(f"  Best model:        {model_path}")
    print(f"  Scaler:            {scaler_path}")
    print(f"  Model comparison:  {comparison_path}")
    print(f"  Metadata:          {meta_path}")

    print_title("P3 complete")


# =============================================================================
# PUBLIC API FOR P4
# =============================================================================

def load_saved_best_model(output_dir: str = OUTPUT_DIR):
    """
    Load the saved best model and metadata.

    Useful for P4 or notebooks after running this script once.
    """
    meta_path = os.path.join(output_dir, "best_model_meta.json")
    model_path = os.path.join(output_dir, "best_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")

    with open(meta_path, "r") as fh:
        meta = json.load(fh)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return meta, model, scaler


if __name__ == "__main__":
    main()
