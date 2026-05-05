"""
src/p3/models.py
Thyroid Cancer Risk — Foundations of Data Science Group Project

P3: Model Training and Evaluation

Models trained:
    1. Logistic Regression  — linear baseline
    2. Random Forest        — GridSearchCV, 5-fold stratified CV
    3. XGBoost              — GridSearchCV, 5-fold stratified CV

Design principles:
    - Thyroid_Cancer_Risk is excluded at the preprocessing stage (P1).
      See P1 for justification; including it would constitute data leakage
      because it is a pre-computed risk label derived from the same features.
    - class_weight='balanced' / scale_pos_weight handles class imbalance
      without synthetic oversampling.
    - All hyperparameter tuning is performed exclusively on training data.
      The test set is touched only once — for final evaluation.
    - F1 is the primary optimisation metric (see written section below).

Usage (from project root):
    python -m src.p3.models

Outputs written to outputs/p3/:
    roc_curves.png
    confusion_matrices.png
    metric_comparison.png
    model_comparison.csv
    best_model.joblib
    best_model_meta.json

P4 can import the best model directly:
    from src.p3.models import get_best_model
    model_name, model, feature_names = get_best_model()
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier 
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — XGBoost model will be skipped.")
    print("Install with: pip install xgboost")

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


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH    = "data/thyroid_cancer_risk_data.csv"
OUTPUT_DIR   = "outputs/p3"
RANDOM_STATE = 42
CV_FOLDS     = 5
SCORING      = "f1"   # primary metric — see written section for justification

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOAD AND PREPROCESS DATA
# =============================================================================

print("\n" + "=" * 70)
print("P3 — MODEL TRAINING AND EVALUATION")
print("=" * 70)

X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
    data_path=DATA_PATH,
    random_state=RANDOM_STATE,
    test_size=0.2,
    verbose=True,
)

n_benign    = int((y_train == 0).sum())
n_malignant = int((y_train == 1).sum())
imbal_ratio = n_benign / n_malignant   # used as scale_pos_weight for XGBoost

print(f"\nClass imbalance (train set):")
print(f"  Benign:    {n_benign:,}")
print(f"  Malignant: {n_malignant:,}")
print(f"  Ratio (neg/pos): {imbal_ratio:.2f}  → XGBoost scale_pos_weight")


# =============================================================================
# 2. CROSS-VALIDATION STRATEGY
# =============================================================================

# StratifiedKFold preserves the Benign/Malignant ratio in every fold,
# preventing any single fold from being dominated by one class.
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# =============================================================================
# 3. MODEL DEFINITIONS + HYPERPARAMETER GRIDS
# =============================================================================

# ── 3A. Logistic Regression (baseline) ───────────────────────────────────────
# No grid search — used as an interpretable linear reference point.
# class_weight='balanced' scales sample weights inversely to class frequency.
lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE,
    solver="lbfgs",
)

# ── 3B. Random Forest ────────────────────────────────────────────────────────
rf_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_param_grid = {
    "n_estimators":      [100, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}
rf_search = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    scoring=SCORING,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,
)

# ── 3C. XGBoost ──────────────────────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    xgb_base = XGBClassifier(
        scale_pos_weight=imbal_ratio,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_param_grid = {
        "n_estimators":     [100, 300],
        "max_depth":        [3, 6],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=xgb_param_grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


# =============================================================================
# 4. TRAINING
# =============================================================================

print("\n" + "-" * 70)
print("Training Logistic Regression (baseline)...")
lr_model.fit(X_train, y_train)
print("  Done.")

print("\nTraining Random Forest with GridSearchCV...")
rf_search.fit(X_train, y_train)
print(f"  Best params : {rf_search.best_params_}")
print(f"  Best CV F1  : {rf_search.best_score_:.4f}")

if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost with GridSearchCV...")
    xgb_search.fit(X_train, y_train)
    print(f"  Best params : {xgb_search.best_params_}")
    print(f"  Best CV F1  : {xgb_search.best_score_:.4f}")


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate(name, model, X, y):
    """Return a dict of test-set metrics plus raw predictions."""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "Model":     name,
        "ROC-AUC":   roc_auc_score(y, y_proba),
        "F1":        f1_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall":    recall_score(y, y_pred),
        "Accuracy":  accuracy_score(y, y_pred),
        "_y_pred":   y_pred,
        "_y_proba":  y_proba,
    }


results = [
    evaluate("Logistic Regression", lr_model,              X_test, y_test),
    evaluate("Random Forest",       rf_search.best_estimator_, X_test, y_test),
]
if XGBOOST_AVAILABLE:
    results.append(evaluate("XGBoost", xgb_search.best_estimator_, X_test, y_test))

# Comparison table (public metrics only)
metrics_df = pd.DataFrame(
    [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
).set_index("Model")

print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE (test set)")
print("=" * 70)
print(metrics_df.round(4).to_string())


# =============================================================================
# 6. IDENTIFY BEST MODEL
# =============================================================================

best_name = metrics_df["F1"].idxmax()
best_f1   = metrics_df.loc[best_name, "F1"]
best_auc  = metrics_df.loc[best_name, "ROC-AUC"]

print(f"\n{'=' * 70}")
print(f"BEST MODEL: {best_name}")
print(f"  F1      = {best_f1:.4f}")
print(f"  ROC-AUC = {best_auc:.4f}")
print(f"  → P4 will run forward/backward selection and clustering on this model.")
print(f"{'=' * 70}")

_model_map = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_search.best_estimator_,
}
if XGBOOST_AVAILABLE:
    _model_map["XGBoost"] = xgb_search.best_estimator_

best_model = _model_map[best_name]


# =============================================================================
# 7. PLOTS
# =============================================================================

COLORS = ["#4C72B0", "#55A868", "#C44E52"]

# ── 7A. ROC Curves ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
    ax.plot(fpr, tpr, color=COLORS[i], lw=2,
            label=f"{r['Model']} (AUC = {r['ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Thyroid Cancer Diagnosis", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
_roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
fig.savefig(_roc_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {_roc_path}")

# ── 7B. Confusion Matrices ────────────────────────────────────────────────────
ncols = len(results)
fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
if ncols == 1:
    axes = [axes]
for i, r in enumerate(results):
    cm   = confusion_matrix(y_test, r["_y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(r["Model"], fontsize=11)
fig.suptitle("Confusion Matrices (test set)", fontsize=13, y=1.02)
fig.tight_layout()
_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
fig.savefig(_cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {_cm_path}")

# ── 7C. Metric Bar Chart ──────────────────────────────────────────────────────
metric_cols = ["ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]
x     = np.arange(len(metric_cols))
width = 0.8 / len(results)

fig, ax = plt.subplots(figsize=(11, 5))
for i, r in enumerate(results):
    vals   = [r[m] for m in metric_cols]
    offset = (i - len(results) / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=r["Model"], color=COLORS[i], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison (test set)", fontsize=13)
ax.legend(fontsize=10)
ax.axhline(0.5, color="grey", lw=0.8, linestyle="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
_bar_path = os.path.join(OUTPUT_DIR, "metric_comparison.png")
fig.savefig(_bar_path, dpi=150)
plt.close(fig)
print(f"Saved: {_bar_path}")


# =============================================================================
# 8. SAVE BEST MODEL + METADATA FOR P4
# =============================================================================

_model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
joblib.dump(best_model, _model_path)
print(f"\nSaved best model ({best_name}): {_model_path}")

metrics_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
print(f"Saved comparison table: {OUTPUT_DIR}/model_comparison.csv")

_meta = {
    "name":          best_name,
    "f1":            round(float(best_f1),  4),
    "roc_auc":       round(float(best_auc), 4),
    "model_path":    _model_path,
    "feature_names": feature_names,
}
_meta_path = os.path.join(OUTPUT_DIR, "best_model_meta.json")
with open(_meta_path, "w") as fh:
    json.dump(_meta, fh, indent=2)
print(f"Saved model metadata:   {_meta_path}")


# =============================================================================
# 9. PUBLIC API FOR P4
# =============================================================================

def get_best_model():
    """
    Return the best-performing model for use by P4.

    Returns
    -------
    model_name   : str         — human-readable model name
    model        : fitted      — sklearn / xgboost estimator (already fitted)
    feature_names: list[str]   — feature names in the order used during training
    """
    return best_name, best_model, feature_names


# =============================================================================
# 10. WRITTEN SECTION — ML METHODS + RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("WRITTEN SECTION: ML METHODS + RESULTS")
print("=" * 70)

print("""
WHY F1 OVER ACCURACY
━━━━━━━━━━━━━━━━━━━
Accuracy is a misleading metric when the classes are imbalanced. A naive
classifier that predicts "Benign" for every patient achieves high accuracy
while providing zero clinical value. F1 (the harmonic mean of Precision and
Recall) penalises both types of error simultaneously:
  • False negatives (cancer cases missed) carry the highest clinical cost.
  • False positives (healthy patients flagged as malignant) cause unnecessary
    anxiety and costly follow-up procedures.
The harmonic mean of Precision and Recall ensures that both must be high
before F1 is high — accuracy cannot do this under imbalance. F1 is therefore
used as the primary optimisation and model-selection metric throughout P3.


WHY THYROID_CANCER_RISK IS EXCLUDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thyroid_Cancer_Risk is a categorical risk label (Low / Medium / High) that is
derived from the same clinical variables used as model features (TSH level,
nodule size, family history, etc.). Its Pearson correlation with Diagnosis
(r = 0.37) makes it one of the strongest individual predictors in the dataset.
Including it would allow the model to shortcut genuine learning by reading a
pre-computed clinical summary rather than extracting patterns from underlying
biomarkers. Excluding it ensures models are evaluated on their ability to
generalise from raw clinical signals — the only setting that would be
meaningful in a prospective clinical deployment. P1 handles this exclusion
during preprocessing.


MODEL CHOICE JUSTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━
Three models spanning a range of complexity and inductive bias were selected:

1. Logistic Regression (baseline)
   A linear discriminative classifier with interpretable coefficients. It
   requires no feature interaction terms and scales well to large datasets.
   Its performance serves as a sanity check: substantially worse performance
   would indicate overfitting in the ensemble models; substantially better
   would suggest the relationship is non-linear.

2. Random Forest
   An ensemble of unpruned decision trees trained on bootstrap samples with
   random feature subsets (bagging). It naturally captures non-linear
   interactions and is robust to correlated features and outliers. The
   built-in feature importances make it useful for the feature selection
   work in P4.

3. XGBoost
   A gradient-boosted tree ensemble that builds trees sequentially, each one
   correcting the residuals of its predecessor. Gradient boosting often
   outperforms bagging on structured / tabular data and provides a different
   bias–variance trade-off compared to Random Forest.

Together, the three models provide a linear baseline, a bagging ensemble, and
a boosting ensemble — enough diversity to assess the benefit of complexity
and non-linearity for this dataset.


HYPERPARAMETER STRATEGY
━━━━━━━━━━━━━━━━━━━━━━
Grid search with 5-fold StratifiedKFold cross-validation is applied to Random
Forest and XGBoost. StratifiedKFold is used rather than standard KFold to
preserve the class ratio in every fold, which is important given the class
imbalance. All search is performed exclusively on the training set; the test
set is never consulted during tuning. F1 is used as the scoring metric for the
grid search, consistent with the primary selection criterion.

Key grid dimensions:
  Random Forest : n_estimators, max_depth, min_samples_split, min_samples_leaf
  XGBoost       : n_estimators, max_depth, learning_rate, subsample,
                  colsample_bytree


CLASS IMBALANCE HANDLING
━━━━━━━━━━━━━━━━━━━━━━━
We use cost-sensitive learning rather than resampling (SMOTE / random
undersampling):
  • Logistic Regression / Random Forest: class_weight='balanced'
    — sklearn computes per-class weights inversely proportional to class
      frequency and folds them into the loss function.
  • XGBoost: scale_pos_weight = n_benign / n_malignant
    — XGBoost multiplies the gradient of positive (Malignant) samples by this
      factor, increasing their effective contribution to the loss.

This strategy avoids synthesising data points (which can introduce artefacts
in feature space) and is computationally efficient for a large dataset.
""")

print("-" * 70)
print("MODEL COMPARISON TABLE (for report)")
print("-" * 70)
print(metrics_df[["ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]].round(4).to_string())

print(f"""
BEST MODEL: {best_name}
  ROC-AUC = {best_auc:.4f}
  F1      = {best_f1:.4f}

{best_name} achieves the highest F1 score on the held-out test set and is
selected as the best-performing model. P4 will apply forward/backward feature
selection and clustering analysis using this model exclusively.
""")

print("=" * 70)
print("P3 complete.")
print("=" * 70)
