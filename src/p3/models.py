"""
src/p3/models.py — compact P3 model pipeline, TA-feedback version
Run from project root:
    python -m src.p3.models
Requires:
    data/thyroid_cancer_risk_data.csv
    src/preprocessing.py with load_and_preprocess(...)
Outputs:
    outputs/p3/*.csv, *.json, *.png, *.joblib

TA-feedback additions:
    - Feature selection remains before final model training, but is validated with
      a nested-CV check where feature selection is repeated inside each outer fold.
    - Models are also trained/evaluated on all features, so feature selection can
      be compared against a simple all-features baseline, not only dummy baselines.
    - Held-out test set is still used only for final evaluation.
"""
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from src.p1.preprocessing import load_and_preprocess
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — skipping XGBoost.")
DATA_PATH = "data/thyroid_cancer_risk_data.csv"
BASE_OUTPUT_DIR = "outputs/p3"
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_ID)
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.20
FEATURE_SELECTION_SCORING = "average_precision"
GRIDSEARCH_SCORING = "average_precision"
MIN_SELECTION_IMPROVEMENT = 0.001
MAIN_THRESHOLD_MODE = "f1"          # "f1", "f05", or "min_precision"
MIN_PRECISION_TARGET = 0.50
BEST_SELECTION_METRIC = "F1"
RUN_XGBOOST = True
RUN_BACKWARD_SELECTION = True
RUN_NESTED_CV_CHECK = True
NESTED_CV_INNER_FOLDS = 3
os.makedirs(OUTPUT_DIR, exist_ok=False)
def title(text: str) -> None:
    print("\n" + "=" * 80 + f"\n{text}\n" + "=" * 80)
def subtitle(text: str) -> None:
    print("\n" + "-" * 80 + f"\n{text}\n" + "-" * 80)
def save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    return path
def cv_score(estimator, X: pd.DataFrame, y: pd.Series, features: List[str], cv, scoring: str) -> Tuple[float, float]:
    if not features:
        return np.nan, np.nan
    scores = cross_val_score(clone(estimator), X[features], y, cv=cv, scoring=scoring, n_jobs=-1)
    return float(scores.mean()), float(scores.std())
def cv_probs(model, X, y, cv) -> np.ndarray:
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
def clean_result_dict(r: Dict) -> Dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}
def single_feature_report(X, y, cv, scoring=FEATURE_SELECTION_SCORING) -> pd.DataFrame:
    rows = []
    base = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    for col in X.columns:
        ap_mean, ap_std = cv_score(base, X, y, [col], cv, scoring)
        auc_mean, _ = cv_score(base, X, y, [col], cv, "roc_auc")
        rows.append({
            "Feature": col,
            f"CV {scoring} mean": round(ap_mean, 5),
            f"CV {scoring} std": round(ap_std, 5),
            "CV ROC-AUC mean": round(auc_mean, 5),
            "Random-like flag": bool(abs(auc_mean - 0.5) < 0.03),
        })
    return pd.DataFrame(rows).sort_values(f"CV {scoring} mean", ascending=False)
def forward_selection(X, y, features, estimator, cv, scoring=FEATURE_SELECTION_SCORING) -> Tuple[List[str], pd.DataFrame]:
    selected, remaining, current = [], list(features), -np.inf
    rows = []
    while remaining:
        trials = []
        for f in remaining:
            mean, std = cv_score(estimator, X, y, selected + [f], cv, scoring)
            trials.append((f, mean, std))
        best_f, best, std = max(trials, key=lambda x: x[1])
        improvement = best - current if np.isfinite(current) else best
        accept = (not np.isfinite(current)) or improvement >= MIN_SELECTION_IMPROVEMENT
        rows.append({
            "Step": len(rows) + 1, "Action": "add" if accept else "stop", "Feature": best_f,
            "Selected features after step": selected + [best_f] if accept else selected,
            f"CV {scoring} mean": round(best, 5), f"CV {scoring} std": round(std, 5),
            "Improvement": round(float(improvement), 5), "Accepted": bool(accept),
        })
        if not accept:
            break
        selected.append(best_f)
        remaining.remove(best_f)
        current = best
    return selected, pd.DataFrame(rows)
def backward_elimination(X, y, features, estimator, cv, scoring=FEATURE_SELECTION_SCORING) -> Tuple[List[str], pd.DataFrame]:
    selected = list(features)
    current, std = cv_score(estimator, X, y, selected, cv, scoring)
    rows = [{
        "Step": 0, "Action": "start", "Removed feature": "",
        "Remaining features after step": selected.copy(),
        f"CV {scoring} mean": round(current, 5), f"CV {scoring} std": round(std, 5),
        "Score change": 0.0, "Accepted": True,
    }]
    while len(selected) > 1:
        trials = []
        for f in selected:
            trial = [x for x in selected if x != f]
            mean, std = cv_score(estimator, X, y, trial, cv, scoring)
            trials.append((f, mean, std, trial))
        removed, best, std, best_features = max(trials, key=lambda x: x[1])
        change = best - current
        accept = change >= -MIN_SELECTION_IMPROVEMENT
        rows.append({
            "Step": len(rows), "Action": "remove" if accept else "stop", "Removed feature": removed,
            "Remaining features after step": best_features if accept else selected.copy(),
            f"CV {scoring} mean": round(best, 5), f"CV {scoring} std": round(std, 5),
            "Score change": round(float(change), 5), "Accepted": bool(accept),
        })
        if not accept:
            break
        selected, current = best_features, best
    return selected, pd.DataFrame(rows)
def compare_feature_sets(X, y, full, forward, backward, estimator, cv, scoring) -> Tuple[str, List[str], pd.DataFrame]:
    candidates = {
        "Full retained features": list(full),
        "Forward-selected features": list(forward),
        "Backward-selected features": list(backward),
    }
    rows = []
    for name, feats in candidates.items():
        mean, std = cv_score(estimator, X, y, feats, cv, scoring)
        rows.append({
            "Feature set": name, "n_features": len(feats), "Features": feats,
            f"CV {scoring} mean": round(mean, 5), f"CV {scoring} std": round(std, 5),
        })
    df = pd.DataFrame(rows).sort_values(f"CV {scoring} mean", ascending=False)
    return str(df.iloc[0]["Feature set"]), list(df.iloc[0]["Features"]), df

def nested_cv_feature_selection(X, y, features, estimator, direction="forward", scoring=FEATURE_SELECTION_SCORING) -> Dict:
    """
    Leakage-free sanity check for feature selection.
    In each outer CV fold, feature selection is performed only on that fold's
    training portion, then evaluated on the fold's validation portion.
    """
    outer_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=NESTED_CV_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores, n_features = [], []
    selected_per_fold = []
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for train_idx, val_idx in outer_cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if direction == "forward":
            selected, _ = forward_selection(X_tr, y_tr, features, estimator, inner_cv, scoring)
        elif direction == "backward":
            selected, _ = backward_elimination(X_tr, y_tr, features, estimator, inner_cv, scoring)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

        model = clone(estimator).fit(X_tr[selected], y_tr)
        proba = model.predict_proba(X_val[selected])[:, 1]
        scores.append(average_precision_score(y_val, proba))
        n_features.append(len(selected))
        selected_per_fold.append(selected)

    if direction == "forward":
        selected_full, _ = forward_selection(X, y, features, estimator, inner_cv, scoring)
    else:
        selected_full, _ = backward_elimination(X, y, features, estimator, inner_cv, scoring)

    return {
        "Direction": direction,
        "Outer CV AP mean": round(float(np.mean(scores)), 5),
        "Outer CV AP std": round(float(np.std(scores)), 5),
        "Outer CV AP scores": [round(float(s), 5) for s in scores],
        "Mean n_features": round(float(np.mean(n_features)), 2),
        "Features when fit on full training set": selected_full,
        "Selected features per fold": selected_per_fold,
    }

def threshold_f1(probs, y):
    p, r, t = precision_recall_curve(y, probs)
    p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / (p + r + 1e-9)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i]), float(p[i]), float(r[i])
def threshold_fbeta(probs, y, beta=0.5):
    p, r, t = precision_recall_curve(y, probs)
    p, r = p[:-1], r[:-1]
    b2 = beta ** 2
    fb = (1 + b2) * p * r / (b2 * p + r + 1e-9)
    i = int(np.argmax(fb))
    return float(t[i]), float(fb[i]), float(p[i]), float(r[i])
def threshold_min_precision(probs, y, target=0.50):
    p, r, t = precision_recall_curve(y, probs)
    p, r = p[:-1], r[:-1]
    valid = np.where(p >= target)[0]
    if len(valid) == 0:
        return None, None, None, None
    i = int(valid[np.argmax(r[valid])])
    f1 = 2 * p[i] * r[i] / (p[i] + r[i] + 1e-9)
    return float(t[i]), float(f1), float(p[i]), float(r[i])
def choose_threshold(probs, y, mode="f1", min_precision=0.50) -> Dict:
    if mode == "f1":
        thr, score, p, r = threshold_f1(probs, y)
        return {"mode": "f1", "label": "F1-optimal (screening)", "threshold": thr, "score": score, "cv_precision": p, "cv_recall": r, "note": ""}
    if mode == "f05":
        thr, score, p, r = threshold_fbeta(probs, y, beta=0.5)
        return {"mode": "f05", "label": "F0.5-optimal (precision-leaning)", "threshold": thr, "score": score, "cv_precision": p, "cv_recall": r, "note": ""}
    if mode == "min_precision":
        thr, score, p, r = threshold_min_precision(probs, y, min_precision)
        if thr is None:
            thr, score, p, r = threshold_fbeta(probs, y, beta=0.5)
            note = f"Precision target {min_precision:.2f} unreachable on CV; used F0.5 fallback."
            return {"mode": "min_precision_fallback_f05", "label": f"Min-precision>={min_precision:.2f} unreachable; fallback F0.5", "threshold": thr, "score": score, "cv_precision": p, "cv_recall": r, "note": note}
        return {"mode": "min_precision", "label": f"Min-precision>={min_precision:.2f}", "threshold": thr, "score": score, "cv_precision": p, "cv_recall": r, "note": ""}
    raise ValueError("mode must be 'f1', 'f05', or 'min_precision'")
def strategies(model, X, y, cv) -> List[Dict]:
    probs = cv_probs(model, X, y, cv)
    return [choose_threshold(probs, y, m, MIN_PRECISION_TARGET) for m in ["f1", "f05", "min_precision"]]
def evaluate(name, model, X_test, y_test, strategy: Dict) -> Dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= strategy["threshold"]).astype(int)
    return {
        "Model": name, "Strategy": strategy["label"], "Mode": strategy["mode"],
        "Threshold": round(float(strategy["threshold"]), 3),
        "CV Precision": round(float(strategy["cv_precision"]), 4),
        "CV Recall": round(float(strategy["cv_recall"]), 4),
        "F1": round(f1_score(y_test, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
        "Average Precision": round(average_precision_score(y_test, proba), 4),
        "Note": strategy["note"], "_y_pred": pred, "_y_proba": proba,
    }
def baseline_results(X_train, y_train, X_test, y_test) -> List[Dict]:
    rows = []
    for name, strategy in [("Dummy (Majority)", "most_frequent"), ("Dummy (Stratified)", "stratified")]:
        model = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE).fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name, "Strategy": strategy, "Mode": "naive", "Threshold": np.nan,
            "CV Precision": np.nan, "CV Recall": np.nan,
            "F1": round(f1_score(y_test, pred, zero_division=0), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
            "Average Precision": round(average_precision_score(y_test, proba), 4),
            "Note": "", "_y_pred": pred, "_y_proba": proba,
        })
    return rows

def train_model_family(X_train, y_train, imbal_ratio, label="") -> Dict:
    """Train LogReg, RF and XGB using the same setup as the original pipeline."""
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE).fit(X_train, y_train),
    }
    rf_grid = {"n_estimators": [100, 300], "max_depth": [None, 8, 16], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
    rf = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), rf_grid, scoring=GRIDSEARCH_SCORING, cv=CV_FOLDS, n_jobs=-1, refit=True, verbose=1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf.best_estimator_
    print(f"RF best {label}:", rf.best_params_)
    if XGBOOST_AVAILABLE and RUN_XGBOOST:
        xgb_grid = {"n_estimators": [100, 300], "max_depth": [3, 6], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}
        xgb = GridSearchCV(XGBClassifier(scale_pos_weight=imbal_ratio, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0), xgb_grid, scoring=GRIDSEARCH_SCORING, cv=CV_FOLDS, n_jobs=-1, refit=True, verbose=1)
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb.best_estimator_
        print(f"XGB best {label}:", xgb.best_params_)
    return models


def evaluate_model_family(models: Dict, X_train, y_train, X_test, y_test, cv, variant: str) -> Tuple[List[Dict], List[Dict]]:
    """Evaluate each fitted model with the same CV-based threshold tuning."""
    all_results, main_results = [], []
    for name, model in models.items():
        for strat in strategies(model, X_train, y_train, cv):
            res = evaluate(name, model, X_test, y_test, strat)
            res["Variant"] = variant
            all_results.append(res)
            if strat["mode"].startswith(MAIN_THRESHOLD_MODE):
                main_results.append(res)
    return all_results, main_results

def barh_plot(df, score_col, std_col, label_col, title_text, xlabel, path):
    d = df.copy().sort_values(score_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(d))
    ax.barh(y, d[score_col], xerr=d[std_col], alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(d[label_col])
    ax.set_xlabel(xlabel); ax.set_title(title_text); ax.grid(axis="x", alpha=0.3)
    for i, val in enumerate(d[score_col]):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
def plot_confusion(results, y_test, filename, suptitle):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ConfusionMatrixDisplay(confusion_matrix(y_test, r["_y_pred"]), display_labels=["Benign", "Malignant"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{r['Model']}\n{r['Strategy']}\nthr={r['Threshold']:.2f}, P={r['Precision']:.2f}, R={r['Recall']:.2f}, F1={r['F1']:.2f}", fontsize=9)
    fig.suptitle(suptitle, fontsize=13, y=1.05)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight"); plt.close(fig)
def make_plots(models, main_results, all_results, baselines, best_name, best_model, best_results, X_test, y_test, selected_features):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in main_results:
        fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
        ax.plot(fpr, tpr, lw=2, label=f"{r['Model']} (AUC = {r['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves — Thyroid Cancer Diagnosis")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in main_results:
        p, rec, _ = precision_recall_curve(y_test, r["_y_proba"])
        ax.plot(rec, p, lw=2, label=r["Model"])
    for marker, r in zip(["o", "s", "^"], best_results):
        ax.scatter(r["Recall"], r["Precision"], s=90, marker=marker, edgecolor="black", label=f"{best_name} — {r['Strategy']}")
    ax.axhline(y_test.mean(), ls="--", lw=1, label=f"Random baseline precision = {y_test.mean():.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Trade-off and Threshold Strategies")
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curves.png"), dpi=150); plt.close(fig)
    plot_confusion(main_results, y_test, "confusion_matrices_models.png", f"Confusion Matrices — model comparison, threshold mode: {MAIN_THRESHOLD_MODE}")
    plot_confusion(best_results, y_test, "confusion_matrices_best_model_strategies.png", f"Confusion Matrices — {best_name} under threshold strategies")
    metrics = ["ROC-AUC", "Average Precision", "F1", "Precision", "Recall", "Accuracy"]
    labels = [r["Model"] for r in main_results] + [b["Model"] for b in baselines]
    vals = [[r[m] for m in metrics] for r in main_results] + [[b[m] for m in metrics] for b in baselines]
    x = np.arange(len(metrics)); width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (lab, v) in enumerate(zip(labels, vals)):
        bars = ax.bar(x + (i - len(labels) / 2 + 0.5) * width, v, width, label=lab, alpha=0.85)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("Score")
    ax.set_title(f"Model Performance vs Naive Baselines — threshold mode: {MAIN_THRESHOLD_MODE}")
    ax.legend(ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "metric_comparison.png"), dpi=150); plt.close(fig)
    proba = best_model.predict_proba(X_test[selected_features])[:, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(proba[y_test == 0], bins=40, density=True, alpha=0.55, label="Benign")
    ax.hist(proba[y_test == 1], bins=40, density=True, alpha=0.55, label="Malignant")
    ax.set(xlabel="Predicted probability of malignant", ylabel="Density", title=f"Probability overlap — {best_name}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "probability_overlap_best_model.png"), dpi=150); plt.close(fig)
    importance = getattr(best_model, "feature_importances_", None)
    if importance is None and hasattr(best_model, "coef_"):
        importance = np.abs(best_model.coef_[0])
    if importance is not None:
        imp = pd.DataFrame({"Feature": selected_features, "Importance": importance}).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(imp))))
        ax.barh(imp["Feature"], imp["Importance"], alpha=0.85)
        ax.set_xlabel("Importance" if hasattr(best_model, "feature_importances_") else "|Coefficient|")
        ax.set_title(f"Feature importance — {best_name}"); ax.grid(axis="x", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance_best_model.png"), dpi=150); plt.close(fig)
    p, rec, thr = precision_recall_curve(y_test, proba)
    p, rec = p[:-1], rec[:-1]
    f1 = 2 * p * rec / (p + rec + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, p, label="Precision"); ax.plot(thr, rec, label="Recall"); ax.plot(thr, f1, label="F1")
    for r in best_results:
        ax.axvline(r["Threshold"], ls="--", alpha=0.7, label=f"{r['Strategy']} thr={r['Threshold']:.2f}")
    ax.set(xlabel="Threshold", ylabel="Score", title=f"Threshold Diagnostics — {best_name}")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "threshold_diagnostics_best_model.png"), dpi=150); plt.close(fig)
def class_weight_sensitivity(X_train, y_train, X_test, y_test, cv):
    configs = [("None (1:1)", None), ("Mild (1:2)", {0: 1, 1: 2}), ("Moderate (1:3)", {0: 1, 1: 3}), ("Balanced (auto)", "balanced"), ("Strong (1:5)", {0: 1, 1: 5}), ("Aggressive (1:7)", {0: 1, 1: 7})]
    fixed, tuned = [], []
    for label, weight in configs:
        model = LogisticRegression(class_weight=weight, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fixed.append((label, precision_score(y_test, pred, zero_division=0), recall_score(y_test, pred, zero_division=0), f1_score(y_test, pred, zero_division=0)))
        strat = choose_threshold(cv_probs(model, X_train, y_train, cv), y_train, "f1")
        pred = (proba >= strat["threshold"]).astype(int)
        tuned.append((label, precision_score(y_test, pred, zero_division=0), recall_score(y_test, pred, zero_division=0), f1_score(y_test, pred, zero_division=0)))
    rows = []
    for mode, data in [("fixed_0.5", fixed), ("f1_tuned", tuned)]:
        for label, p, r, f in data:
            rows.append({"Mode": mode, "Class weight": label, "Precision": p, "Recall": r, "F1": f})
    df = pd.DataFrame(rows)
    save_csv(df, "class_weight_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, data, title_text in zip(axes, [fixed, tuned], ["Fixed threshold = 0.5", "F1-tuned threshold"]):
        labels = [x[0] for x in data]; x = np.arange(len(labels)); width = 0.25
        for i, metric in enumerate(["Precision", "Recall", "F1"]):
            vals = [row[i + 1] for row in data]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=metric, alpha=0.85)
            if metric == "F1":
                for b, val in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title_text); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score"); axes[1].legend()
    fig.suptitle("Class-weight sensitivity: threshold retuning can neutralise class_weight effects", y=1.03)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "class_weight_sensitivity.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    return df
def main() -> None:
    title("P3 — MODEL TRAINING AND EVALUATION")
    result = load_and_preprocess(DATA_PATH, random_state=RANDOM_STATE, test_size=TEST_SIZE, verbose=True)
    if len(result) == 6:
        X_train, X_test, y_train, y_test, feature_names, scaler = result
    else:
        X_train, X_test, y_train, y_test, feature_names = result
        scaler = None
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_benign, n_malignant = int((y_train == 0).sum()), int((y_train == 1).sum())
    imbal_ratio = n_benign / max(n_malignant, 1)
    print(f"\nClass imbalance — Benign: {n_benign:,}, Malignant: {n_malignant:,}, ratio neg/pos: {imbal_ratio:.2f}")
    selector = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    subtitle("Single-feature signal report")
    signal_df = single_feature_report(X_train, y_train, cv)
    print(signal_df.to_string(index=False)); save_csv(signal_df, "single_feature_signal_report.csv")
    subtitle("Forward/backward feature selection")
    forward_features, forward_df = forward_selection(X_train, y_train, feature_names, selector, cv)
    backward_features, backward_df = (backward_elimination(X_train, y_train, feature_names, selector, cv) if RUN_BACKWARD_SELECTION else (feature_names, pd.DataFrame()))
    save_csv(forward_df, "forward_feature_selection.csv"); save_csv(backward_df, "backward_feature_elimination.csv")
    selected_set, selected_features, feature_df = compare_feature_sets(X_train, y_train, feature_names, forward_features, backward_features, selector, cv, FEATURE_SELECTION_SCORING)
    print(feature_df.to_string(index=False)); print(f"\nChosen: {selected_set} — {len(selected_features)} features")
    save_csv(feature_df, "feature_set_comparison.csv")
    barh_plot(feature_df, f"CV {FEATURE_SELECTION_SCORING} mean", f"CV {FEATURE_SELECTION_SCORING} std", "Feature set", "Feature selection comparison", f"CV {FEATURE_SELECTION_SCORING}", os.path.join(OUTPUT_DIR, "feature_selection_comparison.png"))
    Xtr, Xte = X_train[selected_features].copy(), X_test[selected_features].copy()

    # Leakage-free sanity check: feature selection is repeated inside outer CV folds.
    nested_cv_df = pd.DataFrame()
    if RUN_NESTED_CV_CHECK:
        subtitle("Nested-CV check for feature selection leakage")
        nested_reports = []
        for direction in ["forward", "backward"]:
            report = nested_cv_feature_selection(X_train, y_train, feature_names, selector, direction, FEATURE_SELECTION_SCORING)
            nested_reports.append(report)
            print(f"{direction}: AP = {report['Outer CV AP mean']:.5f} ± {report['Outer CV AP std']:.5f}")
            print(f"Features on full training set: {report['Features when fit on full training set']}")
        nested_cv_df = pd.DataFrame(nested_reports)
        save_csv(nested_cv_df, "nested_cv_feature_selection.csv")

    subtitle("Training models on selected features")
    models = train_model_family(Xtr, y_train, imbal_ratio, label="selected features")

    subtitle("Training all-features baseline models")
    all_feature_models = train_model_family(X_train, y_train, imbal_ratio, label="all features")

    subtitle("Threshold tuning and evaluation")
    all_results, main_results = evaluate_model_family(models, Xtr, y_train, Xte, y_test, cv, "selected features")
    all_feature_results, all_feature_main_results = evaluate_model_family(all_feature_models, X_train, y_train, X_test, y_test, cv, "all features")
    baselines = baseline_results(Xtr, y_train, Xte, y_test)
    metrics_df = pd.DataFrame([clean_result_dict(r) for r in main_results]).set_index("Model")
    strategy_df = pd.DataFrame([clean_result_dict(r) for r in all_results])
    baseline_df = pd.DataFrame([clean_result_dict(r) for r in baselines])
    all_feature_metrics_df = pd.DataFrame(
        [clean_result_dict(r) for r in all_feature_main_results]
    ).set_index("Model")

    all_feature_strategy_df = pd.DataFrame(
        [clean_result_dict(r) for r in all_feature_results]
    )

    comparison_df = pd.concat(
        [
            pd.DataFrame([clean_result_dict(r) for r in main_results]),
            pd.DataFrame([clean_result_dict(r) for r in all_feature_main_results]),
            pd.DataFrame([clean_result_dict(r) for r in baselines]),
        ],
        ignore_index=True,
    )
    print(metrics_df.to_string())
    save_csv(metrics_df.reset_index(), "model_metrics_main_threshold.csv")
    save_csv(strategy_df, "model_metrics_all_thresholds.csv")
    save_csv(baseline_df, "baseline_metrics.csv")
    save_csv(all_feature_metrics_df.reset_index(), "all_features_model_metrics_main_threshold.csv")
    save_csv(all_feature_strategy_df, "all_features_model_metrics_all_thresholds.csv")
    save_csv(comparison_df, "selected_vs_all_features_vs_dummy_comparison.csv")

    print("\nSelected features vs all features vs dummy baselines:")
    print(comparison_df.to_string(index=False))
    best_name = str(metrics_df[BEST_SELECTION_METRIC].idxmax())
    best_model = models[best_name]
    best_results = [r for r in all_results if r["Model"] == best_name]
    print(f"\nBest model by {BEST_SELECTION_METRIC}: {best_name}")
    subtitle("Class-weight sensitivity")
    class_weight_sensitivity(Xtr, y_train, Xte, y_test, cv)
    subtitle("Saving plots and artefacts")
    make_plots(models, main_results, all_results, baselines, best_name, best_model, best_results, X_test, y_test, selected_features)
    artefacts = {
        "selected_feature_set": selected_set,
        "selected_features": selected_features,
        "main_threshold_mode": MAIN_THRESHOLD_MODE,
        "best_model": best_name,
        "best_metrics": clean_result_dict(metrics_df.reset_index().query("Model == @best_name").iloc[0].to_dict()),
        "scaler_saved": scaler is not None,
        "note": "Scaler is only saved if src.preprocessing.load_and_preprocess returns it as 6th object.",
        "methodology": {
            "train_test_split": "Stratified 80/20 split from preprocessing; held-out test set used only for final evaluation.",
            "feature_selection": "Forward/backward selection performed on training data; nested-CV check repeats FS inside outer folds to document leakage-free estimate.",
            "all_features_baseline": "LogReg/RF/XGB also evaluated with all features using the same CV-based threshold tuning.",
            "class_imbalance": "LogReg/RF use class_weight='balanced'; XGBoost uses scale_pos_weight; threshold tuning uses CV probabilities.",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "model_summary.json"), "w") as f:
        json.dump(artefacts, f, indent=2)
    joblib.dump({"models": models, "best_model": best_model, "selected_features": selected_features, "scaler": scaler}, os.path.join(OUTPUT_DIR, "trained_models.joblib"))
    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")
if __name__ == "__main__":
    main()
