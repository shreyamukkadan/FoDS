"""
P4: Feature selection and K-Means clustering.

This script:
1. Loads the same preprocessed data used in P3.
2. Loads the best model saved by P3.
3. Runs forward feature selection.
4. Runs backward feature selection.
5. Runs K-Means clustering with k=2.
6. Saves tables and plots to outputs/p4.
"""

# =============================================================================
# 1. IMPORTS AND CONFIG
# =============================================================================

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

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_PATH = "data/thyroid_cancer_risk_data.csv"
P3_OUTPUT_DIR = "outputs/p3"
P4_OUTPUT_DIR = "outputs/p4"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MIN_IMPROVEMENT = 0.001

os.makedirs(P4_OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. LOAD AND PREPROCESS DATA
# =============================================================================

def load_and_preprocess(data_path, random_state=42, test_size=0.2, verbose=True):
    """
    Same preprocessing structure used in P3.

    Returns scaled train/test features, labels, and feature names.
    Diagnosis is the target and is not included in X.
    """
    continuous_cols = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
    binary_cols = [
        "Family_History",
        "Radiation_Exposure",
        "Iodine_Deficiency",
        "Smoking",
        "Obesity",
        "Diabetes",
    ]
    engineered_cols = [
        "TSH_T3_ratio",
        "TSH_T4_ratio",
        "T3_T4_ratio",
        "Risk_count",
        "Age_Nodule",
    ]

    df = pd.read_csv(data_path)
    rows_before_null = len(df)
    df.dropna(inplace=True)

    z = df[continuous_cols].apply(zscore)
    mask_clean = (z.abs() <= 3).all(axis=1)
    df_clean = df[mask_clean].copy()

    df_model = df_clean.copy()
    df_model.drop(
        columns=["Patient_ID", "Country", "Ethnicity", "Thyroid_Cancer_Risk"],
        inplace=True,
    )

    for col in binary_cols:
        df_model[col] = (df_model[col] == "Yes").astype(int)

    df_model["Gender"] = (df_model["Gender"] == "Male").astype(int)

    df_model["TSH_T3_ratio"] = df_model["TSH_Level"] / (df_model["T3_Level"] + 1e-6)
    df_model["TSH_T4_ratio"] = df_model["TSH_Level"] / (df_model["T4_Level"] + 1e-6)
    df_model["T3_T4_ratio"] = df_model["T3_Level"] / (df_model["T4_Level"] + 1e-6)
    df_model["Risk_count"] = df_model[binary_cols].sum(axis=1)
    df_model["Age_Nodule"] = df_model["Age"] * df_model["Nodule_Size"]

    df_model["Diagnosis"] = (df_model["Diagnosis"] == "Malignant").astype(int)

    X = df_model.drop(columns=["Diagnosis"])
    y = df_model["Diagnosis"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    cols_to_scale = continuous_cols + engineered_cols

    X_train_sc = X_train.copy()
    X_test_sc = X_test.copy()

    X_train_sc[cols_to_scale] = X_train_sc[cols_to_scale].astype(float)
    X_test_sc[cols_to_scale] = X_test_sc[cols_to_scale].astype(float)

    X_train_sc[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_sc[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    if verbose:
        print(f"Loaded {rows_before_null:,} rows.")
        print(f"After null removal and outlier filtering: {len(df_model):,} rows.")
        print(f"Features used for P4: {feature_names}")
        print(f"Training rows: {len(X_train_sc):,}")
        print(f"Test rows: {len(X_test_sc):,}")

    return X_train_sc, X_test_sc, y_train, y_test, feature_names

# =============================================================================
# 3. LOAD BEST P3 MODEL
# =============================================================================

def load_best_p3_model():
    """
    Load the single best model selected by P3.
    """
    meta_path = os.path.join(P3_OUTPUT_DIR, "best_model_meta.json")

    with open(meta_path, "r") as fh:
        meta = json.load(fh)

    model = joblib.load(meta["model_path"])

    print("\nBest P3 model loaded:")
    print(f"  Model:     {meta['name']}")
    print(f"  F1:        {meta['f1']}")
    print(f"  ROC-AUC:   {meta['roc_auc']}")
    print(f"  Threshold: {meta['threshold']}")

    return meta, model


def best_cv_f1_for_features(model, X_train, y_train, features, cv):
    """
    Evaluate one feature subset using cross-validated probabilities.

    The threshold is chosen on cross-validated training probabilities.
    This keeps feature selection based only on training data.
    """
    candidate_model = clone(model)
    X_subset = X_train[features]

    probabilities = cross_val_predict(
        candidate_model,
        X_subset,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, probabilities)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = f1_scores[:-1].argmax()
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_f1, best_threshold

# =============================================================================
# 4. FORWARD FEATURE SELECTION
# =============================================================================


def forward_selection(model, X_train, y_train, feature_names, cv):
    """
    Start with no features.
    Add the feature that improves CV F1 the most at each step.
    Stop when the improvement is smaller than MIN_IMPROVEMENT.
    """
    selected = []
    remaining = list(feature_names)
    history = []
    best_score = 0.0

    print("\nRunning forward selection...")

    while remaining:
        candidates = []

        for feature in remaining:
            trial_features = selected + [feature]
            score, threshold = best_cv_f1_for_features(
                model,
                X_train,
                y_train,
                trial_features,
                cv,
            )

            candidates.append({
                "candidate_feature": feature,
                "n_features": len(trial_features),
                "features": ", ".join(trial_features),
                "cv_f1": score,
                "threshold": threshold,
            })

        candidate_df = pd.DataFrame(candidates).sort_values(
            "cv_f1",
            ascending=False,
        )

        best_candidate = candidate_df.iloc[0]
        improvement = best_candidate["cv_f1"] - best_score

        if improvement < MIN_IMPROVEMENT:
            print(
                f"  Stopping: best improvement {improvement:.4f} "
                f"is below {MIN_IMPROVEMENT}."
            )
            break

        selected.append(best_candidate["candidate_feature"])
        remaining.remove(best_candidate["candidate_feature"])
        best_score = best_candidate["cv_f1"]

        history.append({
            "step": len(selected),
            "added_feature": best_candidate["candidate_feature"],
            "cv_f1": round(best_score, 4),
            "threshold": round(best_candidate["threshold"], 4),
            "selected_features": ", ".join(selected),
        })

        print(
            f"  Step {len(selected):2d}: add {best_candidate['candidate_feature']:<25s} "
            f"CV F1 = {best_score:.4f}"
        )

    return selected, pd.DataFrame(history)



# =============================================================================
# 5. BACKWARD FEATURE SELECTION
# =============================================================================

def backward_selection(model, X_train, y_train, feature_names, cv):
    """
    Start with all features.
    Remove the feature whose removal hurts CV F1 the least.
    Stop when removing any feature makes performance meaningfully worse.
    """
    selected = list(feature_names)
    history = []

    print("\nRunning backward selection...")

    current_score, current_threshold = best_cv_f1_for_features(
        model,
        X_train,
        y_train,
        selected,
        cv,
    )

    print(f"  Start with all features: CV F1 = {current_score:.4f}")

    while len(selected) > 1:
        candidates = []

        for feature in selected:
            trial_features = [f for f in selected if f != feature]
            score, threshold = best_cv_f1_for_features(
                model,
                X_train,
                y_train,
                trial_features,
                cv,
            )

            candidates.append({
                "removed_feature": feature,
                "n_features": len(trial_features),
                "features": ", ".join(trial_features),
                "cv_f1": score,
                "threshold": threshold,
                "score_change": score - current_score,
            })

        candidate_df = pd.DataFrame(candidates).sort_values(
            "cv_f1",
            ascending=False,
        )

        best_candidate = candidate_df.iloc[0]
        score_drop = current_score - best_candidate["cv_f1"]

        if score_drop > MIN_IMPROVEMENT:
            print(
                f"  Stopping: removing another feature drops CV F1 by "
                f"{score_drop:.4f}, more than {MIN_IMPROVEMENT}."
            )
            break

        selected.remove(best_candidate["removed_feature"])
        current_score = best_candidate["cv_f1"]
        current_threshold = best_candidate["threshold"]

        history.append({
            "step": len(history) + 1,
            "removed_feature": best_candidate["removed_feature"],
            "cv_f1": round(current_score, 4),
            "threshold": round(current_threshold, 4),
            "remaining_features": ", ".join(selected),
        })

        print(
            f"  Step {len(history):2d}: remove {best_candidate['removed_feature']:<25s} "
            f"CV F1 = {current_score:.4f}"
        )

    return selected, pd.DataFrame(history)


def evaluate_selected_features(model, X_train, X_test, y_train, y_test, features):
    """
    Fit the best P3 model on selected features and evaluate on the held-out test set.
    """
    fitted_model = clone(model)
    fitted_model.fit(X_train[features], y_train)

    train_probabilities = fitted_model.predict_proba(X_train[features])[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, train_probabilities)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1_scores[:-1].argmax()
    threshold = float(thresholds[best_idx])

    test_probabilities = fitted_model.predict_proba(X_test[features])[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_f1 = f1_score(y_test, test_predictions)

    return {
        "n_features": len(features),
        "features": ", ".join(features),
        "threshold": round(threshold, 4),
        "test_f1": round(float(test_f1), 4),
    }

# =============================================================================
# 6. SAVE FEATURE SELECTION RESULTS
# =============================================================================

def save_feature_selection_outputs(
    forward_features,
    forward_history,
    backward_features,
    backward_history,
    feature_names,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Save feature selection tables for the report.
    """
    forward_history.to_csv(
        os.path.join(P4_OUTPUT_DIR, "forward_selection_history.csv"),
        index=False,
    )

    backward_history.to_csv(
        os.path.join(P4_OUTPUT_DIR, "backward_selection_history.csv"),
        index=False,
    )

    summary_rows = []
    for feature in feature_names:
        in_forward = feature in forward_features
        in_backward = feature in backward_features

        summary_rows.append({
            "Feature": feature,
            "Forward Selection": "Yes" if in_forward else "No",
            "Backward Selection": "Yes" if in_backward else "No",
            "Selected By Both": "Yes" if in_forward and in_backward else "No",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(P4_OUTPUT_DIR, "feature_selection_summary.csv"),
        index=False,
    )

    eval_df = pd.DataFrame([
        {
            "method": "Forward Selection",
            **evaluate_selected_features(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                forward_features,
            ),
        },
        {
            "method": "Backward Selection",
            **evaluate_selected_features(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                backward_features,
            ),
        },
    ])

    eval_df.to_csv(
        os.path.join(P4_OUTPUT_DIR, "selected_feature_test_performance.csv"),
        index=False,
    )

    print("\nFeature selection outputs saved:")
    print(f"  {P4_OUTPUT_DIR}/forward_selection_history.csv")
    print(f"  {P4_OUTPUT_DIR}/backward_selection_history.csv")
    print(f"  {P4_OUTPUT_DIR}/feature_selection_summary.csv")
    print(f"  {P4_OUTPUT_DIR}/selected_feature_test_performance.csv")

    print("\nFeatures selected by forward selection:")
    print(forward_features)

    print("\nFeatures selected by backward selection:")
    print(backward_features)

    print("\nFeatures selected by both methods:")
    print(summary_df.loc[summary_df["Selected By Both"] == "Yes", "Feature"].tolist())

# =============================================================================
# 7. K-MEANS CLUSTERING
# =============================================================================


def run_kmeans_clustering(X_all, y_all):
    """
    Run K-Means with k=2 without using Diagnosis as an input feature.

    Diagnosis is used only after clustering to evaluate whether the clusters
    naturally align with Benign and Malignant cases.
    """
    print("\nRunning K-Means clustering with k=2...")

    kmeans = KMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    clusters = kmeans.fit_predict(X_all)

    cluster_df = pd.DataFrame({
        "Diagnosis": y_all.map({0: "Benign", 1: "Malignant"}).values,
        "Diagnosis_Binary": y_all.values,
        "Cluster": clusters,
    })

    crosstab = pd.crosstab(
        cluster_df["Cluster"],
        cluster_df["Diagnosis"],
    )

    crosstab["Majority Diagnosis"] = crosstab.idxmax(axis=1)

    cluster_to_label = {
        cluster: 0 if majority == "Benign" else 1
        for cluster, majority in crosstab["Majority Diagnosis"].items()
    }

    predicted_diagnosis = pd.Series(clusters).map(cluster_to_label).values
    majority_alignment = (predicted_diagnosis == y_all.values).mean()

    ari = adjusted_rand_score(y_all, clusters)
    nmi = normalized_mutual_info_score(y_all, clusters)

    metrics_df = pd.DataFrame([{
        "k": 2,
        "majority_label_alignment": round(float(majority_alignment), 4),
        "adjusted_rand_index": round(float(ari), 4),
        "normalized_mutual_information": round(float(nmi), 4),
    }])

    crosstab.to_csv(os.path.join(P4_OUTPUT_DIR, "kmeans_cluster_crosstab.csv"))
    metrics_df.to_csv(os.path.join(P4_OUTPUT_DIR, "kmeans_metrics.csv"), index=False)

    print("\nK-Means cluster vs diagnosis table:")
    print(crosstab)

    print("\nK-Means alignment metrics:")
    print(metrics_df.to_string(index=False))

    plot_kmeans_cluster_bars(crosstab)
    plot_kmeans_pca(X_all, clusters, y_all)

    print("\nK-Means outputs saved:")
    print(f"  {P4_OUTPUT_DIR}/kmeans_cluster_crosstab.csv")
    print(f"  {P4_OUTPUT_DIR}/kmeans_metrics.csv")
    print(f"  {P4_OUTPUT_DIR}/kmeans_cluster_diagnosis_bar.png")
    print(f"  {P4_OUTPUT_DIR}/kmeans_pca_clusters.png")


def plot_kmeans_cluster_bars(crosstab):
    """
    Bar plot showing Benign/Malignant counts in each cluster.
    """
    plot_df = crosstab.drop(columns=["Majority Diagnosis"])

    ax = plot_df.plot(
        kind="bar",
        stacked=False,
        figsize=(8, 5),
        color=["#4C72B0", "#C44E52"],
    )

    ax.set_title("K-Means Clusters Compared With Diagnosis")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Patients")
    ax.legend(title="Diagnosis")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(P4_OUTPUT_DIR, "kmeans_cluster_diagnosis_bar.png"),
        dpi=150,
    )
    plt.close()


def plot_kmeans_pca(X_all, clusters, y_all):
    """
    PCA scatterplot for visualizing k=2 clusters in two dimensions.
    PCA is only for visualization, not for fitting K-Means.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_all)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter_1 = axes[0].scatter(
        components[:, 0],
        components[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.5,
        s=12,
    )
    axes[0].set_title("K-Means Clusters")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(scatter_1, ax=axes[0], label="Cluster")

    scatter_2 = axes[1].scatter(
        components[:, 0],
        components[:, 1],
        c=y_all,
        cmap="coolwarm",
        alpha=0.5,
        s=12,
    )
    axes[1].set_title("True Diagnosis Labels")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(scatter_2, ax=axes[1], label="Diagnosis: 0=Benign, 1=Malignant")

    fig.suptitle("PCA View of K-Means Clusters vs True Diagnosis")
    plt.tight_layout()
    plt.savefig(
        os.path.join(P4_OUTPUT_DIR, "kmeans_pca_clusters.png"),
        dpi=150,
    )
    plt.close()

# =============================================================================
# 8. MAIN SCRIPT
# =============================================================================


def main():
    print("=" * 70)
    print("P4 — FEATURE SELECTION AND K-MEANS CLUSTERING")
    print("=" * 70)

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        DATA_PATH,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        verbose=True,
    )

    best_model_meta, best_model = load_best_p3_model()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    forward_features, forward_history = forward_selection(
        best_model,
        X_train,
        y_train,
        feature_names,
        cv,
    )

    backward_features, backward_history = backward_selection(
        best_model,
        X_train,
        y_train,
        feature_names,
        cv,
    )

    save_feature_selection_outputs(
        forward_features,
        forward_history,
        backward_features,
        backward_history,
        feature_names,
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)

    run_kmeans_clustering(X_all, y_all)

    print("\nP4 complete.")


if __name__ == "__main__":
    main()
