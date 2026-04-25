"""
Thyroid Cancer Risk — Full Analysis Pipeline
Foundations of Data Science Group Project
Person 1: Data, EDA & Preprocessing

Research Question:
    Can a machine learning model accurately classify thyroid cancer cases as
    Benign vs. Malignant, and which demographic and clinical risk factors are
    most strongly associated with the diagnosis?

Structure (Person 1):
    0. Setup & data loading
    1. EDA & visualization (Figures 1–5)
       NEW: Outlier detection (IQR + Z-score) before Fig. 2
       NEW: Fig. 2 is now a KDE plot on cleaned data (no outliers)
    2. Preprocessing
       CHANGED: Thyroid_Cancer_Risk excluded from feature matrix X

Requirements:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
"""

# =============================================================================
# 0. SETUP & DATA LOADING
# =============================================================================

# We import all the libraries we need up front.
# scipy.stats gives us the Z-score function for outlier detection.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Adjust this path to your CSV location ─────────────────────────────────────
DATA_PATH = "../data/thyroid_cancer_risk_data.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("THYROID CANCER RISK — DATA SCIENCE PROJECT")
print("=" * 65)
print(f"\nDataset shape:  {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop the rows with missing values (0.006% — negligible, so dropping is safe)
df.dropna(inplace=True)
print(f"\nAfter dropping nulls: {len(df):,} rows remain")
print(f"\nDiagnosis distribution:\n{df['Diagnosis'].value_counts()}")
print(df['Diagnosis'].value_counts(normalize=True).mul(100).round(1).to_string(), "%")


# =============================================================================
# 1. EDA & VISUALIZATION
# =============================================================================

PALETTE     = {"Benign": "#4C72B0", "Malignant": "#DD8452"}
RISK_ORDER  = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#55A868", "Medium": "#E8A838", "High": "#C44E52"}

continuous_cols = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
binary_cols     = ["Family_History", "Radiation_Exposure", "Iodine_Deficiency",
                   "Smoking", "Obesity", "Diabetes"]


# ── 1.1 Target and risk category distributions (Fig. 1) ──────────────────────
# This figure shows us how many Benign vs Malignant patients are in the dataset
# and how patients are distributed across the three risk categories.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

diag_counts = df["Diagnosis"].value_counts()
axes[0].bar(
    diag_counts.index, diag_counts.values,
    color=[PALETTE[k] for k in diag_counts.index],
    edgecolor="white", width=0.5,
)
axes[0].set_title("Diagnosis distribution", fontsize=12)
axes[0].set_xlabel("Diagnosis")
axes[0].set_ylabel("Number of patients")
for i, (lbl, val) in enumerate(diag_counts.items()):
    axes[0].text(i, val + 600,
                 f"{val:,}\n({val / len(df) * 100:.1f}%)",
                 ha="center", fontsize=10)

risk_counts = df["Thyroid_Cancer_Risk"].value_counts().reindex(RISK_ORDER)
axes[1].bar(
    risk_counts.index, risk_counts.values,
    color=[RISK_COLORS[r] for r in RISK_ORDER],
    edgecolor="white", width=0.5,
)
axes[1].set_title("Risk category distribution", fontsize=12)
axes[1].set_xlabel("Thyroid cancer risk category")
axes[1].set_ylabel("Number of patients")
for i, (lbl, val) in enumerate(risk_counts.items()):
    axes[1].text(i, val + 600,
                 f"{val:,}\n({val / len(df) * 100:.1f}%)",
                 ha="center", fontsize=10)

fig.suptitle("Fig. 1 - Target and risk category distributions", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig1_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig1_distributions.png")


# ── NEW: Outlier Detection (required before Fig. 2) ───────────────────────────
# TA feedback: we need to statistically confirm outliers exist before removing
# them and replotting Fig. 2 as a KDE.
#
# We use two standard tests:
#
# 1. IQR TEST (Interquartile Range)
#    The IQR is the range of the middle 50% of the data (Q3 - Q1).
#    Any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is flagged as an
#    outlier. This is a standard robust method that doesn't assume normality.
#
# 2. Z-SCORE TEST
#    Measures how many standard deviations a value is from the mean.
#    Any value with |Z| > 3 is flagged as an extreme outlier. This is more
#    sensitive to the actual scale of the data.
#
# We run both tests and print a summary table. If both tests flag many values
# in the same column (especially TSH_Level and T4_Level), that justifies
# removing those rows before plotting.

print("\n" + "=" * 65)
print("OUTLIER DETECTION — IQR TEST + Z-SCORE TEST")
print("=" * 65)

outlier_summary = []

for col in continuous_cols:
    series = df[col]

    # IQR test
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()

    # Z-score test
    # We use scipy.stats.zscore which computes (value - mean) / std for each row.
    z_scores      = np.abs(stats.zscore(series))
    zscore_outliers = (z_scores > 3).sum()

    outlier_summary.append({
        "Feature"         : col,
        "Min"             : round(series.min(), 2),
        "Max"             : round(series.max(), 2),
        "IQR lower bound" : round(lower_bound, 2),
        "IQR upper bound" : round(upper_bound, 2),
        "IQR outliers"    : iqr_outliers,
        "Z>3 outliers"    : zscore_outliers,
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))

# Both tests will flag large numbers of outliers in TSH_Level and T4_Level
# (confirmed by their max values of 2000 and 200 vs means of ~5 and ~8).
# This statistically justifies removing those extreme rows before plotting.

# ── Remove outliers using the IQR rule ───────────────────────────────────────
# We remove any row where at least one continuous feature falls outside its
# IQR bounds. We store the cleaned dataframe as df_clean.
# IMPORTANT: df (the original) is kept intact for everything that doesn't
# need cleaned data. df_clean is used only for Fig. 2 and onwards in EDA.

mask_keep = pd.Series([True] * len(df), index=df.index)

for col in continuous_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask_keep = mask_keep & (df[col] >= lower_bound) & (df[col] <= upper_bound)

df_clean = df[mask_keep].copy()

print(f"\nRows before outlier removal: {len(df):,}")
print(f"Rows after  outlier removal: {len(df_clean):,}")
print(f"Rows removed: {len(df) - len(df_clean):,} ({(1 - len(df_clean)/len(df))*100:.2f}%)")


# ── 1.2 NEW: KDE plots of continuous features by diagnosis (Fig. 2) ───────────
# CHANGED FROM PREVIOUS VERSION: We now plot KDE (Kernel Density Estimation)
# curves instead of histograms, using the cleaned data (outliers removed).
#
# A KDE draws a smooth curve showing the shape of the distribution for each
# group (Benign vs Malignant). Unlike a histogram, it is not squashed by
# extreme outlier values on the x-axis, so you can actually see the full
# shape of each feature's distribution and whether the two groups differ.
#
# If the two curves overlap almost completely, the feature doesn't help
# distinguish Benign from Malignant patients on its own.
# If the curves sit apart, the feature is a useful predictor.

fig, axes = plt.subplots(1, len(continuous_cols), figsize=(18, 4))

for ax, col in zip(axes, continuous_cols):
    for label, grp in df_clean.groupby("Diagnosis"):
        # sns.kdeplot draws the smooth density curve.
        # fill=True shades the area under the curve so the overlap is visible.
        # alpha controls transparency so both curves show through each other.
        sns.kdeplot(
            grp[col],
            ax=ax,
            label=label,
            color=PALETTE[label],
            fill=True,
            alpha=0.4,
            linewidth=1.5,
        )
    ax.set_title(col, fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

fig.suptitle(
    "Fig. 2 - Continuous feature distributions by diagnosis (KDE, outliers removed)",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig("fig2_kde_by_diagnosis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig2_kde_by_diagnosis.png")

print("\nMean values of continuous features by diagnosis (cleaned data):")
print(df_clean.groupby("Diagnosis")[continuous_cols].mean().round(3).to_string())


# ── 1.3 Binary/categorical features vs. diagnosis (Fig. 3) ───────────────────
# Six grouped bar charts — one for each Yes/No clinical feature.
# Each pair of bars shows what % of patients in that group (Yes or No)
# are Benign vs Malignant. This uses the original df (not df_clean)
# since these are categorical columns with no outlier issue.

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, col in zip(axes, binary_cols):
    ct = pd.crosstab(df[col], df["Diagnosis"], normalize="index") * 100
    ct.plot(kind="bar", ax=ax,
            color=[PALETTE["Benign"], PALETTE["Malignant"]],
            edgecolor="white", width=0.55)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("% within group")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Diagnosis", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)

fig.suptitle("Fig. 3 - Malignancy rate by lifestyle and clinical risk factors",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig3_binary_features_vs_diagnosis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig3_binary_features_vs_diagnosis.png")


# ── 1.4 Risk category alignment with actual diagnosis (Fig. 4) ───────────────
# Shows what percentage of patients in each risk category (Low/Medium/High)
# are Benign vs Malignant. A stark difference between High and the other two
# categories shows that the risk label is mostly binary (High vs not-High)
# rather than a smooth gradient — which is why we exclude it from ML models.

crosstab_pct = (
    pd.crosstab(df["Thyroid_Cancer_Risk"], df["Diagnosis"], normalize="index")
    .reindex(RISK_ORDER) * 100
)

fig, ax = plt.subplots(figsize=(8, 5))
crosstab_pct.plot(kind="bar", ax=ax,
                  color=[PALETTE["Benign"], PALETTE["Malignant"]],
                  edgecolor="white", width=0.55)
ax.set_title("Malignancy rate by assigned risk category", fontsize=12)
ax.set_xlabel("Thyroid cancer risk category")
ax.set_ylabel("Percentage of patients (%)")
ax.set_xticklabels(RISK_ORDER, rotation=0)
ax.legend(title="Diagnosis")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)

fig.suptitle("Fig. 4 - Risk category vs. actual diagnosis outcome",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig4_risk_category_vs_diagnosis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig4_risk_category_vs_diagnosis.png")


# ── 1.5 Correlation heatmap (Fig. 5) ─────────────────────────────────────────
# Shows pairwise correlations between all numeric features, the binary-encoded
# diagnosis, and the ordinal risk score. Values close to 0 mean no linear
# relationship; values close to 1 or -1 mean a strong relationship.
# The strong correlation between Diagnosis_bin and Risk_ordinal (r=0.37)
# is precisely why we exclude Thyroid_Cancer_Risk from the ML feature matrix.

numeric_heatmap = df[continuous_cols].copy()
numeric_heatmap["Diagnosis_bin"] = (df["Diagnosis"] == "Malignant").astype(int)
numeric_heatmap["Risk_ordinal"]  = df["Thyroid_Cancer_Risk"].map(
    {"Low": 0, "Medium": 1, "High": 2}
)

corr = numeric_heatmap.corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation matrix - numeric features + target + risk", fontsize=12)
fig.suptitle("Fig. 5 - Correlation heatmap", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("fig5_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig5_correlation_heatmap.png")


# =============================================================================
# 2. PREPROCESSING
# =============================================================================
# We now prepare the data for machine learning.
# IMPORTANT CHANGE: Thyroid_Cancer_Risk is NO LONGER included in the feature
# matrix X. Reason: its correlation with Diagnosis (r = 0.37, the strongest
# in the whole heatmap) means any model that sees it would essentially just
# copy that one column rather than learning from the genuine clinical features.
# Excluding it makes the ML results more meaningful and honest.

print("\n" + "=" * 65)
print("PREPROCESSING")

# We use df_clean (outliers removed) as the basis for modelling,
# since we have already confirmed statistically that TSH and T4 outliers
# are extreme and non-representative.
df_model = df_clean.copy()
df_model.drop(columns=["Patient_ID"], inplace=True)

# Step 1: Encode binary Yes/No columns as 0 and 1
# This is the simplest possible encoding for two-value columns.
for col in binary_cols:
    df_model[col] = (df_model[col] == "Yes").astype(int)

# Step 2: Encode Gender as 0/1
df_model["Gender"] = (df_model["Gender"] == "Male").astype(int)

# Step 3: Drop Country and Ethnicity
# These have many categories with no real geographic signal in a synthetic
# dataset. They would create many one-hot columns with very little predictive
# value, adding noise without adding insight.
df_model.drop(columns=["Country", "Ethnicity"], inplace=True)

# Step 4: Drop Thyroid_Cancer_Risk entirely from the model features.
# NEW CHANGE: We drop this column BEFORE creating X so it never enters
# any model. It is kept in the original df for EDA (Figs 1, 4, 5) but
# must not be used as a predictor — it is too closely tied to the outcome.
df_model.drop(columns=["Thyroid_Cancer_Risk"], inplace=True)

# Step 5: Encode the target variable
# "Malignant" becomes 1, "Benign" becomes 0.
df_model["Diagnosis"] = (df_model["Diagnosis"] == "Malignant").astype(int)

# Step 6: Separate features (X) from target (y)
X = df_model.drop(columns=["Diagnosis"])
y = df_model["Diagnosis"]
feature_names = X.columns.tolist()

print(f"\nFeatures ({len(feature_names)}): {feature_names}")
print(f"\nNote: Thyroid_Cancer_Risk excluded — correlation with Diagnosis too high (r=0.37)")
print(f"\nClass balance:")
print(f"  Benign:    {(y == 0).sum():,} ({(y == 0).mean() * 100:.1f}%)")
print(f"  Malignant: {(y == 1).sum():,} ({(y == 1).mean() * 100:.1f}%)")

# Step 7: Stratified 80/20 train/test split
# stratify=y ensures both the training and test sets contain the same
# proportion of Benign vs Malignant cases as the full dataset.
# Without this, random chance could put almost no Malignant cases in one split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train Malignant rate: {y_train.mean() * 100:.1f}%")
print(f"Test  Malignant rate: {y_test.mean() * 100:.1f}%")

# Step 8: StandardScaler — scale continuous features to mean=0, std=1
# We fit the scaler ONLY on training data, then use it to transform both
# training and test data. If we fitted on the full dataset or on the test
# set, future data would "leak" into training — giving artificially good
# results that wouldn't hold in the real world.
scaler = StandardScaler()

X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()

# Convert continuous columns to float so they can hold scaled decimal values.
# Without this, pandas refuses to store floats in integer columns.
X_train_sc[continuous_cols] = X_train_sc[continuous_cols].astype(float)
X_test_sc[continuous_cols]  = X_test_sc[continuous_cols].astype(float)

# fit_transform on training: learns mean/std and scales in one step.
# transform on test: uses the mean/std learned from training ONLY.
X_train_sc[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_sc[continuous_cols]  = scaler.transform(X_test[continuous_cols])

print(f"\nStandardised features: {continuous_cols}")
print("(Scaler fitted on training set only — no data leakage into test set)")

print("\n✅ Preprocessing complete.")
print("The following variables are ready for P2 (stats) and P3 (ML):")
print("   X_train_sc, X_test_sc, y_train, y_test")
print(f"   Feature names: {feature_names}")