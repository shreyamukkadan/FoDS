from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Paths
DATA_PATH = _ROOT / "data" / "thyroid_cancer_risk_data.csv"
OUT_P1 = _ROOT / "outputs" / "p1_preprocessing"

# Core columns
ID_COL = "Patient_ID"
TARGET = "Diagnosis"

# Feature columns
CONTINUOUS = [
    "Age",
    "TSH_Level",
    "T3_Level",
    "T4_Level",
    "Nodule_Size",
]

BINARY = [
    "Family_History",
    "Radiation_Exposure",
    "Iodine_Deficiency",
    "Smoking",
    "Obesity",
    "Diabetes",
]

GENDER = "Gender"
RISK = "Thyroid_Cancer_Risk"
COUNTRY = "Country"
ETHNICITY = "Ethnicity"

# Split / preprocessing settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
Z_THRESHOLD = 3

# Risk encoding used by P1 preprocessing
# Options: "onehot" or "ordinal"
RISK_ENCODING = "onehot"

# Feature sets used for P1 summaries / P3 ablations
ABLATION_FEATURE_SETS = [
    "full",
    "restricted",
    "risk_only",
    "risk_plus_groups",
    "groups_only",
    "continuous_only",
    "binary_clinical",
]

FEATURE_SET_DESCRIPTIONS = {
    "full": "All available variables.",
    "restricted": "All variables except Country, Ethnicity, and Thyroid_Cancer_Risk.",
    "risk_only": "Thyroid_Cancer_Risk only.",
    "risk_plus_groups": "Thyroid_Cancer_Risk plus Country and Ethnicity.",
    "groups_only": "Country and Ethnicity only.",
    "continuous_only": "Continuous clinical measurements only.",
    "binary_clinical": "Binary clinical/lifestyle flags plus Gender.",
}
