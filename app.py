#import the packages
import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)




# Configuration
# ==========================================

DATA_PATH = "/content/genz_college.parquet"   # Path in Colab -- update to your actual filename
TARGET_COL = "admission_status"                  # Your target column

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)






# Load Dataset
# ==========================================

data = (
    pd.read_parquet(DATA_PATH, engine="pyarrow")
      .drop_duplicates()
      .reset_index(drop=True)
)

# Verify target column exists
if TARGET_COL not in data.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

# Separate features and target
X = data.drop(columns=[TARGET_COL])
y_raw = data[TARGET_COL]

# Display dataset information
print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)
print(f"Rows       : {len(data):,}")
print(f"Columns    : {len(data.columns)}")
print(f"Target     : {TARGET_COL}")
print(f"Memory     : {data.memory_usage(deep=True).sum()/1024**2:.2f} MB")
print("=" * 60)

display(data.head())





data.isnull().sum()
data[data.duplicated()]


# Re-separate after dropna (row count may have changed)
X = data.drop(columns=[TARGET_COL])
y_raw = data[TARGET_COL]


data=pd.get_dummies(data,columns=["gender","state"],drop_first=True)

data.head()





# Target Encoding (classification-specific)
# ==========================================
# admission_status may already be 0/1, or it may be text labels
# (e.g. "Admitted"/"Rejected"). LabelEncoder handles both safely,
# and XGBoost/LightGBM/CatBoost require numeric-encoded targets.

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

n_classes = len(label_encoder.classes_)
is_binary = n_classes == 2

print("=" * 60)
print("Target class distribution")
print("=" * 60)
class_counts = pd.Series(y_raw).value_counts()
print(class_counts)
print(f"\nClasses found: {list(label_encoder.classes_)}")
print(f"Encoded as   : {list(range(n_classes))}")
print(f"Problem type : {'Binary' if is_binary else 'Multi-class'} classification")
print("=" * 60)

print("=" * 60)
print("Feature type detection")
print("=" * 60)
print(f"Numeric features ({len(numeric_features)})    : {list(numeric_features)}")
print(f"Categorical features ({len(categorical_features)}): {list(categorical_features)}")
print("=" * 60)

# Preprocessing Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
], n_jobs=-1)


# Models (5 total: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost)
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "XGBClassifier": XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
        eval_metric="logloss" if is_binary else "mlogloss"
    ),
    "LGBMClassifier": LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
    ),
    "CatBoostClassifier": CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.1,
        random_state=RANDOM_STATE, verbose=0
    )
}

# Train/Test Split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Train/Test Split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Evaluation Loop
results = {}
fitted_pipelines = {}
for name, model in models.items():
    print("=" * 80)
    print(f"MODEL: {name}")

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    fitted_pipelines[name] = pipeline

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Probabilities for ROC-AUC (binary: positive class column; multi-class: all columns)
    try:
        y_test_proba = pipeline.predict_proba(X_test)
        if is_binary:
            test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
        else:
            test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class="ovr")
    except Exception as e:
        test_roc_auc = np.nan
        print(f"  (ROC-AUC unavailable: {e})")

    avg_method = "binary" if is_binary else "weighted"

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average=avg_method)
    test_f1 = f1_score(y_test, y_test_pred, average=avg_method)
    test_precision = precision_score(y_test, y_test_pred, average=avg_method, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=avg_method, zero_division=0)

    results[name] = {
        "train_acc": train_acc, "test_acc": test_acc,
        "train_f1": train_f1, "test_f1": test_f1,
        "test_precision": test_precision, "test_recall": test_recall,
        "test_roc_auc": test_roc_auc
    }

    print(f"[{name}] Train Acc    : {train_acc:.4f}")
    print(f"[{name}] Test  Acc    : {test_acc:.4f}")
    print(f"[{name}] Train F1     : {train_f1:.4f}")
    print(f"[{name}] Test  F1     : {test_f1:.4f}")
    print(f"[{name}] Test  Prec   : {test_precision:.4f}")
    print(f"[{name}] Test  Recall : {test_recall:.4f}")
    print(f"[{name}] Test  ROC-AUC: {test_roc_auc:.4f}")
    print(f"\n[{name}] Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"\n[{name}] Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=[str(c) for c in label_encoder.classes_]))

    joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.joblib"))

# Save label encoder too, since predictions come back as 0/1/2... not original labels
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))


# ==========================================
# Summary comparison across all models
# ==========================================
print("=" * 80)
print("SUMMARY: All Models Compared")
print("=" * 80)
summary_df = pd.DataFrame(results).T
summary_df = summary_df.sort_values("test_f1", ascending=False)
display(summary_df)

best_model_name = summary_df.index[0]
print(f"\nBest performer (by Test F1): {best_model_name}")
