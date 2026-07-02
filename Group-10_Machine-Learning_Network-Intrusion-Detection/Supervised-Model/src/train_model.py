import subprocess
import pandas as pd
import numpy as np
import joblib
import optuna
import sys

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# =========================
# RUN DATASET HANDLER
# =========================
print("\nRunning handle_dataset.py...\n")

subprocess.run([sys.executable, "handle_dataset.py"], check=True)

print("\nDataset preparation completed!\n")


# =========================
# LOAD TRAIN / TEST DATA
# =========================
train_df = pd.read_csv(
    "../Dataset/Combined_Dataset/CICIDS2017_TRAIN.csv"
)

test_df = pd.read_csv(
    "../Dataset/Combined_Dataset/CICIDS2017_TEST.csv"
)

print("Train Dataset Shape:", train_df.shape)
print("Test Dataset Shape:", test_df.shape)

print("\nTrain Label Distribution:")
print(train_df["Label"].value_counts())

print("\nTest Label Distribution:")
print(test_df["Label"].value_counts())


# =========================
# CLEAN DATA
# =========================
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)


# =========================
# SPLIT FEATURES / LABELS
# =========================
X_train = train_df.drop("Label", axis=1)
y_train_raw = train_df["Label"]

X_test = test_df.drop("Label", axis=1)
y_test_raw = test_df["Label"]


# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()

y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

print("\nLabel Mapping:")

for i, cls in enumerate(le.classes_):
    print(i, "=", cls)


# =========================
# FEATURE COLUMNS
# =========================
feature_columns = X_train.columns.tolist()

print("\nTotal Features:", len(feature_columns))


# =========================
# SMALL SAMPLE FOR OPTUNA
# =========================
sample_df = train_df.sample(
    n=min(200000, len(train_df)),
    random_state=42
)

X_sample = sample_df.drop("Label", axis=1)

y_sample = le.transform(sample_df["Label"])


# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            300
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            4,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.2
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss"
    }

    model = XGBClassifier(**params)

    score = cross_val_score(
        model,
        X_sample,
        y_sample,
        cv=3,
        scoring="f1_macro"
    ).mean()

    return score


# =========================
# RUN OPTUNA
# =========================
print("\nRunning Optuna Hyperparameter Tuning...")

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=10
)

print("\nBest Parameters:")
print(study.best_params)


# =========================
# RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

rf.fit(X_train, y_train)


# =========================
# TUNED XGBOOST
# =========================
best_params = study.best_params

xgb = XGBClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
)

print("\nTraining Tuned XGBoost...")

xgb.fit(X_train, y_train)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, X_test, y_test, name):

    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# =========================
# SAVE MODELS
# =========================
joblib.dump(
    rf,
    "../Model/rf_model.pkl"
)

joblib.dump(
    xgb,
    "../Model/xgb_model.pkl"
)

joblib.dump(
    le,
    "../Model/label_encoder.pkl"
)

joblib.dump(
    feature_columns,
    "../Model/feature_columns.pkl"
)

print("\nModels saved successfully!")
print("Feature Count:", len(feature_columns))