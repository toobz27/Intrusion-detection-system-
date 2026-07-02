import pandas as pd
import numpy as np
import joblib
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../Dataset/Combined_Dataset/CICIDS2017_GROUPED.csv")

print("Dataset Shape:", df.shape)
print("\nLabel Distribution:")
print(df["Label"].value_counts())


# =========================
# CLEAN (important for XGBoost/Sklearn)
# =========================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)


# =========================
# SPLIT FEATURES / LABEL
# =========================
X = df.drop("Label", axis=1)
y_raw = df["Label"]


# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
y = le.fit_transform(y_raw)

print("\nLabel Mapping:")
for i, cls in enumerate(le.classes_):
    print(i, "=", cls)


# =========================
# TRAIN TEST SPLIT (FINAL EVAL SET)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# =========================
# SMALL SAMPLE FOR OPTUNA (FAST TUNING)
# =========================
sample_df = df.sample(n=200000, random_state=42)

X_sample = sample_df.drop("Label", axis=1)
y_sample = le.transform(sample_df["Label"])


# =========================
# OPTUNA OBJECTIVE (XGBOOST TUNING)
# =========================
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
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
print("\nRunning Optuna hyperparameter tuning...")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("\nBest Parameters Found:")
print(study.best_params)


# =========================
# FINAL MODELS
# =========================

# Random Forest (baseline)
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)


# Tuned XGBoost
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
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


evaluate(rf, X_test, y_test, "Random Forest")
evaluate(xgb, X_test, y_test, "Tuned XGBoost")


# =========================
# SAVE MODELS (FOR FASTAPI)
# =========================
joblib.dump(rf, "../Model/rf_model.pkl")
joblib.dump(xgb, "../Model/xgb_model.pkl")
joblib.dump(le, "../Model/label_encoder.pkl")

print("\nModels saved successfully!")