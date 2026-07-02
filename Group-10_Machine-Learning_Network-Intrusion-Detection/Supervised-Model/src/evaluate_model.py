import pandas as pd
import joblib
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# =========================
# LOAD TEST DATA
# =========================
test_df = pd.read_csv("../Dataset/Combined_Dataset/CICIDS2017_TEST.csv")

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.dropna(inplace=True)

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

# =========================
# LOAD MODELS
# =========================
rf = joblib.load("../Model/rf_model.pkl")
xgb = joblib.load("../Model/xgb_model.pkl")
le = joblib.load("../Model/label_encoder.pkl")

# encode labels
y_test_enc = le.transform(y_test)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, name):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test_enc, preds)
    f1 = f1_score(y_test_enc, preds, average="macro")

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("F1 Score (macro):", f1)

    print("\nClassification Report:")
    print(classification_report(y_test_enc, preds))

    return preds


# =========================
# RUN EVALUATION
# =========================
rf_preds = evaluate(rf, "Random Forest")
xgb_preds = evaluate(xgb, "XGBoost")


# =========================
# CONFUSION MATRIX PLOT
# =========================
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


plot_cm(y_test_enc, rf_preds, "Random Forest Confusion Matrix")
plot_cm(y_test_enc, xgb_preds, "XGBoost Confusion Matrix")


# =========================
# F1 SCORE COMPARISON PLOT
# =========================
models = ["Random Forest", "XGBoost"]
f1_scores = [
    f1_score(y_test_enc, rf_preds, average="macro"),
    f1_score(y_test_enc, xgb_preds, average="macro")
]

plt.figure()
plt.bar(models, f1_scores)
plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score")
plt.show()