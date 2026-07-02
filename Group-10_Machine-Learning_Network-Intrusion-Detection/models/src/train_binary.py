import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

print("Original Shape:", df.shape)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Replace infinity values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows containing NaN
df.dropna(inplace=True)

print("Cleaned Shape:", df.shape)

# Check label counts
print("\nLabel Counts:")
print(df["Label"].value_counts())

# Convert labels to binary
# BENIGN = 0
# DDoS = 1

df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nEncoded Labels:")
print(df["Label"].value_counts())

# Separate features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

print("\nFeature Shape:", X.shape)
print("Label Shape:", y.shape)





#Random Forest Starting Here#

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Random Forest Ending here#