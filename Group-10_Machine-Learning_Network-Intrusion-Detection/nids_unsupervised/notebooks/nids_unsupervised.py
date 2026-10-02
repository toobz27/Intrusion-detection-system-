# %%
import pandas as pd
import glob

# Point this to wherever you downloaded the dataset
files = glob.glob(r"E:\OneDrive\university work\assignments\sixth semester\ml project work\Group-10_Machine-Learning_Network-Intrusion-Detection\nids_unsupervised\Dataset\*.csv")  # adjust path as needed

dfs = []
for f in files:
    df = pd.read_csv(f, encoding='utf-8', low_memory=False)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = pd.concat(dfs, ignore_index=True)

# Fix column names — strips spaces from all column names
df.columns = df.columns.str.strip()

print(df.shape)
print(df['Label'].value_counts())
print(df.shape)
print(df['Label'].value_counts())

# %%
print(df.head())
print(df.dtypes)
print(df.isnull().sum().sort_values(ascending=False).head(20))
print(df.describe())

# %%
df.columns = df.columns.str.strip()
print(df.columns.tolist())  # check they look right

# %%
import numpy as np

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print(f"Shape after cleaning: {df.shape}")

# %%
df['binary_label'] = (df['Label'] != 'BENIGN').astype(int)
# 0 = Normal, 1 = Attack

print(df['binary_label'].value_counts())
print(f"Attack rate: {df['binary_label'].mean()*100:.1f}%")

# %%
# Take 100k rows, keeping the label balance
df_sampled = df.groupby('binary_label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 50000), random_state=42)
).reset_index(drop=True)

print(df_sampled['binary_label'].value_counts())

# %%
# Drop the label columns — keep only numeric features
X = df_sampled.drop(columns=['Label', 'binary_label'])
y = df_sampled['binary_label']

# Keep only numeric columns (some may be object type)
X = X.select_dtypes(include=[np.number])

print(f"Features shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split for supervised evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]

print("=== Random Forest Results ===")
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Attack']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

# %%
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Attack'],
            yticklabels=['Normal','Attack'])
plt.title('Random Forest — Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('rf_confusion.png')
plt.show()

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth':    [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='f1',
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=['Normal','Attack']))

# %%
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=100,
    contamination=0.3,  # roughly the attack ratio in your data
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train)

# Predict: IsolationForest returns -1 (anomaly) or 1 (normal)
y_pred_iso_raw = iso.predict(X_test)

# Convert to 0/1 to match our labels (1=attack, 0=normal)
y_pred_iso = (y_pred_iso_raw == -1).astype(int)

print("=== Isolation Forest Results ===")
print(classification_report(y_test, y_pred_iso, target_names=['Normal','Attack']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_iso):.4f}")

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)

# Get cluster assignments for test set
cluster_labels = kmeans.predict(X_test)

# Figure out which cluster ID corresponds to "attack"
# Do this by checking which cluster has more actual attacks
from scipy.stats import mode

cluster_to_label = {}
for cluster_id in [0, 1]:
    mask = cluster_labels == cluster_id
    actual = y_test[mask]
    # whichever class is majority in this cluster = that cluster's label
    majority = actual.mode()[0]
    cluster_to_label[cluster_id] = majority

print(f"Cluster mapping: {cluster_to_label}")

y_pred_kmeans = pd.Series(cluster_labels).map(cluster_to_label).values

print("\n=== K-Means Results ===")
print(classification_report(y_test, y_pred_kmeans, target_names=['Normal','Attack']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_kmeans):.4f}")

# %%
import joblib, os

os.makedirs(r"E:\OneDrive\university work\assignments\sixth semester\ml project work\Group-10_Machine-Learning_Network-Intrusion-Detection\Model", exist_ok=True)

model_dir = r"E:\OneDrive\university work\assignments\sixth semester\ml project work\Group-10_Machine-Learning_Network-Intrusion-Detection\Model"

joblib.dump(iso,    f"{model_dir}\\iso_model.pkl")
joblib.dump(kmeans, f"{model_dir}\\kmeans_model.pkl")
joblib.dump(scaler, f"{model_dir}\\scaler.pkl")

print("Saved: iso_model.pkl, kmeans_model.pkl, scaler.pkl")

# %%
import joblib
scaler = joblib.load(r"E:\OneDrive\university work\assignments\sixth semester\ml project work\Group-10_Machine-Learning_Network-Intrusion-Detection\nids_unsupervised\scaler.pkl")
print(scaler.n_features_in_)


