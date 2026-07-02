import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# =========================
# LOAD DATA
# =========================
folder = "../Dataset/"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

dfs = []

for f in files:
    path = os.path.join(folder, f)
    print(f"Loading: {f}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        continue

    dfs.append(df)

# =========================
# ALIGN + MERGE
# =========================
common_columns = set(dfs[0].columns)

for df in dfs[1:]:
    common_columns &= set(df.columns)

common_columns = list(common_columns)
dfs = [df[common_columns] for df in dfs]

full_df = pd.concat(dfs, ignore_index=True)

# =========================
# CLEAN DATA
# =========================
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)
full_df.drop_duplicates(inplace=True)

# =========================
# LABEL CLEANING
# =========================
full_df["Label"] = full_df["Label"].astype(str).str.strip()

def map_labels(label):
    if label == "BENIGN":
        return "BENIGN"
    elif label in ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS"]:
        return "DOS_ATTACK"
    elif label == "PortScan":
        return "RECON_ATTACK"
    elif label in ["Bot", "Infiltration"]:
        return "ADVANCED_ATTACK"
    elif label in ["FTP-Patator", "SSH-Patator"]:
        return "BRUTE_FORCE"
    elif "Web Attack" in label:
        return "WEB_ATTACK"
    else:
        return "OTHER"

full_df["Label"] = full_df["Label"].apply(map_labels)
full_df = full_df[full_df["Label"] != "OTHER"]

print("\nClass distribution (before split):")
print(full_df["Label"].value_counts())

# =========================
# TRAIN / TEST SPLIT
# =========================
train_df, test_df = train_test_split(
    full_df,
    test_size=0.2,
    random_state=42,
    stratify=full_df["Label"]
)

print("\nTrain shape BEFORE SMOTE:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain distribution BEFORE SMOTE:")
print(train_df["Label"].value_counts())

# =========================
# SMOTE (TRAIN ONLY)
# =========================
print("\nApplying SMOTE on TRAIN set only...")

# split features and labels
X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

# encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train_enc)

# rebuild dataframe
train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
train_df["Label"] = le.inverse_transform(y_resampled)

print("\nTrain distribution AFTER SMOTE:")
print(train_df["Label"].value_counts())

print("\nTest distribution (UNCHANGED):")
print(test_df["Label"].value_counts())

# =========================
# SAVE
# =========================
output_folder = "../Dataset/Combined_Dataset"
os.makedirs(output_folder, exist_ok=True)

train_df.to_csv(os.path.join(output_folder, "CICIDS2017_TRAIN.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "CICIDS2017_TEST.csv"), index=False)

print("\nSaved SMOTE-balanced training dataset and clean test dataset")