import pandas as pd
import os
import numpy as np

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
        print(f"Skipping {f}")
        continue

    dfs.append(df)

# =========================
# ALIGN COLUMNS
# =========================
common_columns = set(dfs[0].columns)

for df in dfs[1:]:
    common_columns = common_columns.intersection(set(df.columns))

common_columns = list(common_columns)

dfs = [df[common_columns] for df in dfs]

# =========================
# MERGE
# =========================
full_df = pd.concat(dfs, ignore_index=True)

print("\nRAW SHAPE:", full_df.shape)

# =========================
# CLEAN NUMERIC ISSUES (IMPORTANT FIX)
# =========================
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)
full_df.drop_duplicates(inplace=True)

print("\nAFTER CLEANING SHAPE:", full_df.shape)

# =========================
# LABEL CLEANING
# =========================
full_df["Label"] = full_df["Label"].astype(str)
full_df["Label"] = full_df["Label"].str.strip()

# =========================
# LABEL GROUPING
# =========================
def map_labels(label):

    label = str(label).strip()

    if label == "BENIGN":
        return "BENIGN"

    elif label in [
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
        "DDoS"
    ]:
        return "DOS_ATTACK"

    elif label == "PortScan":
        return "RECON_ATTACK"

    elif label in ["Bot", "Infiltration"]:
        return "ADVANCED_ATTACK"

    elif label in [
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
        "Web Attack � Brute Force",
        "Web Attack � XSS",
        "Web Attack � Sql Injection"
    ]:
        return "WEB_ATTACK"

    elif label in ["FTP-Patator", "SSH-Patator"]:
        return "BRUTE_FORCE"

    else:
        return "OTHER"


full_df["Label"] = full_df["Label"].apply(map_labels)

print("\nGROUPED LABELS:")
print(full_df["Label"].value_counts())

# =========================
# REMOVE NOISE CLASS
# =========================
full_df = full_df[full_df["Label"] != "OTHER"]

print("\nAFTER REMOVING OTHER:")
print(full_df["Label"].value_counts())

# =========================
# SAVE FINAL CLEAN DATASET
# =========================
output_path = "../Dataset/Combined_Dataset/CICIDS2017_GROUPED.csv"
full_df.to_csv(output_path, index=False)

print("\nSAVED:", output_path)
print("FINAL SHAPE:", full_df.shape)