import pandas as pd

df = pd.read_csv("../Dataset/Combined_Dataset/CICIDS2017_FULL.csv")

df.columns = df.columns.str.strip()

def map_labels(label):
    label = label.strip()

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
        "Web Attack � Brute Force",
        "Web Attack � XSS",
        "Web Attack � Sql Injection"
    ]:
        return "WEB_ATTACK"

    elif label in ["FTP-Patator", "SSH-Patator"]:
        return "BRUTE_FORCE"

    else:
        return "OTHER"

df["Label"] = df["Label"].apply(map_labels)

print(df["Label"].value_counts())

df.to_csv("CICIDS2017_GROUPED.csv", index=False)