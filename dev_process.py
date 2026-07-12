import pandas as pd
import os

BASE_PATH = "E:/pep/data/dev"
OUTPUT_PATH = "E:/pep/dev/dev_final.csv"

def process_tamil():
    path = f"{BASE_PATH}/tamil/dev.csv"
    img_folder = f"{BASE_PATH}/tamil"

    df = pd.read_csv(path)
    # cols: image_id, transcriptions, original_labels, irish_labels, chinese_labels
    df = df.rename(columns={
        "transcriptions": "transcription",
        "original_labels": "india_label",
        "irish_labels": "western_label",
        "chinese_labels": "china_label"
    })

    return finalize(df, img_folder)


def process_malayalam():
    path = f"{BASE_PATH}/malayalam/dev.csv"
    img_folder = f"{BASE_PATH}/malayalam"

    df = pd.read_csv(path)
    # cols: image_id, transcriptions, original_labels, irish_labels, chinese_labels
    df = df.rename(columns={
        "transcriptions": "transcription",
        "original_labels": "india_label",
        "irish_labels": "western_label",
        "chinese_labels": "china_label"
    })

    return finalize(df, img_folder)


def process_chinese():
    path = f"{BASE_PATH}/chinese/dev.csv"
    img_folder = f"{BASE_PATH}/chinese"

    df = pd.read_csv(path)
    # cols: image_id, transcriptions, original_labels, indian_labels, irish_labels
    df = df.rename(columns={
        "transcriptions": "transcription",
        "original_labels": "china_label",
        "indian_labels": "india_label",
        "irish_labels": "western_label"
    })

    return finalize(df, img_folder)


def process_western():
    path = f"{BASE_PATH}/western/dev.csv"
    img_folder = f"{BASE_PATH}/western"

    df = pd.read_csv(path)
    # cols: image_id, transcriptions, indian_labels, chinese_labels
    df = df.rename(columns={
        "transcriptions": "transcription",
        "indian_labels": "india_label",
        "chinese_labels": "china_label"
    })

    df["western_label"] = None

    return finalize(df, img_folder)


def finalize(df, img_folder):
    label_map = {
        "misogyny": 1,
        "not-misogyny": 0
    }

    for col in ["india_label", "western_label", "china_label"]:
        if col in df.columns:
            df[col] = df[col].map(label_map)

    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(img_folder, f"{x}.jpg")
    )

    return df[[
        "image_id",
        "transcription",
        "india_label",
        "western_label",
        "china_label",
        "image_path"
    ]]


# ===== RUN =====
dfs = [
    process_tamil(),
    process_malayalam(),
    process_chinese(),
    process_western()
]

dev_final = pd.concat(dfs, ignore_index=True)

# ===== VALIDATION =====
print("Total rows:", len(dev_final))
print("Missing values:")
print(dev_final.isnull().sum())
print()
for col in ["india_label", "western_label", "china_label"]:
    v = dev_final[col].notna().sum()
    p = (dev_final[col] == 1).sum()
    n = (dev_final[col] == 0).sum()
    print(f"  {col}: valid={v}, 1={p}, 0={n}, NaN={dev_final[col].isna().sum()}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
dev_final.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
