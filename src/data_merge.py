"""
Stage 1: Fix the merged dataset.

Problem: The 4 per-culture CSVs have inconsistent column names.
- Tamil/Malayalam: india_label, western_label, china_label (already 0/1)
- Western (MAMI): indian_labels, chinese_labels, western_label (strings, western=NaN)
- Chinese: original_labels, indian_labels, irish_labels (strings)

This script:
1. Loads each per-culture CSV
2. Normalizes columns to: image_id, transcription, india_label, western_label, china_label, image_path, source_culture
3. Converts string labels to 0/1
4. Fixes image paths (data/image not data/images)
5. Validates image existence
6. Merges and saves train_merged.csv
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TRAIN_CSVS, MERGED_CSV, IMAGE_DIRS

LABEL_MAP = {"misogyny": 1, "not-misogyny": 0}
TARGET_COLS = ["image_id", "transcription", "india_label", "western_label", "china_label", "image_path", "source_culture"]


def normalize_tamil(df):
    """Tamil: already has india_label/western_label/china_label as 0/1."""
    df = df.copy()
    df["source_culture"] = "tamil"
    # Fix image path: ensure it points to data/image/tamil/{id}.jpg
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(IMAGE_DIRS["tamil"], f"{x}.jpg")
    )
    return df[TARGET_COLS]


def normalize_malayalam(df):
    """Malayalam: same schema as Tamil."""
    df = df.copy()
    df["source_culture"] = "malayalam"
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(IMAGE_DIRS["malayalam"], f"{x}.jpg")
    )
    return df[TARGET_COLS]


def normalize_chinese(df):
    """
    Chinese has: original_labels (=china), indian_labels (=india), irish_labels (=western)
    All are strings: 'misogyny' / 'not-misogyny'
    """
    df = df.copy()
    df["china_label"] = df["original_labels"].map(LABEL_MAP)
    df["india_label"] = df["indian_labels"].map(LABEL_MAP)
    df["western_label"] = df["irish_labels"].map(LABEL_MAP)
    df["source_culture"] = "chinese"
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(IMAGE_DIRS["chinese"], f"{x}.jpg")
    )
    return df[TARGET_COLS]


def normalize_western(df):
    """
    Western (MAMI) has: indian_labels (=india), chinese_labels (=china), western_label (=NaN always)
    Labels are strings.
    """
    df = df.copy()
    df["india_label"] = df["indian_labels"].map(LABEL_MAP)
    df["china_label"] = df["chinese_labels"].map(LABEL_MAP)
    # western_label is already NaN in this dataset — keep it NaN
    df["western_label"] = np.nan
    df["source_culture"] = "western"
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(IMAGE_DIRS["western"], f"{x}.jpg")
    )
    return df[TARGET_COLS]


def validate_images(df):
    """Check how many image files actually exist."""
    missing = []
    for _, row in df.iterrows():
        if not os.path.exists(row["image_path"]):
            missing.append(row["image_path"])
    return missing


def main():
    print("=" * 60)
    print("STAGE 1: Normalize and merge datasets")
    print("=" * 60)

    normalizers = {
        "tamil": normalize_tamil,
        "malayalam": normalize_malayalam,
        "chinese": normalize_chinese,
        "western": normalize_western,
    }

    all_dfs = []
    for culture, csv_path in TRAIN_CSVS.items():
        print(f"\n--- {culture.upper()} ---")
        df = pd.read_csv(csv_path)
        print(f"  Raw shape: {df.shape}, columns: {list(df.columns)}")

        normalized = normalizers[culture](df)
        print(f"  Normalized shape: {normalized.shape}")

        # Label stats
        for col in ["india_label", "western_label", "china_label"]:
            valid = normalized[col].notna().sum()
            pos = (normalized[col] == 1).sum()
            neg = (normalized[col] == 0).sum()
            print(f"  {col}: valid={valid} (1={pos}, 0={neg}), NaN={normalized[col].isna().sum()}")

        # Image validation (sample)
        sample = normalized.sample(min(20, len(normalized)), random_state=42)
        missing = validate_images(sample)
        if missing:
            print(f"  WARNING: {len(missing)}/20 sample images missing!")
            print(f"    Example: {missing[0]}")
        else:
            print(f"  Image check: 20/20 found OK")

        all_dfs.append(normalized)

    # Merge
    merged = pd.concat(all_dfs, ignore_index=True)

    # Transcription quality
    empty_trans = merged["transcription"].isna() | (merged["transcription"].astype(str).str.strip() == "")
    print(f"\n{'=' * 60}")
    print(f"MERGED DATASET")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(merged)}")
    print(f"Empty transcriptions: {empty_trans.sum()}")
    print(f"Source distribution:\n{merged['source_culture'].value_counts().to_string()}")

    print(f"\nLabel availability:")
    for col in ["india_label", "western_label", "china_label"]:
        valid = merged[col].notna().sum()
        pos = (merged[col] == 1).sum()
        neg = (merged[col] == 0).sum()
        pct = valid / len(merged) * 100
        print(f"  {col}: {valid}/{len(merged)} ({pct:.1f}%) — 1={pos}, 0={neg}")

    # Save
    merged.to_csv(MERGED_CSV, index=False)
    print(f"\nSaved: {MERGED_CSV}")
    print(f"Columns: {list(merged.columns)}")

    return merged


if __name__ == "__main__":
    main()
