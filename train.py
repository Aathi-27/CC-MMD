import pandas as pd
import os

# # ===== CONFIG =====
# INPUT_CSV = "E:/pep/data/image/western/train.csv"
# OUTPUT_CSV = "E:/pep/train/western/train_clean.csv"
# IMAGE_FOLDER = "E:/pep/data/image/western"   # change per dataset

# # ===== LOAD =====
# df = pd.read_csv(INPUT_CSV)

# # ===== RENAME COLUMNS =====
# # df = df.rename(columns={
# #     "transcriptions": "transcription",
# #     "original_labels": "india_label",
# #     "irish_labels": "western_label",
# #     "chinese_labels": "china_label"
# # })

# # df = df.rename(columns={
# #     "transcriptions": "transcription",
# #     "original_culture": "china_label",
# #     "target_culture1": "india_label",
# #     "target_culture2": "western_label"
# # })

# df = df.rename(columns={
#     "transcriptions": "transcription",
#     "target_culture1": "india_label",
#     "target_culture2": "china_label"
# })

# df["western_label"] = None

# # ===== LABEL MAPPING =====
# label_map = {
#     "misogyny": 1,
#     "not-misogyny": 0
# }

# for col in ["india_label", "western_label", "china_label"]:
#     if col in df.columns:
#         df[col] = df[col].map(label_map)

# # ===== ADD IMAGE PATH =====
# def build_path(img_id):
#     return os.path.join(IMAGE_FOLDER, str(img_id))

# df["image_path"] = df["image_id"].apply(build_path)

# # ===== VALIDATION (IMPORTANT) =====
# missing_images = df[~df["image_path"].apply(os.path.exists)]

# print(f"Missing images: {len(missing_images)}")

# # ===== SAVE =====
# df.to_csv(OUTPUT_CSV, index=False)

# print("Saved:", OUTPUT_CSV)





dfs = [
    pd.read_csv("E:/pep/train/tamil/train_clean.csv"),
    pd.read_csv("E:/pep/train/malayalam/train_clean.csv"),
    pd.read_csv("E:/pep/train/chinese/train_clean.csv"),
    pd.read_csv("E:/pep/train/western/train_clean.csv"),
]

final = pd.concat(dfs, ignore_index=True)
final.to_csv("train_final.csv", index=False)