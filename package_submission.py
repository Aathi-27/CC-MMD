"""
CC-MMD Grand Challenge -- Submission Packager
==============================================
Splits combined submission.csv by source_culture and writes per-language
Task B CSVs with the CORRECT column names per language, then zips them.

Column semantics per source language
-------------------------------------
Tamil / Malayalam  (Indian-origin memes):
    original_culture  <- india     (model head index 0)
    irish_culture     <- western   (model head index 1)
    chinese_culture   <- china     (model head index 2)
    File: innovix_taskb_tamil.csv / innovix_taskb_malayalam.csv

Chinese (Chinese-origin memes):
    original_culture  <- china     (model head index 2)
    indian_culture    <- india     (model head index 0)
    irish_culture     <- western   (model head index 1)
    File: innovix_taskb_chinese.csv

Western / English (Western-origin memes):
    original_culture  <- western   (model head index 1)
    indian_culture    <- india     (model head index 0)
    chinese_culture   <- china     (model head index 2)
    File: innovix_taskb_english.csv

Required zip layout:
    innovix.zip/
        tamil/innovix_taskb_tamil.csv
        malayalam/innovix_taskb_malayalam.csv
        chinese/innovix_taskb_chinese.csv
        english/innovix_taskb_english.csv
"""
import os
import zipfile
import pandas as pd

# ─────────── paths ───────────
ROOT        = "E:/pep"
SUBMIT_DIR  = os.path.join(ROOT, "results", "submissions")
SUBMISSION  = os.path.join(SUBMIT_DIR, "submission.csv")
TEST_FINAL  = os.path.join(ROOT, "test_final.csv")
TASKB_DIR   = os.path.join(SUBMIT_DIR, "taskb")
ZIP_OUT     = os.path.join(SUBMIT_DIR, "innovix.zip")

TEAM_NAME   = "innovix"

# ─────────────────────────────────────────────────
# Column spec per source language.
# Each entry maps: source_culture_tag ->
#   (subfolder, file_suffix, [(out_col_name, model_col), ...])
#
# model_col is one of: original_culture / irish_culture / chinese_culture
# which are the column names in the combined submission.csv produced by
# run_test_inference.py (india=original_culture, western=irish_culture,
#                         china=chinese_culture).
# ─────────────────────────────────────────────────
LANG_SPEC = {
    "tamil": (
        "tamil",
        "tamil",
        [
            ("original_culture", "original_culture"),   # india head
            ("irish_culture",    "irish_culture"),       # western head
            ("chinese_culture",  "chinese_culture"),     # china head
        ],
    ),
    "malayalam": (
        "malayalam",
        "malayalam",
        [
            ("original_culture", "original_culture"),   # india head
            ("irish_culture",    "irish_culture"),       # western head
            ("chinese_culture",  "chinese_culture"),     # china head
        ],
    ),
    "chinese": (
        "chinese",
        "chinese",
        [
            ("original_culture", "chinese_culture"),    # china head -> original
            ("indian_culture",   "original_culture"),   # india head -> indian
            ("irish_culture",    "irish_culture"),       # western head -> irish
        ],
    ),
    "western": (
        "english",
        "english",
        [
            ("original_culture", "irish_culture"),      # western head -> original
            ("indian_culture",   "original_culture"),   # india head -> indian
            ("chinese_culture",  "chinese_culture"),    # china head -> chinese
        ],
    ),
}

# ─────────── load ───────────
print("Loading submission.csv ...")
sub = pd.read_csv(SUBMISSION)
print(f"  Rows : {len(sub)}")
print(f"  Cols : {list(sub.columns)}")

print("\nLoading test_final.csv ...")
test_df = pd.read_csv(TEST_FINAL)
print(f"  Rows : {len(test_df)}")
print(f"  Culture distribution:")
print(test_df["source_culture"].value_counts().to_string())

if len(sub) != len(test_df):
    raise ValueError(
        f"Row count mismatch: submission={len(sub)}, test_final={len(test_df)}"
    )

# Positional merge (same row order from run_test_inference.py)
merged = sub.copy()
merged["source_culture"] = test_df["source_culture"].values

print("\nSource culture distribution in submission:")
print(merged["source_culture"].value_counts().to_string())

# ─────────── split, rename columns, write ───────────
print("\n" + "=" * 60)
print("Writing per-language Task B CSVs ...")
print("=" * 60)

written_files = []

for source_cult, (subfolder, lang_suffix, col_map) in LANG_SPEC.items():
    subset = merged[merged["source_culture"] == source_cult].copy()

    # Build output dataframe with correct column names & order
    out = pd.DataFrame()
    out["image_id"] = subset["image_id"].values
    for out_col, src_col in col_map:
        out[out_col] = subset[src_col].values

    out_dir  = os.path.join(TASKB_DIR, subfolder)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{TEAM_NAME}_taskb_{lang_suffix}.csv"
    out_path = os.path.join(out_dir, filename)
    out.to_csv(out_path, index=False)
    written_files.append((subfolder, filename, out_path))

    print(f"\n  [{source_cult}] -> {subfolder}/{filename}")
    print(f"    Rows   : {len(out)}")
    print(f"    Columns: {list(out.columns)}")
    for col in out.columns[1:]:    # skip image_id
        d = out[col].value_counts()
        print(f"    {col:22s}: 1={d.get(1, 0):4d}  0={d.get(0, 0):4d}")

# ─────────── validate ───────────
print("\n" + "=" * 60)
print("Validating ...")
print("=" * 60)

EXPECTED_COLS = {
    "tamil":    ["image_id", "original_culture", "irish_culture",   "chinese_culture"],
    "malayalam":["image_id", "original_culture", "irish_culture",   "chinese_culture"],
    "chinese":  ["image_id", "original_culture", "indian_culture",  "irish_culture"],
    "english":  ["image_id", "original_culture", "indian_culture",  "chinese_culture"],
}

ALL_SOURCES = {
    "tamil":    "tamil",
    "malayalam":"malayalam",
    "chinese":  "chinese",
    "english":  "western",
}

all_ok = True

for subfolder, filename, path in written_files:
    df = pd.read_csv(path)
    lang = subfolder
    expected = EXPECTED_COLS[lang]
    ok = True

    # Column names
    if list(df.columns) != expected:
        print(f"  [ERR] {filename}: columns {list(df.columns)} != expected {expected}")
        ok = False
    else:
        print(f"  [OK ] {filename}: columns correct {list(df.columns)}")

    # Values binary
    for col in df.columns[1:]:
        bad = set(df[col].unique()) - {0, 1}
        if bad:
            print(f"       [ERR] {col} has non-binary values: {bad}")
            ok = False

    # Row count vs test set
    src = ALL_SOURCES[subfolder]
    expected_rows = len(test_df[test_df["source_culture"] == src])
    if len(df) != expected_rows:
        print(f"       [ERR] row count {len(df)} != expected {expected_rows}")
        ok = False
    else:
        print(f"       Rows : {len(df)} (matches test set)")

    if not ok:
        all_ok = False

# ─────────── zip ───────────
print("\n" + "=" * 60)
print(f"Building {os.path.basename(ZIP_OUT)} ...")
print("=" * 60)

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for subfolder, filename, path in written_files:
        arcname = os.path.join(subfolder, filename)
        zf.write(path, arcname)
        print(f"  Added: {arcname}")

zip_size = os.path.getsize(ZIP_OUT) / 1024
print(f"\n[OK] {ZIP_OUT}  ({zip_size:.1f} KB)")

print("\nZip contents:")
with zipfile.ZipFile(ZIP_OUT) as zf:
    for info in zf.infolist():
        inner = pd.read_csv(zf.open(info.filename))
        print(f"  {info.filename:50s}  {len(inner)} rows  cols={list(inner.columns)}")

print("\n" + "=" * 60)
if all_ok:
    print("[DONE] SUBMISSION READY -- upload innovix.zip to the Google Form")
else:
    print("[WARN] Validation errors above -- fix before uploading")
print("=" * 60)
