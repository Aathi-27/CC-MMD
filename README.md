# CC-MMD Grand Challenge — Cross-Cultural Misogyny Detection Pipeline

> **Task B: Cross-Cultural Prediction**  
> Predict misogyny (1 / 0) for every meme across three culture partitions (Indian, Irish/Western, Chinese) simultaneously.

---

## Table of Contents

1. [Project Scope & Objective](#1-project-scope--objective)
2. [The Problem — Why This Is Hard](#2-the-problem--why-this-is-hard)
3. [Dataset Overview](#3-dataset-overview)
4. [System Architecture](#4-system-architecture)
5. [Pipeline — Step by Step](#5-pipeline--step-by-step)
6. [Technical Challenges & Solutions](#6-technical-challenges--solutions)
7. [Results](#7-results)
8. [File & Folder Structure](#8-file--folder-structure)
9. [How to Run](#9-how-to-run)
10. [Submission Format](#10-submission-format)

---

## 1. Project Scope & Objective

### What is the CC-MMD Challenge?
The **Cross-Cultural Meme Misogyny Detection (CC-MMD)** is an academic Grand Challenge focused on identifying misogynistic content in internet memes across multiple languages and cultural contexts.

### Our Task — Task B
Given a meme (image + transcribed text), predict:
- `original_culture` — Is it misogynistic for the **original/Indian** audience?
- `irish_culture` — Is it misogynistic for the **Western/Irish** audience?
- `chinese_culture` — Is it misogynistic for the **Chinese** audience?

The same meme can be perceived differently by people from different cultural backgrounds. A joke that is offensive in one culture may be neutral in another. The competition's official ranking metric is the **culture-balanced Macro F1** averaged across all three culture partitions.

### Submission Format
```
image_id,original_culture,irish_culture,chinese_culture
654,1,0,0
1339,1,1,1
288,0,0,0
```

---

## 2. The Problem — Why This Is Hard

### 2.1 Multimodal Input
Memes combine **two modalities**: an image (often with embedded visual humor or cultural references) and **text** (transcribed OCR of the meme text, often in Tamil, Malayalam, Chinese, or English). Neither modality alone is sufficient.

### 2.2 Cross-Cultural Subjectivity
Misogyny is **culturally subjective**. What one culture labels misogynistic, another may not. The annotations in the dataset reflect this — the same meme carries *three different binary labels*, one per cultural perspective.

### 2.3 Class Imbalance
The datasets are imbalanced — not all cultures have equal numbers of misogynistic vs. non-misogynistic samples. Models trained naively will predict the majority class.

### 2.4 Missing Labels
Western memes don't have a "western_label" for themselves (they provide Indian and Chinese cross-cultural views). So training must handle `NaN` labels gracefully.

### 2.5 CPU-Only Hardware Constraint
This project was built and run entirely on **CPU**. Embedding extraction with CLIP and XLM-R models (700MB+ each) on CPU is slow — we had to engineer around this with embedding caching.

---

## 3. Dataset Overview

| Culture | Source | Train | Dev | Test |
|---------|--------|-------|-----|------|
| Tamil | Tamil meme dataset | ~1200 | yes | 356 |
| Malayalam | Malayalam meme dataset | ~800 | yes | 200 |
| Chinese | Chinese meme dataset | ~1000 | yes | 340 |
| Western | MAMI dataset (English) | ~3000 | yes | 1000 |
| **Total** | | **~6000** | | **1896** |

Each row: `image_id`, `transcription`, and **three binary labels** (india, western, china).

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT (per meme)                      │
│         Image file  +  Transcription text                │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
   CLIP ViT-B/32      XLM-RoBERTa-base
   (frozen encoder)   (frozen encoder)
        │                  │
   512-d image emb    768-d text emb
        │                  │
        └────────┬─────────┘
                 │ concat
           1280-d vector
                 │
    ┌────────────▼────────────┐
    │   Shared MLP Trunk      │
    │   1280 → 512 → 256      │
    │   LayerNorm + Dropout   │
    └────────────┬────────────┘
                 │ 256-d latent
       ┌─────────┼─────────┐
       │         │         │
   India     Western    China
   head      head       head
   (linear)  (linear)   (linear)
       │         │         │
       └─────────┼─────────┘
                 │
    ┌────────────▼─────────────┐
    │  Cultural Prototype Gate  │
    │  (per culture):           │
    │  - Cosine sim to pos/neg  │
    │    K-Means prototypes     │
    │  - Learned gate weight g  │
    │  - Final = g*cultural +   │
    │    (1-g)*base             │
    └────────────┬─────────────┘
                 │
    ┌────────────▼─────────────┐
    │  Calibrated Thresholds   │
    │  india=0.59, w=0.63,     │
    │  china=0.47              │
    └────────────┬─────────────┘
                 │
    3 binary predictions → submission.csv
```

---

## 5. Pipeline — Step by Step

### Stage 1 — Data Normalization (`src/data_merge.py`)
Each culture's CSV had different column names. We normalized all 4 cultures to a unified schema:

| Culture | Raw label columns | Action |
|---------|------------------|--------|
| Tamil | `india_label`, `western_label`, `china_label` (int) | Keep as-is |
| Malayalam | same as Tamil | Keep as-is |
| Chinese | `original_labels`, `indian_labels`, `irish_labels` (strings) | Map `misogyny→1`, `not-misogyny→0` |
| Western | `indian_labels`, `chinese_labels` (strings); no `western_label` | Map strings; `western_label = NaN` |

Output: `train_merged.csv` — 6000+ rows with unified columns.

---

### Stage 2 — Embedding Extraction (`src/embedding_extractor.py`)
Run once, results cached as `.npy` files.

```
CLIP ViT-B/32  →  clip_image.npy       (N, 512)
XLM-RoBERTa   →  xlmr_text.npy        (N, 768)
```

- Images: loaded by path, converted to RGB, L2-normalized
- Text: tokenized, CLS token extracted, L2-normalized
- Missing images: replaced with grey 224×224 fallback
- **Why cache?** Single extraction run was 45–90 mins on CPU. Cached as `.npy` so all downstream training is instant.

---

### Stage 3 — Model Training (`run_full.py`, Stage 5)
Trained a **masked multi-task Binary Cross-Entropy** loss — only computes loss for samples that have a valid label for that culture (skips NaN entries).

- Class weights computed per culture to handle imbalance (`neg_count / pos_count`)
- Optimizer: AdamW, LR=1e-3, Cosine Annealing schedule
- Early stopping: patience=8 epochs
- Best model saved to `results/checkpoints/best_model.pt`

---

### Stage 4 — Cultural Prototypes (`run_full.py`, Stage 6)
After training the base model:
1. Extract 256-d latent vectors for all training samples
2. Per culture, per class, run **K-Means (k=6)** on the latent space
3. Save 6 misogyny centroids + 6 not-misogyny centroids per culture

These prototypes encode what "culturally misogynistic" samples look like in the model's internal representation.

---

### Stage 5 — Gated Cultural Layer Fine-Tuning (`run_full.py`, Stage 7)
Freeze the base model. Train only the **cultural gate** weights (small: ~15K parameters):

```
For each culture c:
  sim_pos = max cosine similarity to misogyny prototypes[c]
  sim_neg = max cosine similarity to not-misogyny prototypes[c]
  gate = sigmoid(W * [latent, sim_pos, sim_neg])
  output = gate * cultural_head(latent) + (1-gate) * base_logit
```

The gate learns when to trust the prototype signal vs. the base model. Best cultural model saved to `results/checkpoints/best_cultural.pt` (avg F1 = **0.784**).

---

### Stage 6 — Threshold Calibration (`src/calibration.py`, run in Stage 8)
Fixed threshold (0.5) is rarely optimal for imbalanced datasets. We sweep thresholds from 0.30 to 0.70 on the **dev set** to maximize Macro F1 independently per culture.

| Culture | Calibrated threshold |
|---------|---------------------|
| India | 0.59 |
| Western | 0.63 |
| China | 0.47 |

Expected gain: **+3 to +6 F1 points** vs. using 0.5.

---

### Stage 7 — Test Inference (`run_test_inference.py`)
The dedicated test pipeline, built to be fully independent of train/dev:

1. Read test CSVs from `test/{culture}/test.csv`
2. Normalize (same column mapping as train)
3. Merge all 4 cultures → `test_final.csv` (1896 rows)
4. Extract CLIP + XLM-R embeddings (cached in `data/embeddings/test/`)
5. Load `best_cultural.pt` + prototypes
6. Run inference with calibrated thresholds
7. Format + validate → `results/submissions/submission.csv`

---

## 6. Technical Challenges & Solutions

### Challenge 1: Inconsistent raw data schemas
*Different column names across 4 culture datasets.*
**Solution:** Culture-specific normalizers in `data_merge.py` that handle each format separately, then concat to a unified dataframe.

### Challenge 2: CPU-only embedding extraction was too slow
*CLIP + XLM-R on CPU: ~45–90 minutes per run.*
**Solution:** Extract once, save as `.npy` cache. All training and inference reads from cache. The pipeline checks if cache exists before extracting.

### Challenge 3: Missing / NaN labels (Western dataset has no `western_label`)
*Standard BCE loss crashes on NaN.*
**Solution:** Implemented a `masked_bce` loss function that tracks a per-sample, per-culture mask and only computes loss where labels exist.

### Challenge 4: Class imbalance
*Some cultures had 3:1 or 4:1 negative-to-positive ratios.*
**Solution:** Computed `pos_weight = neg_count / pos_count` per culture and applied it inside BCE loss to upweight misogyny samples.

### Challenge 5: Model checkpoint filename mismatch
*`inference.py` looked for `best_model_cultural.pt` but the actual file was `best_cultural.pt`.*
**Solution:** Fixed the filename in `src/inference.py` and added a fallback search for both names.

### Challenge 6: Threshold file naming mismatch
*`run_full.py` saved `thresholds.json` but `inference.py` read `calibrated_thresholds.json`.*
**Solution:** Updated `load_thresholds()` to check both filenames, returning the first found.

### Challenge 7: Windows encoding errors (`cp1252`)
*Unicode characters (✓, ✗) in print statements caused crashes on Windows PowerShell.*
**Solution:** Replaced all unicode symbols with ASCII equivalents (`[OK]`, `[ERR]`) and ran critical scripts with `-X utf8` flag.

### Challenge 8: Test data had different column names than train
*Test CSVs used `transcriptions` (not `transcription`) and had label columns even though test is unlabeled.*
**Solution:** `run_test_inference.py` handles each culture's test schema independently (Tamil/Malayalam: `transcriptions`, `original_labels`, `irish_labels`, `chinese_labels`; Chinese: adds `indian_labels`; Western: `indian_labels`, `chinese_labels` only).

### Challenge 9: Same image_id across different culture datasets
*Tamil and Malaysian share image ID ranges (1–1777), so duplicate IDs in the merged file are expected.*
**Solution:** Validation code flags these as "expected duplicates" rather than errors.

---

## 7. Results

| Metric | Value |
|--------|-------|
| Best model | `best_cultural.pt` |
| Training avg Macro F1 (dev) | **0.784** |
| Calibrated thresholds | india=0.59, western=0.63, china=0.47 |
| Test samples | 1896 |
| Predicted misogyny (india) | 650 / 1896 (34%) |
| Predicted misogyny (western) | 850 / 1896 (45%) |
| Predicted misogyny (china) | 954 / 1896 (50%) |

---

## 8. File & Folder Structure

```
E:/pep/
├── data/
│   ├── image/                        # Training images (per culture)
│   │   ├── tamil/
│   │   ├── malayalam/
│   │   ├── chinese/
│   │   └── western/
│   ├── embeddings/                   # Train embeddings (cached)
│   │   ├── clip_image.npy            # (N_train, 512)
│   │   ├── xlmr_text.npy             # (N_train, 768)
│   │   └── test/                     # Test embeddings (cached)
│   │       ├── image_emb_test.npy    # (1896, 512)
│   │       └── text_emb_test.npy     # (1896, 768)
│   └── embeddings_dev/               # Dev embeddings (cached)
│
├── train/                            # Raw train CSVs (per culture)
│   ├── tamil/train_clean.csv
│   ├── malayalam/train_clean.csv
│   ├── chinese/train_clean.csv
│   └── western/train_clean.csv
│
├── test/                             # Raw test CSVs + images
│   ├── tamil/   test.csv + *.jpg
│   ├── malayalam/ test.csv + *.jpg
│   ├── chinese/  test.csv + *.jpg
│   └── western/  test.csv + *.jpg
│
├── dev/
│   └── dev_final.csv                 # Normalized dev set (with labels)
│
├── src/
│   ├── config.py                     # All paths + hyperparameters
│   ├── data_merge.py                 # Stage 1: normalize + merge CSVs
│   ├── embedding_extractor.py        # Stage 2: CLIP + XLM-R extraction
│   ├── dataset.py                    # PyTorch Dataset wrapper
│   ├── model.py                      # MLP + CulturalPrototypeLayer + FullModel
│   ├── trainer.py                    # Training loop with masked BCE
│   ├── prototypes.py                 # K-Means prototype construction
│   ├── gate_finetune.py              # Gated cultural layer fine-tuning
│   ├── calibration.py                # Per-culture threshold calibration
│   └── inference.py                  # Generic inference functions
│
├── results/
│   ├── checkpoints/
│   │   ├── best_model.pt             # Base MLP checkpoint
│   │   └── best_cultural.pt          # Gated cultural model (best)
│   ├── prototypes/                   # K-Means centroids per culture
│   │   ├── india_pos.npy / india_neg.npy
│   │   ├── western_pos.npy / western_neg.npy
│   │   └── china_pos.npy / china_neg.npy
│   ├── logs/
│   │   ├── thresholds.json           # Calibrated per-culture thresholds
│   │   └── history.json              # Training epoch history
│   └── submissions/
│       └── submission.csv            # FINAL SUBMISSION FILE (1896 rows)
│
├── train_merged.csv                  # Merged + normalized training set
├── test_final.csv                    # Merged + normalized test set
│
├── run_full.py                       # End-to-end train pipeline (Stages 3–10)
├── run_test_inference.py             # Test inference pipeline (generates submission.csv)
├── train.py                          # Shortcut training script
├── extract_embeddings.py             # Standalone embedding extractor
└── requirements.txt                  # Python dependencies
```

---

## 9. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Normalize and merge training data
```bash
python -m src.data_merge
```

### Step 2: Extract embeddings (run once, takes ~60–90 min on CPU)
```bash
python extract_embeddings.py
```

### Step 3: Train the full pipeline (Stages 3–10)
```bash
python run_full.py
```
This runs: embedding extraction → train MLP → build prototypes → fine-tune gates → calibrate thresholds → generate dev submission.

### Step 4: Generate test submission
```bash
python -X utf8 run_test_inference.py
```
Output: `results/submissions/submission.csv` — ready to upload.

---

## 10. Submission Format

The final file follows the Task B specification exactly:

```csv
image_id,original_culture,irish_culture,chinese_culture
1006,0,0,0
688,0,0,0
1771,1,0,1
```

- `original_culture` → India prediction (1=misogyny, 0=not)
- `irish_culture` → Western/Irish prediction
- `chinese_culture` → China prediction

Validated checks that pass:
- All required columns present
- Only 0 and 1 values in prediction columns
- All 1896 test image_ids covered
- No missing rows

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Frozen encoders (CLIP + XLM-R) | CPU feasibility; pre-trained weights already encode rich cross-lingual and visual features |
| Shared trunk + separate heads | Learns culture-invariant features while allowing per-culture decision boundaries |
| Masked BCE loss | Only valid labels contribute to training; prevents NaN gradient explosions |
| K-Means prototypes (k=6) | Captures intra-class diversity; "average misogynistic meme" not meaningful for diverse cultural expressions |
| Per-culture threshold calibration | Free F1 gain; optimal boundary differs per culture due to class imbalance |
| Embedding caching | Decouples slow extraction (hours) from fast training iteration (minutes) |
