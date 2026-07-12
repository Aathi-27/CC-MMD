"""
CC-MMD Task B — Test Inference Pipeline
========================================
Steps:
  1. Normalize test CSVs from all 4 cultures (no labels)
  2. Merge into test_final.csv
  3. Extract CLIP + XLM-R embeddings  (cached in data/embeddings/test/)
  4. Load best_cultural.pt (or best_model.pt fallback)
  5. Inference with prototype similarity + gating + calibrated thresholds
  6. Format & validate submission.csv

Output: E:/pep/results/submissions/submission.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ROOT        = "E:/pep"
TEST_DIR    = os.path.join(ROOT, "test")          # E:/pep/test/{culture}/test.csv + images
TEST_EMB    = os.path.join(ROOT, "data", "embeddings", "test")
CHECKPOINT  = os.path.join(ROOT, "results", "checkpoints")
LOG_DIR     = os.path.join(ROOT, "results", "logs")
PROTO_DIR   = os.path.join(ROOT, "results", "prototypes")
SUBMIT_DIR  = os.path.join(ROOT, "results", "submissions")
TEST_CSV    = os.path.join(ROOT, "test_final.csv")

CLIP_MODEL  = "openai/clip-vit-base-patch32"
XLM_MODEL   = "xlm-roberta-base"
CLIP_DIM    = 512
TEXT_DIM    = 768
HIDDEN      = 512
LATENT      = 256
DROPOUT     = 0.3
CULTURES    = ["india", "western", "china"]

for d in [TEST_EMB, SUBMIT_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — NORMALIZE TEST CSVs  (no labels)
# ─────────────────────────────────────────────
def normalize_test_csvs():
    """
    Read each culture's test.csv, keep only image_id + transcription,
    add image_path pointing to E:/pep/test/{culture}/{image_id}.jpg
    """
    print("=" * 60)
    print("STEP 1 — Normalize test CSVs")
    print("=" * 60)

    all_dfs = []

    # ── Tamil ──────────────────────────────────────
    # Columns: image_id, transcriptions, original_labels, irish_labels, chinese_labels
    print("\n--- TAMIL ---")
    df = pd.read_csv(os.path.join(TEST_DIR, "tamil", "test.csv"))
    print(f"  Raw cols: {list(df.columns)}  rows: {len(df)}")
    df = df[["image_id", "transcriptions"]].rename(columns={"transcriptions": "transcription"})
    df["source_culture"] = "tamil"
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(TEST_DIR, "tamil", f"{x}.jpg"))
    all_dfs.append(df)
    print(f"  Normalized: {len(df)} rows")

    # ── Malayalam ──────────────────────────────────
    # Same schema as Tamil
    print("\n--- MALAYALAM ---")
    df = pd.read_csv(os.path.join(TEST_DIR, "malayalam", "test.csv"))
    print(f"  Raw cols: {list(df.columns)}  rows: {len(df)}")
    df = df[["image_id", "transcriptions"]].rename(columns={"transcriptions": "transcription"})
    df["source_culture"] = "malayalam"
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(TEST_DIR, "malayalam", f"{x}.jpg"))
    all_dfs.append(df)
    print(f"  Normalized: {len(df)} rows")

    # ── Chinese ────────────────────────────────────
    # Columns: image_id, transcriptions, original_labels, indian_labels, irish_labels
    print("\n--- CHINESE ---")
    df = pd.read_csv(os.path.join(TEST_DIR, "chinese", "test.csv"))
    print(f"  Raw cols: {list(df.columns)}  rows: {len(df)}")
    df = df[["image_id", "transcriptions"]].rename(columns={"transcriptions": "transcription"})
    df["source_culture"] = "chinese"
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(TEST_DIR, "chinese", f"{x}.jpg"))
    all_dfs.append(df)
    print(f"  Normalized: {len(df)} rows")

    # ── Western ────────────────────────────────────
    # Columns: image_id, transcriptions, indian_labels, chinese_labels
    print("\n--- WESTERN ---")
    df = pd.read_csv(os.path.join(TEST_DIR, "western", "test.csv"))
    print(f"  Raw cols: {list(df.columns)}  rows: {len(df)}")
    df = df[["image_id", "transcriptions"]].rename(columns={"transcriptions": "transcription"})
    df["source_culture"] = "western"
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(TEST_DIR, "western", f"{x}.jpg"))
    all_dfs.append(df)
    print(f"  Normalized: {len(df)} rows")

    # ── Merge ──────────────────────────────────────
    merged = pd.concat(all_dfs, ignore_index=True)
    merged["transcription"] = merged["transcription"].fillna("").astype(str)

    print(f"\nTotal merged: {len(merged)} rows")
    print(f"Source distribution:\n{merged['source_culture'].value_counts().to_string()}")

    # Validate image presence (sample)
    missing = [r["image_path"] for _, r in merged.iterrows()
               if not os.path.exists(r["image_path"])]
    if missing:
        print(f"\nWARNING: {len(missing)} image files not found. First 5:")
        for p in missing[:5]:
            print(f"  {p}")
    else:
        print("\n[OK] All image files found")

    merged.to_csv(TEST_CSV, index=False)
    print(f"\nSaved: {TEST_CSV}")
    return merged


# ─────────────────────────────────────────────
# STEP 2 — EXTRACT EMBEDDINGS
# ─────────────────────────────────────────────
def extract_clip(df, batch_size=16):
    from transformers import CLIPProcessor, CLIPModel
    print(f"  Loading CLIP ({CLIP_MODEL}) ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()
    all_embs = []
    for start in tqdm(range(0, len(df), batch_size), desc="  CLIP images"):
        batch = df.iloc[start:start + batch_size]
        images = []
        for _, row in batch.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            images.append(img)
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
        all_embs.append(out.numpy())
    return np.concatenate(all_embs, axis=0)


def extract_xlmr(df, batch_size=32):
    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading XLM-R ({XLM_MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL)
    model = AutoModel.from_pretrained(XLM_MODEL)
    model.eval()
    all_embs = []
    for start in tqdm(range(0, len(df), batch_size), desc="  XLM-R text"):
        batch = df.iloc[start:start + batch_size]
        texts = batch["transcription"].fillna("").astype(str).tolist()
        inputs = tokenizer(texts, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            cls = cls / cls.norm(dim=-1, keepdim=True)
        all_embs.append(cls.numpy())
    return np.concatenate(all_embs, axis=0)


def run_embedding_extraction(df):
    print("\n" + "=" * 60)
    print("STEP 2 — Extract embeddings (CLIP + XLM-R)")
    print("=" * 60)

    img_path = os.path.join(TEST_EMB, "image_emb_test.npy")
    txt_path = os.path.join(TEST_EMB, "text_emb_test.npy")

    if os.path.exists(img_path):
        print(f"\nCLIP cache found: {img_path}")
        img_embs = np.load(img_path)
        print(f"  Shape: {img_embs.shape}")
    else:
        print("\nExtracting CLIP embeddings ...")
        img_embs = extract_clip(df)
        np.save(img_path, img_embs)
        print(f"  Saved: {img_path}  {img_embs.shape}")

    if os.path.exists(txt_path):
        print(f"\nXLM-R cache found: {txt_path}")
        txt_embs = np.load(txt_path)
        print(f"  Shape: {txt_embs.shape}")
    else:
        print("\nExtracting XLM-R embeddings ...")
        txt_embs = extract_xlmr(df)
        np.save(txt_path, txt_embs)
        print(f"  Saved: {txt_path}  {txt_embs.shape}")

    assert img_embs.shape == (len(df), CLIP_DIM), \
        f"CLIP shape mismatch: {img_embs.shape} vs expected ({len(df)}, {CLIP_DIM})"
    assert txt_embs.shape == (len(df), TEXT_DIM), \
        f"XLM-R shape mismatch: {txt_embs.shape} vs expected ({len(df)}, {TEXT_DIM})"
    print(f"\n[OK] Embeddings verified: {len(df)} x ({CLIP_DIM}+{TEXT_DIM})")
    return img_embs, txt_embs


# ─────────────────────────────────────────────
# STEP 3 — MODEL DEFINITIONS (mirrors run_full.py)
# ─────────────────────────────────────────────
import torch.nn as nn

class MisogynyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(CLIP_DIM + TEXT_DIM, HIDDEN),
            nn.LayerNorm(HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, LATENT),
            nn.LayerNorm(LATENT), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.head_india   = nn.Linear(LATENT, 1)
        self.head_western = nn.Linear(LATENT, 1)
        self.head_china   = nn.Linear(LATENT, 1)

    def get_latent(self, x):
        return self.trunk(x)

    def forward(self, x):
        lat = self.trunk(x)
        logits = torch.cat([self.head_india(lat),
                            self.head_western(lat),
                            self.head_china(lat)], dim=-1)
        return logits, lat


class GatedCulturalModel(nn.Module):
    def __init__(self, base, pos_protos, neg_protos):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.pos_p = [torch.from_numpy(p).float() for p in pos_protos]
        self.neg_p = [torch.from_numpy(p).float() for p in neg_protos]
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(LATENT + 2, 64), nn.ReLU(),
                          nn.Linear(64, 1), nn.Sigmoid())
            for _ in range(3)
        ])
        self.refine = nn.ModuleList([nn.Linear(LATENT + 2, 1) for _ in range(3)])

    def forward(self, x):
        base_logits, lat = self.base(x)
        lat_norm = nn.functional.normalize(lat, dim=-1)
        outs = []
        for c in range(3):
            pos_norm = nn.functional.normalize(self.pos_p[c], dim=-1)
            neg_norm = nn.functional.normalize(self.neg_p[c], dim=-1)
            sim_pos = torch.mm(lat_norm, pos_norm.t()).max(dim=-1)[0]
            sim_neg = torch.mm(lat_norm, neg_norm.t()).max(dim=-1)[0]
            proto_feat = torch.stack([sim_pos, sim_neg], dim=-1)
            combined = torch.cat([lat, proto_feat], dim=-1)
            g = self.gates[c](combined)
            cultural = self.refine[c](combined)
            base = base_logits[:, c:c+1]
            outs.append(g * cultural + (1 - g) * base)
        return torch.cat(outs, dim=-1), lat


# ─────────────────────────────────────────────
# STEP 4 — LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    print("\n" + "=" * 60)
    print("STEP 3 — Load trained model")
    print("=" * 60)

    # Try cultural model first (best_cultural.pt)
    cultural_ckpt = os.path.join(CHECKPOINT, "best_cultural.pt")
    base_ckpt     = os.path.join(CHECKPOINT, "best_model.pt")

    if os.path.exists(cultural_ckpt):
        ckpt = torch.load(cultural_ckpt, weights_only=False)

        base = MisogynyMLP()
        base.load_state_dict(ckpt["base_state_dict"])

        pos_p = [np.load(os.path.join(PROTO_DIR, f"{n}_pos.npy")) for n in CULTURES]
        neg_p = [np.load(os.path.join(PROTO_DIR, f"{n}_neg.npy")) for n in CULTURES]

        model = GatedCulturalModel(base, pos_p, neg_p)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[OK] Loaded cultural model  (avg F1={ckpt.get('avg_f1', '?'):.3f},"
              f" epoch={ckpt.get('epoch', '?')})")
    elif os.path.exists(base_ckpt):
        ckpt = torch.load(base_ckpt, weights_only=False)
        model = MisogynyMLP()
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[OK] Loaded base model  (avg F1={ckpt.get('avg_f1', '?'):.3f},"
              f" epoch={ckpt.get('epoch', '?')})")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {CHECKPOINT}. "
            f"Expected best_cultural.pt or best_model.pt")

    model.eval()
    return model


# ─────────────────────────────────────────────
# STEP 5 — LOAD THRESHOLDS
# ─────────────────────────────────────────────
def load_thresholds():
    # Stage 8 in run_full.py saves thresholds.json
    for name in ["thresholds.json", "calibrated_thresholds.json"]:
        path = os.path.join(LOG_DIR, name)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # data may be {"india": 0.45, ...} directly
            thresholds = data if isinstance(data, dict) else data.get("thresholds", data)
            print(f"\n[OK] Calibrated thresholds loaded from {name}: {thresholds}")
            return thresholds

    default = {c: 0.5 for c in CULTURES}
    print(f"\nWARNING: No threshold file found — using defaults {default}")
    return default


# ─────────────────────────────────────────────
# STEP 6 — INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, img_embs, txt_embs, thresholds):
    print("\n" + "=" * 60)
    print("STEP 4 — Inference")
    print("=" * 60)

    combined = np.concatenate([img_embs, txt_embs], axis=1).astype(np.float32)
    all_probs = []
    batch_size = 256

    with torch.no_grad():
        for start in tqdm(range(0, len(combined), batch_size), desc="Predicting"):
            batch = torch.from_numpy(combined[start:start + batch_size])
            logits, _ = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.numpy())

    all_probs = np.concatenate(all_probs, axis=0)   # (N, 3)

    t_arr = np.array([
        thresholds.get("india",   0.5),
        thresholds.get("western", 0.5),
        thresholds.get("china",   0.5),
    ])
    preds = (all_probs > t_arr).astype(int)

    print(f"  Predictions shape: {preds.shape}")
    for i, name in enumerate(CULTURES):
        n1 = preds[:, i].sum()
        print(f"  {name}: misogyny={n1}, not={len(preds)-n1}")

    return preds, all_probs


# ─────────────────────────────────────────────
# STEP 7 — FORMAT SUBMISSION
# ─────────────────────────────────────────────
def format_submission(df, preds):
    """
    Task B format:
      image_id, original_culture, irish_culture, chinese_culture
      india → original_culture
      western → irish_culture
      china → chinese_culture
    """
    submission = pd.DataFrame({
        "image_id":          df["image_id"].values,
        "original_culture":  preds[:, 0],   # india
        "irish_culture":     preds[:, 1],   # western
        "chinese_culture":   preds[:, 2],   # china
    })
    return submission


# ─────────────────────────────────────────────
# STEP 8 — VALIDATION
# ─────────────────────────────────────────────
def validate_submission(sub, expected_ids=None):
    print("\n" + "=" * 60)
    print("STEP 5 — Validate submission")
    print("=" * 60)

    errors = []
    required = ["image_id", "original_culture", "irish_culture", "chinese_culture"]
    for col in required:
        if col not in sub.columns:
            errors.append(f"Missing column: {col}")

    for col in ["original_culture", "irish_culture", "chinese_culture"]:
        if col in sub.columns:
            bad = set(sub[col].unique()) - {0, 1}
            if bad:
                errors.append(f"{col} has invalid values: {bad}")

    if expected_ids is not None:
        missing = set(expected_ids) - set(sub["image_id"])
        if missing:
            errors.append(f"{len(missing)} image_ids missing from submission")

    n_dup = sub.duplicated("image_id").sum()

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  [ERR] {e}")
    else:
        print("[OK] All validation checks passed")

    print(f"\nSubmission stats:")
    print(f"  Total rows : {len(sub)}")
    for col in ["original_culture", "irish_culture", "chinese_culture"]:
        d = sub[col].value_counts()
        print(f"  {col:20s}: misogyny={d.get(1,0):4d}  not={d.get(0,0):4d}")
    if n_dup:
        print(f"\n  NOTE: {n_dup} duplicate image_ids "
              f"(expected — same ID can appear across cultures)")

    return len(errors) == 0


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    # 1. Normalize + merge test CSVs
    test_df = normalize_test_csvs()

    # 2. Extract embeddings
    img_embs, txt_embs = run_embedding_extraction(test_df)

    # 3. Load model
    model = load_model()

    # 4. Load thresholds
    thresholds = load_thresholds()

    # 5. Inference
    preds, probs = run_inference(model, img_embs, txt_embs, thresholds)

    # 6. Format
    submission = format_submission(test_df, preds)

    # 7. Save
    out_path = os.path.join(SUBMIT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"\n[OK] Saved: {out_path}")

    # 8. Validate
    ok = validate_submission(submission, expected_ids=test_df["image_id"].tolist())

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    if ok:
        print(f"[OK] PIPELINE COMPLETE — submission.csv is ready to upload")
    else:
        print(f"[WARN]  PIPELINE COMPLETE WITH WARNINGS — check errors above")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")
