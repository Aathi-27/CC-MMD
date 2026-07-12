"""
CC-MMD Full Pipeline: End-to-End Execution
Stages 3-10 in one deterministic script.

Prerequisites (already done):
  - train_merged.csv exists
  - dev/dev_final.csv exists
  - Images in data/image/{culture}/

This script:
  Stage 3: Extract embeddings (train + dev)
  Stage 4: Dataset loader (embedding level)
  Stage 5: Train MLP model
  Stage 6: Build cultural prototypes
  Stage 7: Fine-tune gated cultural layer
  Stage 8: Threshold calibration (on dev)
  Stage 9: Test inference
  Stage 10: Generate submission CSV
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image

# ============================================================
# CONFIG (inline, no imports)
# ============================================================
ROOT = "E:/pep"
TRAIN_CSV = os.path.join(ROOT, "train_merged.csv")
DEV_CSV = os.path.join(ROOT, "dev", "dev_final.csv")

EMB_DIR = os.path.join(ROOT, "data", "embeddings")
DEV_EMB_DIR = os.path.join(ROOT, "data", "embeddings_dev")
CHECKPOINT_DIR = os.path.join(ROOT, "results", "checkpoints")
LOG_DIR = os.path.join(ROOT, "results", "logs")
SUBMISSION_DIR = os.path.join(ROOT, "results", "submissions")

for d in [EMB_DIR, DEV_EMB_DIR, CHECKPOINT_DIR, LOG_DIR, SUBMISSION_DIR]:
    os.makedirs(d, exist_ok=True)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
XLM_MODEL_NAME = "xlm-roberta-base"
CLIP_DIM = 512
TEXT_DIM = 768

LABEL_COLS = ["india_label", "western_label", "china_label"]
CULTURE_NAMES = ["india", "western", "china"]

# Training
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
PATIENCE = 8
DROPOUT = 0.3
HIDDEN = 512
LATENT = 256
N_PROTO = 6


# ============================================================
# STAGE 3: EMBEDDING EXTRACTION
# ============================================================
def extract_clip_embeddings(df, batch_size=16):
    from transformers import CLIPProcessor, CLIPModel

    print(f"  Loading CLIP: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    all_embs = []
    for start in tqdm(range(0, len(df), batch_size), desc="  CLIP"):
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


def extract_text_embeddings(df, batch_size=32):
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading XLM-R: {XLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL_NAME)
    model = AutoModel.from_pretrained(XLM_MODEL_NAME)
    model.eval()

    all_embs = []
    for start in tqdm(range(0, len(df), batch_size), desc="  XLM-R"):
        batch = df.iloc[start:start + batch_size]
        texts = batch["transcription"].fillna("").astype(str).tolist()
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            cls = cls / cls.norm(dim=-1, keepdim=True)
        all_embs.append(cls.numpy())

    return np.concatenate(all_embs, axis=0)


def run_stage3():
    print("=" * 60)
    print("STAGE 3: Extract Embeddings")
    print("=" * 60)

    for name, csv_path, emb_dir in [("TRAIN", TRAIN_CSV, EMB_DIR), ("DEV", DEV_CSV, DEV_EMB_DIR)]:
        print(f"\n--- {name} ---")
        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}")

        img_path = os.path.join(emb_dir, "clip_image.npy")
        txt_path = os.path.join(emb_dir, "xlmr_text.npy")

        if os.path.exists(img_path) and os.path.exists(txt_path):
            print(f"  Already cached, skipping.")
            continue

        if not os.path.exists(img_path):
            img_embs = extract_clip_embeddings(df)
            np.save(img_path, img_embs)
            print(f"  Saved: {img_path} {img_embs.shape}")
        
        if not os.path.exists(txt_path):
            txt_embs = extract_text_embeddings(df)
            np.save(txt_path, txt_embs)
            print(f"  Saved: {txt_path} {txt_embs.shape}")


# ============================================================
# STAGE 4: DATASET LOADER
# ============================================================
class EmbeddingDataset(Dataset):
    def __init__(self, image_embs, text_embs, df):
        self.X = np.concatenate([image_embs, text_embs], axis=1).astype(np.float32)
        labels = df[LABEL_COLS].values.astype(np.float64)
        self.mask = (~np.isnan(labels)).astype(np.float32)
        self.labels = np.nan_to_num(labels, nan=0.0).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.mask[idx]),
        )


# ============================================================
# STAGE 5: MODEL
# ============================================================
class MisogynyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(CLIP_DIM + TEXT_DIM, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, LATENT),
            nn.LayerNorm(LATENT),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.head_india = nn.Linear(LATENT, 1)
        self.head_western = nn.Linear(LATENT, 1)
        self.head_china = nn.Linear(LATENT, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_latent(self, x):
        return self.trunk(x)

    def forward(self, x):
        lat = self.trunk(x)
        logits = torch.cat([
            self.head_india(lat),
            self.head_western(lat),
            self.head_china(lat),
        ], dim=-1)
        return logits, lat


def masked_bce(logits, labels, mask, pos_weights):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pw = pos_weights.unsqueeze(0).expand_as(logits)
    weight = labels * (pw - 1) + 1.0
    bce = bce * weight * mask
    n = mask.sum()
    return bce.sum() / n if n > 0 else bce.sum()


def compute_class_weights(df):
    weights = []
    for col in LABEL_COLS:
        valid = df[col].dropna()
        pos = (valid == 1).sum()
        neg = (valid == 0).sum()
        w = neg / pos if pos > 0 else 1.0
        weights.append(w)
        print(f"  {col}: pos={pos}, neg={neg}, weight={w:.2f}")
    return torch.tensor(weights, dtype=torch.float32)


def eval_model(model, loader, pos_weights, threshold=0.5):
    model.eval()
    all_logits, all_labels, all_masks = [], [], []
    total_loss = 0
    n = 0
    with torch.no_grad():
        for x, y, m in loader:
            logits, _ = model(x)
            total_loss += masked_bce(logits, y, m, pos_weights).item()
            n += 1
            all_logits.append(logits)
            all_labels.append(y)
            all_masks.append(m)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_masks = torch.cat(all_masks)
    probs = torch.sigmoid(all_logits)
    preds = (probs > threshold).float()

    f1s = {}
    for i, name in enumerate(CULTURE_NAMES):
        valid = all_masks[:, i] == 1.0
        if valid.sum() == 0:
            f1s[name] = float("nan")
            continue
        f1s[name] = f1_score(
            all_labels[valid, i].numpy(),
            preds[valid, i].numpy(),
            average="macro", zero_division=0
        )

    valid_f1 = [v for v in f1s.values() if not np.isnan(v)]
    avg = np.mean(valid_f1) if valid_f1 else 0.0
    return total_loss / max(n, 1), f1s, avg


def run_stage5():
    print("=" * 60)
    print("STAGE 5: Train MLP Model")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)

    train_img = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    train_txt = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))
    dev_img = np.load(os.path.join(DEV_EMB_DIR, "clip_image.npy"))
    dev_txt = np.load(os.path.join(DEV_EMB_DIR, "xlmr_text.npy"))

    print(f"Train: {len(train_df)} | Dev: {len(dev_df)}")

    train_ds = EmbeddingDataset(train_img, train_txt, train_df)
    dev_ds = EmbeddingDataset(dev_img, dev_txt, dev_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nClass weights:")
    pos_weights = compute_class_weights(train_df)

    model = MisogynyMLP()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_f1 = 0.0
    patience_ctr = 0
    history = []

    print(f"\nTraining up to {EPOCHS} epochs, patience={PATIENCE}")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        nb = 0
        for x, y, m in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = masked_bce(logits, y, m, pos_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            nb += 1
        scheduler.step()

        v_loss, cf1, avg_f1 = eval_model(model, dev_loader, pos_weights)
        lr = optimizer.param_groups[0]["lr"]

        f1_str = " | ".join(
            f"{n}={f:.3f}" if not np.isnan(f) else f"{n}=N/A"
            for n, f in cf1.items()
        )
        print(f"E{epoch:3d} | Tr={t_loss/max(nb,1):.4f} Val={v_loss:.4f} | {f1_str} | Avg={avg_f1:.3f} | LR={lr:.1e}")

        history.append({"epoch": epoch, "avg_f1": avg_f1, "culture_f1": cf1})

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "avg_f1": avg_f1,
                "culture_f1": cf1,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> BEST saved (avg={avg_f1:.3f})")
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch}")
            break

    with open(os.path.join(LOG_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nBest avg Macro F1: {best_f1:.3f}")
    return model


# ============================================================
# STAGE 6: CULTURAL PROTOTYPES
# ============================================================
def run_stage6():
    print("\n" + "=" * 60)
    print("STAGE 6: Build Cultural Prototypes")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_CSV)
    train_img = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    train_txt = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))

    model = MisogynyMLP()
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    # Extract latents
    combined = np.concatenate([train_img, train_txt], axis=1).astype(np.float32)
    all_latents = []
    with torch.no_grad():
        for s in range(0, len(combined), 256):
            lat = model.get_latent(torch.from_numpy(combined[s:s+256]))
            all_latents.append(lat.numpy())
    latents = np.concatenate(all_latents)
    print(f"Latents: {latents.shape}")

    proto_dir = os.path.join(ROOT, "results", "prototypes")
    os.makedirs(proto_dir, exist_ok=True)

    for i, (col, name) in enumerate(zip(LABEL_COLS, CULTURE_NAMES)):
        valid = train_df[col].notna().values
        labs = train_df[col].values[valid]
        lats = latents[valid]

        pos_l = lats[labs == 1]
        neg_l = lats[labs == 0]

        k_pos = min(N_PROTO, len(pos_l)) if len(pos_l) > 0 else 1
        k_neg = min(N_PROTO, len(neg_l)) if len(neg_l) > 0 else 1

        pos_c = KMeans(k_pos, random_state=42, n_init=10).fit(pos_l).cluster_centers_ if len(pos_l) > 0 else np.zeros((1, LATENT))
        neg_c = KMeans(k_neg, random_state=42, n_init=10).fit(neg_l).cluster_centers_ if len(neg_l) > 0 else np.zeros((1, LATENT))

        np.save(os.path.join(proto_dir, f"{name}_pos.npy"), pos_c.astype(np.float32))
        np.save(os.path.join(proto_dir, f"{name}_neg.npy"), neg_c.astype(np.float32))
        print(f"  {name}: pos={pos_c.shape}, neg={neg_c.shape}")

    np.save(os.path.join(EMB_DIR, "latents_train.npy"), latents)
    print("Prototypes saved.")


# ============================================================
# STAGE 7: GATED CULTURAL LAYER (fine-tune gates only)
# ============================================================
class GatedCulturalModel(nn.Module):
    def __init__(self, base_model, pos_protos, neg_protos):
        super().__init__()
        self.base = base_model
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        self.pos_p = [torch.from_numpy(p).float() for p in pos_protos]
        self.neg_p = [torch.from_numpy(p).float() for p in neg_protos]

        # Gate + refinement per culture
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(LATENT + 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
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


def run_stage7():
    print("\n" + "=" * 60)
    print("STAGE 7: Fine-tune Gated Cultural Layer")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)

    train_img = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    train_txt = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))
    dev_img = np.load(os.path.join(DEV_EMB_DIR, "clip_image.npy"))
    dev_txt = np.load(os.path.join(DEV_EMB_DIR, "xlmr_text.npy"))

    train_loader = DataLoader(EmbeddingDataset(train_img, train_txt, train_df), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(EmbeddingDataset(dev_img, dev_txt, dev_df), batch_size=BATCH_SIZE, shuffle=False)

    pos_weights = compute_class_weights(train_df)

    # Load base model
    base = MisogynyMLP()
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), weights_only=False)
    base.load_state_dict(ckpt["model_state_dict"])

    # Load prototypes
    proto_dir = os.path.join(ROOT, "results", "prototypes")
    pos_p = [np.load(os.path.join(proto_dir, f"{n}_pos.npy")) for n in CULTURE_NAMES]
    neg_p = [np.load(os.path.join(proto_dir, f"{n}_neg.npy")) for n in CULTURE_NAMES]

    model = GatedCulturalModel(base, pos_p, neg_p)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (gates only): {trainable:,}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

    best_f1 = 0.0
    print("-" * 80)

    for epoch in range(1, 16):
        model.train()
        tl, nb = 0, 0
        for x, y, m in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = masked_bce(logits, y, m, pos_weights)
            loss.backward()
            optimizer.step()
            tl += loss.item()
            nb += 1

        vl, cf1, avg = eval_model(model, dev_loader, pos_weights)
        f1s = " | ".join(f"{n}={f:.3f}" if not np.isnan(f) else f"{n}=N/A" for n, f in cf1.items())
        print(f"E{epoch:2d} | Tr={tl/max(nb,1):.4f} Val={vl:.4f} | {f1s} | Avg={avg:.3f}")

        if avg > best_f1:
            best_f1 = avg
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "base_state_dict": base.state_dict(),
                "avg_f1": avg,
                "culture_f1": cf1,
                "has_prototypes": True,
            }, os.path.join(CHECKPOINT_DIR, "best_cultural.pt"))
            print(f"  -> BEST cultural saved (avg={avg:.3f})")

    print(f"\nBest gated F1: {best_f1:.3f}")


# ============================================================
# STAGE 8: THRESHOLD CALIBRATION (on dev)
# ============================================================
def run_stage8():
    print("\n" + "=" * 60)
    print("STAGE 8: Threshold Calibration")
    print("=" * 60)

    dev_df = pd.read_csv(DEV_CSV)
    dev_img = np.load(os.path.join(DEV_EMB_DIR, "clip_image.npy"))
    dev_txt = np.load(os.path.join(DEV_EMB_DIR, "xlmr_text.npy"))

    # Load best model (try cultural first)
    cultural_path = os.path.join(CHECKPOINT_DIR, "best_cultural.pt")
    base_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if os.path.exists(cultural_path):
        base = MisogynyMLP()
        ckpt = torch.load(cultural_path, weights_only=False)
        base.load_state_dict(ckpt["base_state_dict"])
        proto_dir = os.path.join(ROOT, "results", "prototypes")
        pos_p = [np.load(os.path.join(proto_dir, f"{n}_pos.npy")) for n in CULTURE_NAMES]
        neg_p = [np.load(os.path.join(proto_dir, f"{n}_neg.npy")) for n in CULTURE_NAMES]
        model = GatedCulturalModel(base, pos_p, neg_p)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Using cultural model (F1={ckpt['avg_f1']:.3f})")
    else:
        model = MisogynyMLP()
        ckpt = torch.load(base_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Using base model (F1={ckpt['avg_f1']:.3f})")

    model.eval()

    # Get dev predictions
    combined = np.concatenate([dev_img, dev_txt], axis=1).astype(np.float32)
    all_probs = []
    with torch.no_grad():
        for s in range(0, len(combined), 256):
            logits, _ = model(torch.from_numpy(combined[s:s+256]))
            all_probs.append(torch.sigmoid(logits).numpy())
    probs = np.concatenate(all_probs)

    # Sweep thresholds
    thresholds = {}
    print("\nCalibrating:")
    for i, (col, name) in enumerate(zip(LABEL_COLS, CULTURE_NAMES)):
        valid = dev_df[col].notna().values
        if valid.sum() == 0:
            thresholds[name] = 0.5
            print(f"  {name}: no valid labels, using 0.5")
            continue

        y_true = dev_df[col].values[valid]
        p = probs[valid, i]

        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.30, 0.71, 0.01):
            f = f1_score(y_true, (p > t).astype(float), average="macro", zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t

        thresholds[name] = round(best_t, 3)

        # Compare with 0.5
        f_default = f1_score(y_true, (p > 0.5).astype(float), average="macro", zero_division=0)
        print(f"  {name}: threshold={best_t:.3f}, F1={best_f:.4f} (vs 0.5: {f_default:.4f}, gain={best_f - f_default:+.4f})")

    cal_path = os.path.join(LOG_DIR, "thresholds.json")
    with open(cal_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved: {cal_path}")


# ============================================================
# STAGE 9+10: INFERENCE + SUBMISSION
# ============================================================
def run_stage9_10():
    print("\n" + "=" * 60)
    print("STAGE 9-10: Inference + Submission")
    print("=" * 60)

    # Load thresholds
    cal_path = os.path.join(LOG_DIR, "thresholds.json")
    with open(cal_path) as f:
        thresholds = json.load(f)
    print(f"Thresholds: {thresholds}")

    # Load model
    cultural_path = os.path.join(CHECKPOINT_DIR, "best_cultural.pt")
    base_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if os.path.exists(cultural_path):
        base = MisogynyMLP()
        ckpt = torch.load(cultural_path, weights_only=False)
        base.load_state_dict(ckpt["base_state_dict"])
        proto_dir = os.path.join(ROOT, "results", "prototypes")
        pos_p = [np.load(os.path.join(proto_dir, f"{n}_pos.npy")) for n in CULTURE_NAMES]
        neg_p = [np.load(os.path.join(proto_dir, f"{n}_neg.npy")) for n in CULTURE_NAMES]
        model = GatedCulturalModel(base, pos_p, neg_p)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Using cultural model")
    else:
        model = MisogynyMLP()
        ckpt = torch.load(base_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Using base model")

    model.eval()

    # Run on dev set (as proxy for test — test set not released yet)
    dev_df = pd.read_csv(DEV_CSV)
    dev_img = np.load(os.path.join(DEV_EMB_DIR, "clip_image.npy"))
    dev_txt = np.load(os.path.join(DEV_EMB_DIR, "xlmr_text.npy"))

    combined = np.concatenate([dev_img, dev_txt], axis=1).astype(np.float32)
    all_probs = []
    with torch.no_grad():
        for s in range(0, len(combined), 256):
            logits, _ = model(torch.from_numpy(combined[s:s+256]))
            all_probs.append(torch.sigmoid(logits).numpy())
    probs = np.concatenate(all_probs)

    # Apply thresholds
    t_arr = np.array([thresholds["india"], thresholds["western"], thresholds["china"]])
    preds = (probs > t_arr).astype(int)

    # Format submission
    # NOTE: same image_id can appear across cultures (different folders),
    # so we use row position for uniqueness — submission file uses raw image_id per row.
    submission = pd.DataFrame({
        "image_id": dev_df["image_id"].values,
        "original_culture": preds[:, 0],    # india
        "irish_culture": preds[:, 1],       # western
        "chinese_culture": preds[:, 2],     # china
    })

    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)

    # Validate
    print(f"\nSubmission: {len(submission)} rows")
    for col in ["original_culture", "irish_culture", "chinese_culture"]:
        d = submission[col].value_counts()
        print(f"  {col}: 1={d.get(1,0)}, 0={d.get(0,0)}")

    # Check values are all 0 or 1
    vals = set()
    for col in ["original_culture", "irish_culture", "chinese_culture"]:
        vals.update(submission[col].unique())
    assert vals.issubset({0, 1}), f"Invalid prediction values: {vals}"

    # Warn about duplicate image_ids (expected: same ID exists across cultures)
    n_dup = submission.duplicated("image_id").sum()
    if n_dup > 0:
        print(f"  NOTE: {n_dup} duplicate image_ids (expected — cross-culture datasets share IDs)")

    # Score against dev labels
    print("\nDev set Macro F1 (per culture):")
    f1_scores = []
    for i, (col, name, sub_col) in enumerate(zip(
        LABEL_COLS, CULTURE_NAMES,
        ["original_culture", "irish_culture", "chinese_culture"]
    )):
        valid = dev_df[col].notna().values
        if valid.sum() == 0:
            print(f"  {name}: N/A (no labels)")
            continue
        y_true = dev_df[col].values[valid]
        y_pred = submission[sub_col].values[valid]
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_scores.append(f1)
        print(f"  {name}: {f1:.4f}")

    if f1_scores:
        print(f"\n  → Culture-balanced Avg Macro F1: {np.mean(f1_scores):.4f}")

    print(f"\n✓ Saved: {out_path}")
    print("✓ Submission validated")


# ============================================================
# MAIN: Execute all stages
# ============================================================
if __name__ == "__main__":
    t0 = time.time()

    run_stage3()
    run_stage5()
    run_stage6()
    run_stage7()
    run_stage8()
    run_stage9_10()

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {total:.0f}s ({total/60:.1f}min)")
    print(f"{'=' * 60}")
