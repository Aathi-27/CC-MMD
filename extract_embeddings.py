"""
CC-MMD Stage 3: Extract embeddings (train + dev).
Standalone script with print-based progress (no tqdm buffering issues).
Saves to data/embeddings/ and data/embeddings_dev/
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = "E:/pep"
TRAIN_CSV = os.path.join(ROOT, "train_merged.csv")
DEV_CSV = os.path.join(ROOT, "dev", "dev_final.csv")
EMB_DIR = os.path.join(ROOT, "data", "embeddings")
DEV_EMB_DIR = os.path.join(ROOT, "data", "embeddings_dev")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(DEV_EMB_DIR, exist_ok=True)


def extract_clip(df, batch_size=16):
    from transformers import CLIPProcessor, CLIPModel
    print("  Loading CLIP model...", flush=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print("  CLIP loaded.", flush=True)

    total = (len(df) + batch_size - 1) // batch_size
    all_embs = []
    t0 = time.time()

    for idx, start in enumerate(range(0, len(df), batch_size)):
        batch = df.iloc[start:start + batch_size]
        images = []
        for _, row in batch.iterrows():
            try:
                images.append(Image.open(row["image_path"]).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
        all_embs.append(out.numpy())

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate
            print(f"  CLIP: {idx+1}/{total} batches ({(idx+1)*batch_size}/{len(df)} imgs) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)

    return np.concatenate(all_embs, axis=0)


def extract_xlmr(df, batch_size=32):
    from transformers import AutoTokenizer, AutoModel
    print("  Loading XLM-R model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    model.eval()
    print("  XLM-R loaded.", flush=True)

    total = (len(df) + batch_size - 1) // batch_size
    all_embs = []
    t0 = time.time()

    for idx, start in enumerate(range(0, len(df), batch_size)):
        batch = df.iloc[start:start + batch_size]
        texts = batch["transcription"].fillna("").astype(str).tolist()
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            cls = cls / cls.norm(dim=-1, keepdim=True)
        all_embs.append(cls.numpy())

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate
            print(f"  XLM-R: {idx+1}/{total} batches | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)

    return np.concatenate(all_embs, axis=0)


def process_split(name, csv_path, emb_dir):
    print(f"\n--- {name} ---", flush=True)
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}", flush=True)

    img_path = os.path.join(emb_dir, "clip_image.npy")
    txt_path = os.path.join(emb_dir, "xlmr_text.npy")

    if os.path.exists(img_path) and os.path.exists(txt_path):
        ie = np.load(img_path)
        te = np.load(txt_path)
        print(f"  Already cached: img={ie.shape}, txt={te.shape}", flush=True)
        return

    if not os.path.exists(img_path):
        img_embs = extract_clip(df)
        np.save(img_path, img_embs)
        print(f"  Saved CLIP: {img_path} {img_embs.shape}", flush=True)
    
    if not os.path.exists(txt_path):
        txt_embs = extract_xlmr(df)
        np.save(txt_path, txt_embs)
        print(f"  Saved XLM-R: {txt_path} {txt_embs.shape}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60, flush=True)
    print("STAGE 3: Extract Embeddings", flush=True)
    print("=" * 60, flush=True)

    process_split("TRAIN", TRAIN_CSV, EMB_DIR)
    process_split("DEV", DEV_CSV, DEV_EMB_DIR)

    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
