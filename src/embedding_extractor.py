"""
Stage 2: Extract and cache embeddings from frozen encoders.

CLIP ViT-B/32 → 512-d image embedding
XLM-RoBERTa-base → 768-d text embedding

Runs once on CPU. All downstream training operates on cached .npy files.
"""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MERGED_CSV, EMB_DIR, CLIP_MODEL, XLM_MODEL, CLIP_DIM, TEXT_DIM
)


def extract_clip_embeddings(df, batch_size=16):
    """Extract CLIP image embeddings for all rows. Returns (N, 512) array."""
    from transformers import CLIPProcessor, CLIPModel

    print(f"Loading CLIP: {CLIP_MODEL}")
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    all_embs = []
    failed_indices = []

    for start in tqdm(range(0, len(df), batch_size), desc="CLIP images"):
        batch = df.iloc[start:start + batch_size]
        images = []
        valid_mask = []

        for _, row in batch.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                images.append(img)
                valid_mask.append(True)
            except Exception as e:
                # Create blank image as fallback
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
                valid_mask.append(False)
                failed_indices.append(row.name)

        inputs = processor(images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # L2 normalize
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

        all_embs.append(outputs.numpy())

    if failed_indices:
        print(f"  WARNING: {len(failed_indices)} images failed to load")

    return np.concatenate(all_embs, axis=0)


def extract_text_embeddings(df, batch_size=32):
    """Extract XLM-R text embeddings for all transcriptions. Returns (N, 768) array."""
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading XLM-R: {XLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL)
    model = AutoModel.from_pretrained(XLM_MODEL)
    model.eval()

    all_embs = []

    for start in tqdm(range(0, len(df), batch_size), desc="XLM-R text"):
        batch = df.iloc[start:start + batch_size]
        texts = batch["transcription"].fillna("").astype(str).tolist()

        # Truncate to max 128 tokens (meme text is short)
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :]
            # L2 normalize
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)

        all_embs.append(cls_emb.numpy())

    return np.concatenate(all_embs, axis=0)


def main():
    print("=" * 60)
    print("STAGE 2: Extract embeddings (one-time, frozen encoders)")
    print("=" * 60)

    df = pd.read_csv(MERGED_CSV)
    print(f"Dataset: {len(df)} samples")

    # Paths for cached embeddings
    image_emb_path = os.path.join(EMB_DIR, "clip_image.npy")
    text_emb_path = os.path.join(EMB_DIR, "xlmr_text.npy")

    # --- CLIP image embeddings ---
    if os.path.exists(image_emb_path):
        print(f"\nCLIP embeddings already cached: {image_emb_path}")
        image_embs = np.load(image_emb_path)
        print(f"  Shape: {image_embs.shape}")
    else:
        print(f"\nExtracting CLIP image embeddings...")
        image_embs = extract_clip_embeddings(df, batch_size=16)
        np.save(image_emb_path, image_embs)
        print(f"  Saved: {image_emb_path}, shape: {image_embs.shape}")

    # --- XLM-R text embeddings ---
    if os.path.exists(text_emb_path):
        print(f"\nXLM-R embeddings already cached: {text_emb_path}")
        text_embs = np.load(text_emb_path)
        print(f"  Shape: {text_embs.shape}")
    else:
        print(f"\nExtracting XLM-R text embeddings...")
        text_embs = extract_text_embeddings(df, batch_size=32)
        np.save(text_emb_path, text_embs)
        print(f"  Saved: {text_emb_path}, shape: {text_embs.shape}")

    # Verify
    assert image_embs.shape == (len(df), CLIP_DIM), f"CLIP shape mismatch: {image_embs.shape}"
    assert text_embs.shape == (len(df), TEXT_DIM), f"XLM-R shape mismatch: {text_embs.shape}"
    print(f"\n✓ All embeddings verified: {len(df)} samples × ({CLIP_DIM} + {TEXT_DIM}) = {CLIP_DIM + TEXT_DIM}d")


if __name__ == "__main__":
    main()
