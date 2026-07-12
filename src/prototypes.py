"""
Stage 5: Build class-conditional cultural prototypes.

After training the core model:
1. Extract 256-d latent representations for all training samples
2. For each culture, cluster misogyny samples and non-misogyny samples separately
3. Store centroids as prototype vectors
4. Optionally fine-tune the gating layer with prototypes active

Key difference from naive K-means:
- Class-conditional: separate clusters for positive and negative samples
- Relative scoring: delta = sim_pos - sim_neg (not raw similarity)
- Only samples WITH valid labels for that culture are used
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MERGED_CSV, EMB_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    COMBINED_DIM, HIDDEN_DIM, LATENT_DIM, DROPOUT, NUM_CULTURES,
    LABEL_COLS, CULTURE_NAMES, NUM_PROTOTYPES_PER_CLASS
)
from src.model import FullModel


def extract_all_latents(model, image_embs, text_embs, batch_size=256):
    """
    Run all embeddings through the trained trunk to get 256-d latent representations.
    """
    model.eval()
    combined = np.concatenate([image_embs, text_embs], axis=1).astype(np.float32)

    all_latents = []
    for start in range(0, len(combined), batch_size):
        batch = torch.from_numpy(combined[start:start + batch_size])
        with torch.no_grad():
            latent = model.get_latent(batch)
        all_latents.append(latent.numpy())

    return np.concatenate(all_latents, axis=0)  # (N, 256)


def build_prototypes_for_culture(latents, labels, n_clusters):
    """
    Build class-conditional prototypes for one culture.

    Args:
        latents: (N_valid, 256) — latent representations for samples with valid labels
        labels: (N_valid,) — 0 or 1
        n_clusters: number of prototypes per class

    Returns:
        pos_centroids: (K, 256) — misogyny cluster centroids
        neg_centroids: (K, 256) — not-misogyny cluster centroids
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_latents = latents[pos_mask]
    neg_latents = latents[neg_mask]

    # Handle edge cases
    if len(pos_latents) < n_clusters:
        n_pos = max(1, len(pos_latents))
    else:
        n_pos = n_clusters

    if len(neg_latents) < n_clusters:
        n_neg = max(1, len(neg_latents))
    else:
        n_neg = n_clusters

    # Cluster positive samples
    if len(pos_latents) > 0:
        km_pos = KMeans(n_clusters=n_pos, random_state=42, n_init=10)
        km_pos.fit(pos_latents)
        pos_centroids = km_pos.cluster_centers_
    else:
        pos_centroids = np.zeros((1, latents.shape[1]))

    # Cluster negative samples
    if len(neg_latents) > 0:
        km_neg = KMeans(n_clusters=n_neg, random_state=42, n_init=10)
        km_neg.fit(neg_latents)
        neg_centroids = km_neg.cluster_centers_
    else:
        neg_centroids = np.zeros((1, latents.shape[1]))

    return pos_centroids.astype(np.float32), neg_centroids.astype(np.float32)


def main():
    print("=" * 60)
    print("STAGE 5: Build Cultural Prototypes")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MERGED_CSV)
    image_embs = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    text_embs = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))
    train_indices = np.load(os.path.join(EMB_DIR, "train_indices.npy"))

    # Load trained model
    model = FullModel(
        input_dim=COMBINED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
        num_cultures=NUM_CULTURES,
    )

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']}, avg F1={ckpt['avg_f1']:.3f}")

    # Extract latents for training data only
    print("\nExtracting latent representations...")
    all_latents = extract_all_latents(model, image_embs, text_embs)
    train_latents = all_latents[train_indices]
    train_df = df.iloc[train_indices]

    print(f"Training latents: {train_latents.shape}")

    # Build prototypes per culture
    proto_dir = os.path.join(RESULTS_DIR, "prototypes")
    os.makedirs(proto_dir, exist_ok=True)

    pos_prototypes = []
    neg_prototypes = []

    for i, (col, name) in enumerate(zip(LABEL_COLS, CULTURE_NAMES)):
        print(f"\n--- {name.upper()} ---")

        # Only use samples with valid labels for this culture
        valid_mask = train_df[col].notna().values
        valid_labels = train_df[col].values[valid_mask]
        valid_latents = train_latents[valid_mask]

        print(f"  Valid samples: {len(valid_latents)}")
        print(f"  Positive: {(valid_labels == 1).sum()}, Negative: {(valid_labels == 0).sum()}")

        pos_c, neg_c = build_prototypes_for_culture(
            valid_latents, valid_labels, NUM_PROTOTYPES_PER_CLASS
        )

        print(f"  Positive prototypes: {pos_c.shape}")
        print(f"  Negative prototypes: {neg_c.shape}")

        # Save
        np.save(os.path.join(proto_dir, f"{name}_pos.npy"), pos_c)
        np.save(os.path.join(proto_dir, f"{name}_neg.npy"), neg_c)

        pos_prototypes.append(pos_c)
        neg_prototypes.append(neg_c)

    # Save all latents for potential reuse
    np.save(os.path.join(EMB_DIR, "latents_all.npy"), all_latents)

    print(f"\n{'=' * 60}")
    print("Prototypes saved to:", proto_dir)
    print(f"{'=' * 60}")

    return pos_prototypes, neg_prototypes


if __name__ == "__main__":
    main()
