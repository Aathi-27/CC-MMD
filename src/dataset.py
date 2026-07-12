"""
PyTorch Dataset for cached embeddings with missing-label masking.

Key design decisions:
- Loads pre-computed .npy embeddings (no image/text processing at train time)
- Returns a mask tensor alongside labels: 1.0 = valid label, 0.0 = missing (NaN)
- Loss computation must multiply by mask to ignore missing labels
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MemeEmbeddingDataset(Dataset):
    """
    Dataset operating on cached embeddings.

    Returns:
        embedding: (1280,) float32 — concatenated [CLIP_512 | XLM-R_768]
        labels: (3,) float32 — [india, western, china], NaN → 0.0
        mask: (3,) float32 — 1.0 where label is valid, 0.0 where NaN
        culture_id: int — source culture index (0=tamil, 1=malayalam, 2=chinese, 3=western)
    """

    CULTURE_MAP = {"tamil": 0, "malayalam": 1, "chinese": 2, "western": 3}
    LABEL_COLS = ["india_label", "western_label", "china_label"]

    def __init__(self, image_embs, text_embs, df):
        """
        Args:
            image_embs: np.ndarray (N, 512)
            text_embs: np.ndarray (N, 768)
            df: pd.DataFrame with label columns and source_culture
        """
        assert len(image_embs) == len(text_embs) == len(df)

        # Concatenate embeddings: [CLIP | XLM-R]
        self.embeddings = np.concatenate([image_embs, text_embs], axis=1).astype(np.float32)

        # Labels and mask
        labels_raw = df[self.LABEL_COLS].values.astype(np.float64)
        self.mask = (~np.isnan(labels_raw)).astype(np.float32)
        self.labels = np.nan_to_num(labels_raw, nan=0.0).astype(np.float32)

        # Source culture
        self.culture_ids = df["source_culture"].map(self.CULTURE_MAP).values.astype(np.int64)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(self.culture_ids[idx], dtype=torch.long),
        )


def create_dataloaders(image_embs, text_embs, df, val_frac=0.15, batch_size=64, seed=42):
    """
    Split data into train/val and create DataLoaders.

    Stratification: split per source_culture to maintain culture balance.
    """
    np.random.seed(seed)

    train_indices = []
    val_indices = []

    for culture in df["source_culture"].unique():
        culture_idx = df[df["source_culture"] == culture].index.tolist()
        np.random.shuffle(culture_idx)
        split = int(len(culture_idx) * (1 - val_frac))
        train_indices.extend(culture_idx[:split])
        val_indices.extend(culture_idx[split:])

    # Create datasets
    train_ds = MemeEmbeddingDataset(
        image_embs[train_indices], text_embs[train_indices],
        df.iloc[train_indices].reset_index(drop=True)
    )
    val_ds = MemeEmbeddingDataset(
        image_embs[val_indices], text_embs[val_indices],
        df.iloc[val_indices].reset_index(drop=True)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Train label coverage:")
    for i, col in enumerate(MemeEmbeddingDataset.LABEL_COLS):
        valid = train_ds.mask[:, i].sum()
        print(f"  {col}: {int(valid)}/{len(train_ds)} valid")

    return train_loader, val_loader, train_indices, val_indices
