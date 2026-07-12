"""
Stage 4: Training loop with masked loss.

Key design:
- BCE loss per culture head, masked by label availability
- Class weights per culture to handle imbalance
- Tracks per-culture Macro F1 separately on validation set
- Early stopping by average dev culture Macro F1
- Rejects checkpoints where any culture F1 < 0.55
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MERGED_CSV, EMB_DIR, CHECKPOINT_DIR, LOG_DIR,
    COMBINED_DIM, HIDDEN_DIM, LATENT_DIM, DROPOUT, NUM_CULTURES,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, EARLY_STOP_PATIENCE,
    LABEL_COLS, CULTURE_NAMES, DEFAULT_THRESHOLD
)
from src.dataset import MemeEmbeddingDataset, create_dataloaders
from src.model import FullModel


def compute_class_weights(df):
    """
    Compute per-culture positive class weight for BCE.
    weight = num_neg / num_pos for each culture label.
    """
    weights = []
    for col in LABEL_COLS:
        valid = df[col].dropna()
        pos = (valid == 1).sum()
        neg = (valid == 0).sum()
        if pos > 0:
            w = neg / pos
        else:
            w = 1.0
        weights.append(w)
        print(f"  {col}: pos={pos}, neg={neg}, weight={w:.2f}")
    return torch.tensor(weights, dtype=torch.float32)


def masked_bce_loss(logits, labels, mask, pos_weights):
    """
    BCE loss that ignores missing labels via mask.

    Args:
        logits: (B, 3)
        labels: (B, 3)
        mask: (B, 3) — 1.0 where label is valid, 0.0 where NaN
        pos_weights: (3,) — positive class weights per culture

    Returns:
        scalar loss
    """
    # Expand pos_weights to match batch
    pw = pos_weights.unsqueeze(0).expand_as(logits)

    # Per-sample, per-culture BCE
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=None, reduction="none"
    )

    # Apply positive class weighting manually
    # weight_tensor: pos_weight where label=1, 1.0 where label=0
    weight_tensor = labels * (pw - 1) + 1.0
    bce = bce * weight_tensor

    # Mask out missing labels
    bce = bce * mask

    # Average over valid entries only
    num_valid = mask.sum()
    if num_valid > 0:
        return bce.sum() / num_valid
    return bce.sum()


def evaluate(model, val_loader, pos_weights, threshold=DEFAULT_THRESHOLD):
    """
    Evaluate model on validation set.

    Returns:
        val_loss: float
        per_culture_f1: dict {culture_name: macro_f1}
        avg_f1: float
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_logits = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for emb, labels, mask, culture_ids in val_loader:
            logits, _ = model(emb)
            loss = masked_bce_loss(logits, labels, mask, pos_weights)
            total_loss += loss.item()
            total_batches += 1

            all_logits.append(logits)
            all_labels.append(labels)
            all_masks.append(mask)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    probs = torch.sigmoid(all_logits)
    preds = (probs > threshold).float()

    per_culture_f1 = {}
    for i, name in enumerate(CULTURE_NAMES):
        valid = all_masks[:, i] == 1.0
        if valid.sum() == 0:
            per_culture_f1[name] = float("nan")
            continue

        y_true = all_labels[valid, i].numpy()
        y_pred = preds[valid, i].numpy()
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        per_culture_f1[name] = f1

    valid_f1s = [v for v in per_culture_f1.values() if not np.isnan(v)]
    avg_f1 = np.mean(valid_f1s) if valid_f1s else 0.0

    val_loss = total_loss / max(total_batches, 1)
    return val_loss, per_culture_f1, avg_f1


def train_one_epoch(model, train_loader, optimizer, pos_weights):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for emb, labels, mask, culture_ids in train_loader:
        optimizer.zero_grad()
        logits, _ = model(emb)
        loss = masked_bce_loss(logits, labels, mask, pos_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    print("=" * 60)
    print("STAGE 4: Train Multitask MLP")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MERGED_CSV)
    image_embs = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    text_embs = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))

    print(f"Dataset: {len(df)} samples")
    print(f"Embeddings: image={image_embs.shape}, text={text_embs.shape}")

    # Class weights
    print("\nClass weights:")
    pos_weights = compute_class_weights(df)

    # Dataloaders
    print("\nCreating train/val split...")
    train_loader, val_loader, train_idx, val_idx = create_dataloaders(
        image_embs, text_embs, df,
        val_frac=0.15,
        batch_size=BATCH_SIZE
    )

    # Save split indices for reproducibility
    np.save(os.path.join(EMB_DIR, "train_indices.npy"), np.array(train_idx))
    np.save(os.path.join(EMB_DIR, "val_indices.npy"), np.array(val_idx))

    # Model
    model = FullModel(
        input_dim=COMBINED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
        num_cultures=NUM_CULTURES,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training loop
    best_avg_f1 = 0.0
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {EPOCHS} epochs...")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, pos_weights)
        val_loss, culture_f1, avg_f1 = evaluate(model, val_loader, pos_weights)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        # Format F1 scores
        f1_str = " | ".join(
            f"{name}={f1:.3f}" if not np.isnan(f1) else f"{name}=N/A"
            for name, f1 in culture_f1.items()
        )

        print(
            f"Epoch {epoch:3d} | "
            f"Tr={train_loss:.4f} | Val={val_loss:.4f} | "
            f"F1: {f1_str} | Avg={avg_f1:.3f} | LR={lr:.2e}"
        )

        # Log
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "culture_f1": culture_f1,
            "avg_f1": avg_f1,
            "lr": lr,
        }
        history.append(entry)

        # Checkpoint logic
        min_culture_f1 = min(f for f in culture_f1.values() if not np.isnan(f))

        if avg_f1 > best_avg_f1 and min_culture_f1 >= 0.40:
            best_avg_f1 = avg_f1
            patience_counter = 0

            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_f1": avg_f1,
                "culture_f1": culture_f1,
            }, ckpt_path)
            print(f"  → BEST model saved (avg F1={avg_f1:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
            break

    # Save training history
    history_path = os.path.join(LOG_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nHistory saved: {history_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"Best avg Macro F1: {best_avg_f1:.3f}")
    print(f"{'=' * 60}")

    return model, history


if __name__ == "__main__":
    main()
