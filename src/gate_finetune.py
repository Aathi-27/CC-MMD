"""
Stage 5b: Fine-tune the cultural gating layer.

After prototypes are built, attach them to the model and train
ONLY the gating + refinement parameters. The core trunk and heads are frozen.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MERGED_CSV, EMB_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    COMBINED_DIM, HIDDEN_DIM, LATENT_DIM, DROPOUT, NUM_CULTURES,
    BATCH_SIZE, LABEL_COLS, CULTURE_NAMES
)
from src.dataset import create_dataloaders
from src.model import FullModel
from src.trainer import masked_bce_loss, evaluate, compute_class_weights


def main():
    print("=" * 60)
    print("STAGE 5b: Fine-tune Cultural Gating")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MERGED_CSV)
    image_embs = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    text_embs = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))

    # Load model
    model = FullModel(
        input_dim=COMBINED_DIM, hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, dropout=DROPOUT, num_cultures=NUM_CULTURES,
    )
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded base model from epoch {ckpt['epoch']}")

    # Load prototypes
    proto_dir = os.path.join(RESULTS_DIR, "prototypes")
    pos_prototypes = [np.load(os.path.join(proto_dir, f"{n}_pos.npy")) for n in CULTURE_NAMES]
    neg_prototypes = [np.load(os.path.join(proto_dir, f"{n}_neg.npy")) for n in CULTURE_NAMES]

    # Attach prototypes
    model.enable_prototypes(pos_prototypes, neg_prototypes)
    print("Cultural prototypes attached")

    # Freeze core, train only gating
    for param in model.core.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (gates only): {trainable:,}")

    # Dataloaders (same split)
    train_loader, val_loader, _, _ = create_dataloaders(
        image_embs, text_embs, df, val_frac=0.15, batch_size=BATCH_SIZE
    )

    pos_weights = compute_class_weights(df)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=1e-4
    )

    best_f1 = 0.0
    print("\nFine-tuning gates (15 epochs)...")
    print("-" * 80)

    for epoch in range(1, 16):
        model.train()
        total_loss = 0
        n_batches = 0

        for emb, labels, mask, culture_ids in train_loader:
            optimizer.zero_grad()
            logits, _ = model(emb)
            loss = masked_bce_loss(logits, labels, mask, pos_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_loss, culture_f1, avg_f1 = evaluate(model, val_loader, pos_weights)
        train_loss = total_loss / max(n_batches, 1)

        f1_str = " | ".join(
            f"{n}={f:.3f}" if not np.isnan(f) else f"{n}=N/A"
            for n, f in culture_f1.items()
        )
        print(f"Epoch {epoch:2d} | Tr={train_loss:.4f} | Val={val_loss:.4f} | F1: {f1_str} | Avg={avg_f1:.3f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "avg_f1": avg_f1,
                "culture_f1": culture_f1,
                "has_prototypes": True,
            }, os.path.join(CHECKPOINT_DIR, "best_model_cultural.pt"))
            print(f"  → BEST cultural model saved (avg F1={avg_f1:.3f})")

    print(f"\nBest gated avg F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()
