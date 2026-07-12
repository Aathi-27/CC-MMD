"""
Stage 6: Per-culture threshold calibration.

Sweeps thresholds from 0.30 to 0.70 independently per culture to maximize Macro F1.
This is FREE performance — no model change, just optimal decision boundary.

Expected gain: 3-6 F1 points vs fixed 0.5.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MERGED_CSV, EMB_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR,
    COMBINED_DIM, HIDDEN_DIM, LATENT_DIM, DROPOUT, NUM_CULTURES,
    BATCH_SIZE, LABEL_COLS, CULTURE_NAMES,
    THRESHOLD_RANGE, THRESHOLD_STEP
)
from src.dataset import MemeEmbeddingDataset
from src.model import FullModel


def get_val_predictions(model, image_embs, text_embs, df, val_indices):
    """Get sigmoid probabilities for validation set."""
    val_embs = np.concatenate([image_embs[val_indices], text_embs[val_indices]], axis=1)
    val_embs = torch.from_numpy(val_embs.astype(np.float32))

    model.eval()
    all_probs = []
    batch_size = 256

    with torch.no_grad():
        for start in range(0, len(val_embs), batch_size):
            batch = val_embs[start:start + batch_size]
            logits, _ = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0).numpy()
    val_df = df.iloc[val_indices]

    return all_probs, val_df


def calibrate_thresholds(probs, val_df):
    """
    Find optimal threshold per culture by sweeping and maximizing Macro F1.

    Returns:
        thresholds: dict {culture_name: optimal_threshold}
        f1_scores: dict {culture_name: best_macro_f1}
    """
    thresholds = {}
    f1_scores = {}

    lo, hi = THRESHOLD_RANGE
    sweep = np.arange(lo, hi + THRESHOLD_STEP, THRESHOLD_STEP)

    for i, (col, name) in enumerate(zip(LABEL_COLS, CULTURE_NAMES)):
        valid_mask = val_df[col].notna().values
        if valid_mask.sum() == 0:
            print(f"  {name}: No valid labels in val set — using default 0.5")
            thresholds[name] = 0.5
            f1_scores[name] = float("nan")
            continue

        y_true = val_df[col].values[valid_mask]
        culture_probs = probs[valid_mask, i]

        best_t = 0.5
        best_f1 = 0.0

        for t in sweep:
            y_pred = (culture_probs > t).astype(float)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[name] = round(best_t, 3)
        f1_scores[name] = round(best_f1, 4)
        print(f"  {name}: threshold={best_t:.3f}, Macro F1={best_f1:.4f}")

    return thresholds, f1_scores


def main():
    print("=" * 60)
    print("STAGE 6: Per-Culture Threshold Calibration")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MERGED_CSV)
    image_embs = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    text_embs = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))
    val_indices = np.load(os.path.join(EMB_DIR, "val_indices.npy"))

    # Try cultural model first, fallback to base
    cultural_ckpt = os.path.join(CHECKPOINT_DIR, "best_model_cultural.pt")
    base_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    model = FullModel(
        input_dim=COMBINED_DIM, hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, dropout=DROPOUT, num_cultures=NUM_CULTURES,
    )

    if os.path.exists(cultural_ckpt):
        ckpt = torch.load(cultural_ckpt, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        if ckpt.get("has_prototypes"):
            proto_dir = os.path.join(RESULTS_DIR, "prototypes")
            pos_p = [np.load(os.path.join(proto_dir, f"{n}_pos.npy")) for n in CULTURE_NAMES]
            neg_p = [np.load(os.path.join(proto_dir, f"{n}_neg.npy")) for n in CULTURE_NAMES]
            model.enable_prototypes(pos_p, neg_p)

        print(f"Using cultural model (F1={ckpt['avg_f1']:.3f})")
    else:
        ckpt = torch.load(base_ckpt, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Using base model (F1={ckpt['avg_f1']:.3f})")

    # Get predictions
    print("\nGetting validation predictions...")
    probs, val_df = get_val_predictions(model, image_embs, text_embs, df, val_indices)

    # Calibrate
    print("\nCalibrating thresholds:")
    thresholds, f1_scores = calibrate_thresholds(probs, val_df.reset_index(drop=True))

    # Compare with default 0.5
    print("\nComparison with default threshold (0.5):")
    for i, (col, name) in enumerate(zip(LABEL_COLS, CULTURE_NAMES)):
        valid_mask = val_df[col].notna().values
        if valid_mask.sum() == 0:
            continue
        y_true = val_df[col].values[valid_mask]
        y_pred_default = (probs[valid_mask, i] > 0.5).astype(float)
        f1_default = f1_score(y_true, y_pred_default, average="macro", zero_division=0)
        gain = f1_scores[name] - f1_default
        print(f"  {name}: default={f1_default:.4f} → calibrated={f1_scores[name]:.4f} (gain={gain:+.4f})")

    # Save
    cal_path = os.path.join(LOG_DIR, "calibrated_thresholds.json")
    with open(cal_path, "w") as f:
        json.dump({"thresholds": thresholds, "f1_scores": f1_scores}, f, indent=2)
    print(f"\nSaved: {cal_path}")

    valid_f1s = [v for v in f1_scores.values() if not np.isnan(v)]
    avg_calibrated = np.mean(valid_f1s) if valid_f1s else 0.0
    print(f"\nFinal calibrated avg Macro F1: {avg_calibrated:.4f}")


if __name__ == "__main__":
    main()
