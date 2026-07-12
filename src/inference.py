"""
Stage 7: Test-time inference and submission CSV generation.

For each test meme:
1. Load/extract CLIP + XLM-R embeddings
2. Forward through trained model (with cultural prototypes if available)
3. Apply per-culture calibrated thresholds
4. Format as Task B submission CSV

Output format:
    image_id,original_culture,irish_culture,chinese_culture
    654,1,0,0
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MERGED_CSV, EMB_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR,
    SUBMISSION_DIR, COMBINED_DIM, HIDDEN_DIM, LATENT_DIM, DROPOUT,
    NUM_CULTURES, CULTURE_NAMES, DEFAULT_THRESHOLD
)
from src.model import FullModel


def load_model():
    """Load the best available model (cultural or base)."""
    model = FullModel(
        input_dim=COMBINED_DIM, hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, dropout=DROPOUT, num_cultures=NUM_CULTURES,
    )

    cultural_ckpt = os.path.join(CHECKPOINT_DIR, "best_cultural.pt")
    base_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if os.path.exists(cultural_ckpt):
        ckpt = torch.load(cultural_ckpt, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        if ckpt.get("has_prototypes"):
            proto_dir = os.path.join(RESULTS_DIR, "prototypes")
            pos_p = [np.load(os.path.join(proto_dir, f"{n}_pos.npy")) for n in CULTURE_NAMES]
            neg_p = [np.load(os.path.join(proto_dir, f"{n}_neg.npy")) for n in CULTURE_NAMES]
            model.enable_prototypes(pos_p, neg_p)

        print(f"Loaded cultural model (F1={ckpt['avg_f1']:.3f})")
    else:
        ckpt = torch.load(base_ckpt, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded base model (F1={ckpt['avg_f1']:.3f})")

    model.eval()
    return model


def load_thresholds():
    """Load calibrated thresholds, or use defaults.
    Stage 8 (run_full.py) saves 'thresholds.json'; older runs may use
    'calibrated_thresholds.json'. We check both for compatibility.
    """
    for fname in ["thresholds.json", "calibrated_thresholds.json"]:
        cal_path = os.path.join(LOG_DIR, fname)
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                data = json.load(f)
            # data is either {"india":0.45,...} or {"thresholds":{...}}
            thresholds = data if isinstance(data, dict) and "india" in data \
                else data.get("thresholds", data)
            print(f"Loaded calibrated thresholds from {fname}: {thresholds}")
            return thresholds

    print(f"No calibrated thresholds found, using default {DEFAULT_THRESHOLD}")
    return {name: DEFAULT_THRESHOLD for name in CULTURE_NAMES}


def predict_from_embeddings(model, image_embs, text_embs, thresholds):
    """
    Run inference on pre-computed embeddings.

    Returns:
        preds: (N, 3) int array — binary predictions [india, western, china]
        probs: (N, 3) float array — sigmoid probabilities
    """
    combined = np.concatenate([image_embs, text_embs], axis=1).astype(np.float32)

    all_probs = []
    batch_size = 256

    with torch.no_grad():
        for start in tqdm(range(0, len(combined), batch_size), desc="Predicting"):
            batch = torch.from_numpy(combined[start:start + batch_size])
            logits, _ = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Apply per-culture thresholds
    threshold_array = np.array([
        thresholds.get("india", DEFAULT_THRESHOLD),
        thresholds.get("western", DEFAULT_THRESHOLD),
        thresholds.get("china", DEFAULT_THRESHOLD),
    ])

    preds = (all_probs > threshold_array).astype(int)
    return preds, all_probs


def format_submission_taskb(df, preds, output_path):
    """
    Format predictions as Task B CSV.

    Task B format:
        image_id,original_culture,irish_culture,chinese_culture
    """
    submission = pd.DataFrame({
        "image_id": df["image_id"].values,
        "original_culture": preds[:, 0],   # india
        "irish_culture": preds[:, 1],      # western
        "chinese_culture": preds[:, 2],    # china
    })

    submission.to_csv(output_path, index=False)
    return submission


def validate_submission(submission, expected_ids=None):
    """Validate submission CSV format."""
    errors = []

    required_cols = ["image_id", "original_culture", "irish_culture", "chinese_culture"]
    for col in required_cols:
        if col not in submission.columns:
            errors.append(f"Missing column: {col}")

    for col in ["original_culture", "irish_culture", "chinese_culture"]:
        if col in submission.columns:
            unique = set(submission[col].unique())
            if not unique.issubset({0, 1}):
                errors.append(f"{col} contains values other than 0/1: {unique}")

    if submission.duplicated(subset=["image_id"]).any():
        errors.append("Duplicate image_ids found")

    if expected_ids is not None:
        missing = set(expected_ids) - set(submission["image_id"])
        if missing:
            errors.append(f"{len(missing)} image_ids missing from submission")

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        return False

    print("✓ Submission format validated OK")
    print(f"  Rows: {len(submission)}")
    for col in ["original_culture", "irish_culture", "chinese_culture"]:
        dist = submission[col].value_counts()
        print(f"  {col}: 1={dist.get(1, 0)}, 0={dist.get(0, 0)}")
    return True


def predict_on_train(output_name="train_predictions"):
    """Run inference on training data for debugging and analysis."""
    print("=" * 60)
    print("Inference on training data (for analysis)")
    print("=" * 60)

    df = pd.read_csv(MERGED_CSV)
    image_embs = np.load(os.path.join(EMB_DIR, "clip_image.npy"))
    text_embs = np.load(os.path.join(EMB_DIR, "xlmr_text.npy"))

    model = load_model()
    thresholds = load_thresholds()

    preds, probs = predict_from_embeddings(model, image_embs, text_embs, thresholds)

    output_path = os.path.join(SUBMISSION_DIR, f"{output_name}.csv")
    submission = format_submission_taskb(df, preds, output_path)

    validate_submission(submission)
    print(f"\nSaved: {output_path}")

    return submission


def predict_on_test(test_csv, test_image_embs_path, test_text_embs_path,
                    output_name="test_submission"):
    """
    Run inference on test data and generate submission CSV.

    Args:
        test_csv: path to test CSV (must have image_id column)
        test_image_embs_path: path to cached CLIP embeddings for test images
        test_text_embs_path: path to cached XLM-R embeddings for test text
    """
    print("=" * 60)
    print("Test inference → Task B submission")
    print("=" * 60)

    test_df = pd.read_csv(test_csv)
    image_embs = np.load(test_image_embs_path)
    text_embs = np.load(test_text_embs_path)

    model = load_model()
    thresholds = load_thresholds()

    preds, probs = predict_from_embeddings(model, image_embs, text_embs, thresholds)

    output_path = os.path.join(SUBMISSION_DIR, f"{output_name}.csv")
    submission = format_submission_taskb(test_df, preds, output_path)

    validate_submission(submission, expected_ids=test_df["image_id"].tolist())
    print(f"\nSaved: {output_path}")

    return submission


if __name__ == "__main__":
    # Default: run on training data for analysis
    predict_on_train()
