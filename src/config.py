"""
Central configuration for the CC-MMD pipeline.
All paths, hyperparameters, and model choices in one place.
"""
import os

# ============================================================
# PATHS
# ============================================================
ROOT_DIR = "E:/pep"
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_BASE = os.path.join(DATA_DIR, "image")  # E:/pep/data/image/{culture}/

TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TRAIN_CSVS = {
    "tamil": os.path.join(TRAIN_DIR, "tamil", "train_clean.csv"),
    "malayalam": os.path.join(TRAIN_DIR, "malayalam", "train_clean.csv"),
    "chinese": os.path.join(TRAIN_DIR, "chinese", "train_clean.csv"),
    "western": os.path.join(TRAIN_DIR, "western", "train_clean.csv"),
}

MERGED_CSV = os.path.join(ROOT_DIR, "train_merged.csv")
DEV_CSV = os.path.join(ROOT_DIR, "dev", "dev_final.csv")

# Embedding cache
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
DEV_EMB_DIR = os.path.join(DATA_DIR, "embeddings_dev")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(DEV_EMB_DIR, exist_ok=True)

# Results
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
SUBMISSION_DIR = os.path.join(RESULTS_DIR, "submissions")
for d in [RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR, SUBMISSION_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# IMAGE FOLDERS — images stored per-culture to avoid ID collision
# ============================================================
IMAGE_DIRS = {
    "tamil": os.path.join(IMAGE_BASE, "tamil"),
    "malayalam": os.path.join(IMAGE_BASE, "malayalam"),
    "chinese": os.path.join(IMAGE_BASE, "chinese"),
    "western": os.path.join(IMAGE_BASE, "western"),
}

# ============================================================
# MODEL CHOICES (frozen, CPU-compatible)
# ============================================================
CLIP_MODEL = "openai/clip-vit-base-patch32"  # 512-d image embeddings
XLM_MODEL = "xlm-roberta-base"               # 768-d text embeddings

CLIP_DIM = 512
TEXT_DIM = 768
COMBINED_DIM = CLIP_DIM + TEXT_DIM  # 1280

# ============================================================
# ARCHITECTURE
# ============================================================
HIDDEN_DIM = 512
LATENT_DIM = 256
DROPOUT = 0.3
NUM_CULTURES = 3  # india, western, china

# Culture names — order matters, keep consistent everywhere
CULTURE_NAMES = ["india", "western", "china"]
LABEL_COLS = ["india_label", "western_label", "china_label"]

# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOP_PATIENCE = 8

# Multitask loss weights
AUX_WEIGHT_TARGET = 0.3
AUX_WEIGHT_HUMOR = 0.2
AUX_WEIGHT_SENTIMENT = 0.2

# ============================================================
# PROTOTYPES
# ============================================================
NUM_PROTOTYPES_PER_CLASS = 6  # per culture per class (pos/neg)

# ============================================================
# CALIBRATION
# ============================================================
THRESHOLD_RANGE = (0.30, 0.70)
THRESHOLD_STEP = 0.01
DEFAULT_THRESHOLD = 0.5
