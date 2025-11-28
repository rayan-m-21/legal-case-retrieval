"""
Central configuration for paths and constants.

This module auto-detects the project root and provides flexible path resolution
so the code works regardless of where it's run from (notebook, script, etc.).
"""

from pathlib import Path

def _find_project_root() -> Path:
    """
    Walk up from this file until we find a directory containing 'data/' and 'src/'.
    This makes the code portable across different execution contexts.
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent
    # Fallback: assume we're in src/
    return current.parent

PROJECT_ROOT = _find_project_root()

# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

# Raw data
TRAIN_FILES_DIR = DATA_DIR / "train_files"
TEST_FILES_DIR = DATA_DIR / "test_files"

# Cleaned/preprocessed data
CLEAN_TRAIN_FILES_DIR = DATA_DIR / "clean_train_files"
CLEAN_TEST_FILES_DIR = DATA_DIR / "clean_test_files"

# Labels
TRAIN_LABELS_FILE = DATA_DIR / "train_labels.json"
TEST_LABELS_FILE = DATA_DIR / "test_labels.json"

# ─────────────────────────────────────────────────────────────────────────────
# Output paths (rankings, embeddings, results)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"

EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
RANKINGS_DIR = OUTPUT_DIR / "rankings"
RESULTS_DIR = OUTPUT_DIR / "results"

# ─────────────────────────────────────────────────────────────────────────────
# Model configuration
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "BERT-base": "bert-base-uncased",
    "LegalBERT": "nlpaueb/legal-bert-base-uncased",
}

MAX_CHUNK_SIZE = {
    "BERT-base": 512,
    "LegalBERT": 512,
}

POOLING_STRATEGIES = {
    "cls_only": "Use [CLS] token",
    "tokens": "Mean of non-special tokens",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def get_labels_file(train: bool = True) -> Path:
    """Return the appropriate labels file path."""
    return TRAIN_LABELS_FILE if train else TEST_LABELS_FILE

def get_clean_files_dir(train: bool = True) -> Path:
    """Return the appropriate cleaned files directory."""
    return CLEAN_TRAIN_FILES_DIR if train else CLEAN_TEST_FILES_DIR

def get_raw_files_dir(train: bool = True) -> Path:
    """Return the appropriate raw files directory."""
    return TRAIN_FILES_DIR if train else TEST_FILES_DIR

def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    for d in [OUTPUT_DIR, EMBEDDINGS_DIR, RANKINGS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
