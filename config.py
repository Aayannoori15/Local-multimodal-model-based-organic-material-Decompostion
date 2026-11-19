"""Shared settings for the Organic Waste Decomposition Stage Detector."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Folder names inside dataset/ (sorted order = class index)
CLASSES = [
    "1_fresh_waste",
    "2_early_decomposition",
    "3_active_decomposition",
    "4_mature_compost",
]

CLASS_LABELS = {
    "1_fresh_waste": "Stage 1 - Fresh Waste",
    "2_early_decomposition": "Stage 2 - Early Decomposition",
    "3_active_decomposition": "Stage 3 - Active Decomposition",
    "4_mature_compost": "Stage 4 - Mature Compost",
}

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e2b"
