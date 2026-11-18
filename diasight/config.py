"""Every path and default setting lives here, so training and the app never drift apart."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "app" / "assets"
SAMPLES_DIR = ROOT / "app" / "samples"
SEED = 42

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
GRADE_COLOURS = ["green", "green", "amber", "red", "red"]

# Used when models/config.json is missing (demo mode). Training writes the real one.
DEFAULT_CONFIG = {
    "model_name": "efficientnet_b0",
    "img_size": 224,
    "head": "classification",          # or "regression"
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "ben_graham_sigma": 10,
    "reg_thresholds": [0.5, 1.5, 2.5, 3.5],
    # refer if refer_score >= refer_threshold
    # classification: refer_score = P(grade >= 2); regression: refer_score = raw score
    "refer_threshold": 0.5,
    # Demo-mode fallback only. scripts/train.py writes calibrated values (2nd percentile of dataset
    # photos) into models/config.json; the current model uses sharpness 10, brightness 41.
    "quality": {"min_sharpness": 10.0, "min_brightness": 41.0, "max_brightness": 230.0},
}
