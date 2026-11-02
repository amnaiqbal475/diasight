"""Image -> grade, refer decision, probabilities."""
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import GRADE_NAMES
from .preprocess import preprocess, to_tensor


@dataclass
class Result:
    grade: int
    grade_name: str
    refer: bool
    refer_score: float
    probs: list = field(default_factory=list)   # empty for the regression head
    display_img: np.ndarray = None
    model_img: np.ndarray = None
    tensor: torch.Tensor = None


def score_to_grade(score: float, thresholds) -> int:
    return int(np.digitize([score], thresholds)[0])


@torch.no_grad()
def predict(model, cfg: dict, img: np.ndarray) -> Result:
    display, model_img = preprocess(img, cfg["img_size"], cfg["ben_graham_sigma"])
    x = to_tensor(model_img, cfg["mean"], cfg["std"])
    out = model(x)
    if cfg["head"] == "classification":
        probs = torch.softmax(out, dim=1)[0].tolist()
        grade = int(np.argmax(probs))
        refer_score = float(sum(probs[2:]))          # P(grade >= 2)
    else:
        score = float(out[0, 0])
        probs, refer_score = [], score
        grade = score_to_grade(score, cfg["reg_thresholds"])
    refer = refer_score >= cfg["refer_threshold"]
    return Result(grade, GRADE_NAMES[grade], refer, refer_score, probs, display, model_img, x)
