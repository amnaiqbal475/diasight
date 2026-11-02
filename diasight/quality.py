"""Resolution, blur and brightness checks. Thresholds are calibrated on dataset images."""
import cv2
import numpy as np

from .preprocess import crop_black_border

MIN_SIDE = 150      # below this the retina is too small to show lesions at all
MEASURE_SIDE = 512  # big photos are shrunk to this before measuring; small ones are never enlarged


def measure(img: np.ndarray) -> dict:
    cropped = crop_black_border(img)
    h, w = cropped.shape[:2]
    scale = min(1.0, MEASURE_SIDE / max(h, w))
    if scale < 1.0:
        cropped = cv2.resize(cropped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    fundus = gray[gray > 7]  # ignore the black surround
    return {
        "min_side": int(min(h, w)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(fundus.mean()) if fundus.size else 0.0,
    }


def check(img: np.ndarray, thresholds: dict):
    """Returns (ok, message, measurements)."""
    m = measure(img)
    if m["min_side"] < MIN_SIDE:
        return False, "Please retake: image resolution too low.", m
    if m["brightness"] < thresholds["min_brightness"]:
        return False, "Please retake: image too dark.", m
    if m["brightness"] > thresholds["max_brightness"]:
        return False, "Please retake: image too bright / washed out.", m
    if m["sharpness"] < thresholds["min_sharpness"]:
        return False, "Please retake: image too blurry.", m
    return True, "Good", m


def calibrate(images, low_pct: float = 2, high_pct: float = 98) -> dict:
    """Run on a few hundred training images; the result goes into config.json."""
    ms = [measure(i) for i in images]
    s = np.array([m["sharpness"] for m in ms]); b = np.array([m["brightness"] for m in ms])
    return {"min_sharpness": float(np.percentile(s, low_pct)),
            "min_brightness": float(np.percentile(b, low_pct)),
            "max_brightness": float(np.percentile(b, high_pct))}