"""Same preprocessing for training, validation, IDRiD and the app."""
import cv2
import numpy as np
from PIL import Image


def load_rgb(source) -> np.ndarray:
    """Path, file-like object or PIL image -> RGB uint8 array."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    return np.array(img.convert("RGB"))


def crop_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > tol)
    if len(ys) == 0:  # completely dark image: leave it for the quality check
        return img
    return img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def circular_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * 0.95), 1, -1)
    return img * mask[..., None]


def ben_graham(img: np.ndarray, sigma: float = 10) -> np.ndarray:
    """image = 4*image - 4*GaussianBlur(image) + 128, clipped to 0..255."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def preprocess(img: np.ndarray, size: int = 384, sigma: float = 10):
    """Returns (display_img, model_img), both RGB uint8 size x size.
    display_img = cropped + resized (for showing to people and the Grad-CAM overlay)
    model_img   = display_img + Ben Graham + circular mask (what the network sees)"""
    cropped = crop_black_border(img)
    display = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    model_img = circular_mask(ben_graham(display, sigma))
    return circular_mask(display), model_img


def to_tensor(img: np.ndarray, mean, std):
    import torch
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
