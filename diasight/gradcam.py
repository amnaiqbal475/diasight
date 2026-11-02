"""Grad-CAM heatmap overlay on the retina image."""
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .model import target_layer


def heatmap(model, cfg: dict, result) -> np.ndarray:
    """Returns an RGB uint8 overlay the same size as result.display_img."""
    if cfg["head"] == "classification":
        targets = [ClassifierOutputTarget(result.grade)]
    else:
        targets = [lambda out: out[0]]  # explain the regression score
    with GradCAM(model=model, target_layers=[target_layer(model)]) as cam:
        mask = cam(input_tensor=result.tensor, targets=targets)[0]
    base = result.display_img.astype(np.float32) / 255.0
    return show_cam_on_image(base, mask, use_rgb=True, image_weight=0.6)
