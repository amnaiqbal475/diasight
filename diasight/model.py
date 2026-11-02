"""Build models from timm and load the trained checkpoint (or fall back to demo mode)."""
import json
from pathlib import Path

import torch
import timm

from .config import DEFAULT_CONFIG, MODELS_DIR


def build_model(name: str, head: str = "classification", pretrained: bool = True):
    num_outputs = 5 if head == "classification" else 1
    return timm.create_model(name, pretrained=pretrained, num_classes=num_outputs)


def load(models_dir: Path = MODELS_DIR):
    """Returns (model, cfg, is_demo).
    Real mode: models/best.pt + models/config.json written by the training notebook.
    Demo mode: no checkpoint yet -> untrained network, so the full app flow can be
    shown and tested. Its predictions are meaningless and the app says so."""
    cfg_path, ckpt_path = Path(models_dir) / "config.json", Path(models_dir) / "best.pt"
    if cfg_path.exists() and ckpt_path.exists():
        cfg = {**DEFAULT_CONFIG, **json.loads(cfg_path.read_text())}
        model = build_model(cfg["model_name"], cfg["head"], pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        is_demo = False
    else:
        cfg = dict(DEFAULT_CONFIG)
        torch.manual_seed(cfg.get("seed", 42))
        model = build_model(cfg["model_name"], cfg["head"], pretrained=False)
        is_demo = True
    model.eval()
    return model, cfg, is_demo


def target_layer(model):
    """Last convolutional block, used by Grad-CAM."""
    if hasattr(model, "conv_head"):      # EfficientNet (timm)
        return model.conv_head
    if hasattr(model, "layer4"):         # ResNet
        return model.layer4[-1]
    raise ValueError("Unknown architecture: set the Grad-CAM layer in model.target_layer")
