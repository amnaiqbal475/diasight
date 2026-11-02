"""PyTorch Dataset over cached preprocessed images + training augmentations."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def train_augs(size: int):
    import albumentations as A
    return A.Compose([
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.HorizontalFlip(), A.VerticalFlip(),
        A.RandomBrightnessContrast(0.2, 0.2),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), p=0.5),
        A.Resize(size, size),
    ])


class RetinaDataset(Dataset):
    """df needs columns: path (cached preprocessed PNG), diagnosis (0-4)."""

    def __init__(self, df, cfg, augs=None, head="classification"):
        self.paths, self.labels = df["path"].tolist(), df["diagnosis"].tolist()
        self.cfg, self.augs, self.head = cfg, augs, head
        self.mean = np.array(cfg["mean"], np.float32); self.std = np.array(cfg["std"], np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        if self.augs is not None:
            img = self.augs(image=img)["image"]
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1))
        y = self.labels[i]
        y = torch.tensor(y, dtype=torch.long if self.head == "classification" else torch.float32)
        return x, y
