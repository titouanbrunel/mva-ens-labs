from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recvis import config


@dataclass(frozen=True)
class Split:
    name: str
    images: torch.Tensor          # uint8 (N, 32, 32, 3)
    labels: torch.Tensor | None   # int64 (N,), None for test
    ids: torch.Tensor | None      # int64 (N,), test only

    def __len__(self):
        return len(self.images)


def load_split(name, data_dir=None):
    path = Path(data_dir or config.DATA_DIR) / config.SPLIT_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; set RECVIS_DATA_DIR.")

    with np.load(path) as archive:
        images = archive["images"]
        labels = archive["labels"].astype(np.int64) if "labels" in archive else None
        ids = archive["ids"].astype(np.int64) if "ids" in archive else None
        if "class_names" in archive:
            stored = tuple(str(x) for x in archive["class_names"])
            if stored != config.CLASS_NAMES:
                raise ValueError(f"{path}: class order {stored} != {config.CLASS_NAMES}.")

    return Split(
        name=name,
        images=torch.from_numpy(np.ascontiguousarray(images)),
        labels=None if labels is None else torch.from_numpy(labels),
        ids=None if ids is None else torch.from_numpy(ids),
    )


def to_model_input(images_u8, size, mean, std, device):
    """uint8 (B, 32, 32, 3) -> normalised float (B, 3, size, size)."""
    x = images_u8.to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    if size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
        x = x.clamp_(0.0, 1.0)  # bicubic overshoots at sharp edges
    return (x - mean) / std


def iter_batches(images_u8, batch_size):
    for start in range(0, len(images_u8), batch_size):
        yield images_u8[start : start + batch_size]
