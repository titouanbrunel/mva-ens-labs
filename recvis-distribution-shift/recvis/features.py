from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recvis import config, data


def cache_path(name, split, cache_dir=None):
    return Path(cache_dir or config.CACHE_DIR) / f"{name}__{split}_flip.npy"


def text_head_path(name, cache_dir=None):
    return Path(cache_dir or config.CACHE_DIR) / f"{name}__text_head.pt"


@torch.inference_mode()
def encode_images(backbone, images_u8, batch_size=256):
    """L2-normalised embeddings, averaged over the image and its mirror."""
    chunks = []
    for batch in data.iter_batches(images_u8, batch_size):
        pixels = data.to_model_input(
            batch, backbone.image_size, backbone.mean, backbone.std, backbone.device
        ).to(backbone.mean.dtype)
        embedding = backbone.encode(pixels)
        embedding = F.normalize(embedding + backbone.encode(pixels.flip(-1)), dim=-1)
        chunks.append(embedding.cpu())
    return torch.cat(chunks)


def get_features(backbone, split, batch_size=256, cache_dir=None):
    """(features, labels, ids). Cached as float16, returned as float32."""
    name = backbone if isinstance(backbone, str) else backbone.name
    path = cache_path(name, split, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    loaded = data.load_split(split)

    if path.is_file():
        features = torch.from_numpy(np.load(path)).float()
        if len(features) != len(loaded):
            raise ValueError(f"{path}: {len(features)} rows for {len(loaded)} images.")
    elif isinstance(backbone, str):
        raise FileNotFoundError(f"{path} missing and no encoder given.")
    else:
        features = encode_images(backbone, loaded.images, batch_size)
        np.save(path, features.half().numpy())

    return F.normalize(features, dim=-1), loaded.labels, loaded.ids


def get_text_head(backbone, cache_dir=None):
    name = backbone if isinstance(backbone, str) else backbone.name
    path = text_head_path(name, cache_dir)
    if path.is_file():
        return torch.load(path, map_location="cpu")
    if isinstance(backbone, str):
        raise FileNotFoundError(f"{path} missing and no encoder given.")
    head = backbone.zero_shot_head().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head, path)
    return head
