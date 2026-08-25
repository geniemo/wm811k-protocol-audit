"""Fixed preprocessing: nearest-neighbour resize and one-hot encoding.

Wafer-map pixels are discrete (0 = no die, 1 = pass, 2 = fail). Nearest-neighbour
keeps that vocabulary; bilinear/bicubic would invent values that do not exist.
"""
import numpy as np
import torch

from .constants import IMG_SIZE


def nearest_index(src_len: int, out_len: int) -> np.ndarray:
    """Centre-aligned nearest-neighbour source index for each output position."""
    if src_len <= 0 or out_len <= 0:
        raise ValueError("lengths must be positive")
    idx = np.floor((np.arange(out_len) + 0.5) * src_len / out_len).astype(np.int64)
    return np.clip(idx, 0, src_len - 1)


def resize_nearest(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D wafer map, got shape {a.shape}")
    rows = nearest_index(a.shape[0], size)
    cols = nearest_index(a.shape[1], size)
    return a[rows][:, cols].astype(np.uint8, copy=False)


def one_hot_maps(x: torch.Tensor, n_values: int = 3) -> torch.Tensor:
    """uint8/long [B,H,W] with values in {0,1,2} -> float32 [B,3,H,W]."""
    oh = torch.nn.functional.one_hot(x.long(), num_classes=n_values)
    return oh.permute(0, 3, 1, 2).float()
