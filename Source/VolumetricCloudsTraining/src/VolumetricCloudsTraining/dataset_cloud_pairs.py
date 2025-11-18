import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .pfm_io import read_pfm


# ================================================================
#                    DATASET: CLOUD PAIR PFM FILES
# ================================================================


@dataclass
class CloudPairDatasetConfig:
    root_dir: str
    crop_size: Optional[int] = 256
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    upsample_mode: str = "bicubic"  # 'bilinear'|'bicubic'
    align_corners: bool = False
    limit_pairs: Optional[int] = None  # For quick sanity checks


class CloudPairDataset(Dataset):
    """
    Pairs of low/high .pfm images:
      - low  resolution: 320 x 180
      - high resolution: 1280 x 720
    Input to the model is the low image upsampled to high resolution (by factor 4).
    The target is the high-resolution image.
    """

    def __init__(self, config: CloudPairDatasetConfig):
        super().__init__()
        self._config = config
        self._pairs: List[Tuple[str, str]] = self._discover_pairs(config.root_dir, config.limit_pairs)

    # ----------------- Getters / Setters -----------------
    @property
    def config(self) -> CloudPairDatasetConfig:
        return self._config

    @config.setter
    def config(self, new_config: CloudPairDatasetConfig) -> None:
        self._config = new_config

    @property
    def num_pairs(self) -> int:
        return len(self._pairs)

    @property
    def crop_size(self) -> Optional[int]:
        return self._config.crop_size

    @crop_size.setter
    def crop_size(self, value: Optional[int]) -> None:
        self._config.crop_size = value

    # ----------------- Dataset API -----------------
    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int):
        low_path, high_path = self._pairs[index]

        # Array layout notes:
        # (H, W, 3) = (image height in pixels, image width in pixels, 3 color channels RGB), dtype=float32
        high_np = read_pfm(high_path)  # (H, W, 3) float32
        low_np = read_pfm(low_path)    # (h, w, 3) float32

        # Convert grayscale to 3ch if needed
        if high_np.ndim == 2:
            high_np = np.repeat(high_np[..., None], 3, axis=2)
        if low_np.ndim == 2:
            low_np = np.repeat(low_np[..., None], 3, axis=2)

        # Clamp values to a sane range
        high_np = np.clip(high_np, self._config.clamp_min, self._config.clamp_max)
        low_np = np.clip(low_np, self._config.clamp_min, self._config.clamp_max)

        # To torch (C, H, W)
        # Convert HWC (NumPy) -> CHW (PyTorch). Channel-first is the standard for torch layers.
        # permute(2,0,1): move channels axis from last to first. contiguous(): ensure memory layout is contiguous.
        high = torch.from_numpy(high_np).permute(2, 0, 1).contiguous()  # [3, H, W]
        low = torch.from_numpy(low_np).permute(2, 0, 1).contiguous()    # [3, h, w]

        # Upsample low to high resolution using chosen interpolation
        _, target_h, target_w = high.shape
        low_up = F.interpolate(
            low.unsqueeze(0),
            size=(target_h, target_w),
            mode=self._config.upsample_mode,
            align_corners=self._config.align_corners if self._config.upsample_mode in ("bilinear", "bicubic") else None,
        ).squeeze(0)

        # Why cropping?
        # - Data augmentation: exposes the model to diverse local patterns, reducing overfitting.
        # - Memory/performance: smaller 256x256 patches fit GPU RAM and allow larger batches.
        # - More updates: multiple distinct patches per image improve gradient signal.
        # - Translation invariance: encourages learning features independent of absolute position.
        if self._config.crop_size is not None:
            cs = self._config.crop_size
            if cs <= 0 or cs > min(target_h, target_w):
                raise ValueError(f"Invalid crop_size={cs}; must be in (0, min(H, W)] = (0, {min(target_h, target_w)}].")
            top = torch.randint(0, target_h - cs + 1, (1,)).item()
            left = torch.randint(0, target_w - cs + 1, (1,)).item()
            high = high[:, top:top + cs, left:left + cs]
            low_up = low_up[:, top:top + cs, left:left + cs]

        return {
            "low_up": low_up,  # upsampled low-res input at high resolution
            "high": high,      # target high-res image
            "low_path": low_path,
            "high_path": high_path,
        }

    # ----------------- Helpers -----------------
    @staticmethod
    def _discover_pairs(root_dir: str, limit: Optional[int]) -> List[Tuple[str, str]]:
        """
        Find files named pair_XXXXXX_low.pfm and pair_XXXXXX_high.pfm in root_dir and pair them by index.
        """
        candidates = []
        for name in os.listdir(root_dir):
            if not name.endswith("_low.pfm"):
                continue
            prefix = name[:-8]  # strip "_low.pfm"
            low_path = os.path.join(root_dir, f"{prefix}_low.pfm")
            high_path = os.path.join(root_dir, f"{prefix}_high.pfm")
            if os.path.isfile(low_path) and os.path.isfile(high_path):
                candidates.append((low_path, high_path))

        # Sort deterministically by prefix (pair index lexicographically works here)
        candidates.sort(key=lambda p: p[0])
        if limit is not None:
            candidates = candidates[: max(0, int(limit))]

        if len(candidates) == 0:
            raise FileNotFoundError(f"No PFM pairs found in {root_dir}")
        return candidates


