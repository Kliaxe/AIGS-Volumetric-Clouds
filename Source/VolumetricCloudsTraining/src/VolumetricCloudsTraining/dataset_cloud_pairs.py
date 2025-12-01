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

    # Optional auxiliary data channels extracted from *_low_data.pfm / *_high_data.pfm
    # These are concatenated to the low-resolution RGB input before upsampling.
    use_view_transmittance: bool = True
    use_light_transmittance: bool = True
    use_linear_depth: bool = True
    # Optional RGB normal channels extracted from *_low_normals.pfm. When enabled,
    # these are concatenated after any scalar auxiliary channels.
    use_normals: bool = False

    # Depth normalization scale in world units. The raw linear depth stored in the PFM
    # (blue channel of the data image) is divided by this value and clamped to [0, 1]
    # before being fed to the network.
    depth_normalization_max: float = 40000.0


def compute_dataset_index_splits(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic split of [0, num_samples) into train/val/test indices.

    The split is controlled by explicit fractions and a seed so that
    training and inference can reproduce the exact same partition of
    the underlying dataset without duplicating files on disk.
    """
    if num_samples <= 0:
        return [], [], []

    if train_fraction < 0.0 or val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("train_fraction, val_fraction and test_fraction must be non‑negative.")

    total_fraction = train_fraction + val_fraction + test_fraction
    if total_fraction <= 0.0:
        raise ValueError("At least one of train_fraction, val_fraction or test_fraction must be > 0.")

    # Normalise fractions if they do not sum exactly to 1.0. This makes the
    # behaviour robust to small user errors while preserving relative ratios.
    train_f = train_fraction / total_fraction
    val_f = val_fraction / total_fraction
    test_f = test_fraction / total_fraction

    # Derive counts. We compute val/test directly and assign the remaining
    # samples to train so that all samples are used.
    val_count = int(num_samples * val_f)
    test_count = int(num_samples * test_f)
    train_count = max(0, num_samples - val_count - test_count)

    # Guard against degenerate cases where rounding leads to zero‑sized splits
    # even though the corresponding fraction is non‑zero.
    if train_count == 0 and train_f > 0.0:
        train_count = 1
    if val_count == 0 and val_f > 0.0 and num_samples - train_count > 1:
        val_count = 1
    if test_count == 0 and test_f > 0.0 and num_samples - train_count - val_count > 0:
        test_count = 1

    # Final safety clamp to ensure counts sum to num_samples.
    overflow = (train_count + val_count + test_count) - num_samples
    if overflow > 0:
        # Prefer to reduce test, then val, leaving train as large as possible.
        reduce_test = min(overflow, test_count)
        test_count -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            reduce_val = min(overflow, val_count)
            val_count -= reduce_val
            overflow -= reduce_val
        if overflow > 0:
            train_count = max(0, train_count - overflow)

    rng = np.random.default_rng(split_seed)
    all_indices = np.arange(num_samples)
    rng.shuffle(all_indices)

    train_end = train_count
    val_end = train_end + val_count

    train_indices = all_indices[:train_end].tolist()
    val_indices = all_indices[train_end:val_end].tolist()
    test_indices = all_indices[val_end:].tolist()

    return train_indices, val_indices, test_indices


class CloudPairDataset(Dataset):
    """
    Pairs of low/high .pfm images:
      - low  resolution colour:   pair_XXXXXX_low.pfm
      - high resolution colour:   pair_XXXXXX_high.pfm
      - optional low data image:  pair_XXXXXX_low_data.pfm
      - optional high data image: pair_XXXXXX_high_data.pfm (currently unused)

    Input to the model is the low image upsampled to high resolution (by factor 4),
    optionally concatenated with upsampled auxiliary channels derived from the low
    data image (view transmittance, light transmittance, linear depth).
    The target is the high-resolution colour image.
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

        # Clamp colour values to a sane range
        high_np = np.clip(high_np, self._config.clamp_min, self._config.clamp_max)
        low_np = np.clip(low_np, self._config.clamp_min, self._config.clamp_max)

        # Optional auxiliary data image at low resolution ------------------------------------------
        # Expected layout in the renderer:
        #   R: view transmittance to camera (T_view in [0, 1])
        #   G: light-space transmittance / visibility (T_light in [0, 1])
        #   B: linear depth to effective scattering point (t_depth in world units)
        #   A: reserved
        low_data_path = low_path.replace("_low.pfm", "_low_data.pfm")
        low_data_np: Optional[np.ndarray]
        if os.path.isfile(low_data_path):
            low_data_np = read_pfm(low_data_path)
        else:
            low_data_np = None

        # Build a physically reasonable default data image if none is present so that the channel
        # layout remains consistent even for legacy captures.
        if low_data_np is None:
            h_low, w_low, _ = low_np.shape
            # Default to fully transparent / unoccluded with zero depth.
            # R=T_view=1, G=T_light=1, B=depth=0
            low_data_np = np.zeros((h_low, w_low, 3), dtype=np.float32)
            low_data_np[..., 0] = 1.0
            low_data_np[..., 1] = 1.0
        elif low_data_np.ndim == 2:
            low_data_np = np.repeat(low_data_np[..., None], 3, axis=2)

        # Ensure at least 3 channels; ignore any potential alpha channel beyond RGB.
        if low_data_np.shape[2] < 3:
            # Pad missing channels with reasonable defaults (T_view=1, T_light=1, depth=0)
            pad = 3 - low_data_np.shape[2]
            pad_values = np.zeros((low_data_np.shape[0], low_data_np.shape[1], pad), dtype=np.float32)
            if pad >= 1:
                pad_values[..., 0] = 1.0
            if pad >= 2:
                pad_values[..., 1] = 1.0
            low_data_np = np.concatenate([low_data_np, pad_values], axis=2)

        # Extract and normalise auxiliary channels
        t_view_np = np.clip(low_data_np[..., 0:1], 0.0, 1.0)
        t_light_np = np.clip(low_data_np[..., 1:2], 0.0, 1.0)
        depth_np = low_data_np[..., 2:3]
        depth_np = np.maximum(depth_np, 0.0)
        if self._config.depth_normalization_max > 0.0:
            depth_np = depth_np / float(self._config.depth_normalization_max)
        depth_np = np.clip(depth_np, 0.0, 1.0)

        # Optional normals image at low resolution ------------------------------------------------
        # Layout in the renderer:
        #   RGB: world-space unit normal components (each in approximately [-1, 1]).
        low_normals_path = low_path.replace("_low.pfm", "_low_normals.pfm")
        low_normals_np: Optional[np.ndarray]
        if self._config.use_normals:
            if os.path.isfile(low_normals_path):
                low_normals_np = read_pfm(low_normals_path)
            else:
                # Fall back to a zero normal field if normals are requested but missing.
                h_low, w_low, _ = low_np.shape
                low_normals_np = np.zeros((h_low, w_low, 3), dtype=np.float32)
        else:
            low_normals_np = None

        if low_normals_np is not None:
            if low_normals_np.ndim == 2:
                low_normals_np = np.repeat(low_normals_np[..., None], 3, axis=2)
            if low_normals_np.shape[2] < 3:
                pad = 3 - low_normals_np.shape[2]
                pad_values = np.zeros((low_normals_np.shape[0], low_normals_np.shape[1], pad), dtype=np.float32)
                low_normals_np = np.concatenate([low_normals_np, pad_values], axis=2)
            # Clamp to a reasonable signed range; we deliberately keep the sign.
            low_normals_np = np.clip(low_normals_np, -1.0, 1.0)

        # To torch (C, H, W)
        # Convert HWC (NumPy) -> CHW (PyTorch). Channel-first is the standard for torch layers.
        # permute(2,0,1): move channels axis from last to first. contiguous(): ensure memory layout is contiguous.
        high = torch.from_numpy(high_np).permute(2, 0, 1).contiguous()  # [3, H, W]
        low = torch.from_numpy(low_np).permute(2, 0, 1).contiguous()    # [3, h, w]
        t_view = torch.from_numpy(t_view_np).permute(2, 0, 1).contiguous()   # [1, h, w]
        t_light = torch.from_numpy(t_light_np).permute(2, 0, 1).contiguous() # [1, h, w]
        depth = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous()     # [1, h, w]
        normals: Optional[torch.Tensor]
        if low_normals_np is not None:
            normals = torch.from_numpy(low_normals_np).permute(2, 0, 1).contiguous()  # [3, h, w]
        else:
            normals = None

        # Upsample low to high resolution using chosen interpolation
        _, target_h, target_w = high.shape
        low_up_rgb = F.interpolate(
            low.unsqueeze(0),
            size=(target_h, target_w),
            mode=self._config.upsample_mode,
            align_corners=self._config.align_corners if self._config.upsample_mode in ("bilinear", "bicubic") else None,
        ).squeeze(0)

        # Upsample auxiliary channels and concatenate with RGB input
        aux_channels: List[torch.Tensor] = []
        if self._config.use_view_transmittance:
            aux_channels.append(t_view)
        if self._config.use_light_transmittance:
            aux_channels.append(t_light)
        if self._config.use_linear_depth:
            aux_channels.append(depth)

        if aux_channels:
            aux_stack = torch.cat(aux_channels, dim=0)  # [C_aux, h, w]
            aux_up = F.interpolate(
                aux_stack.unsqueeze(0),
                size=(target_h, target_w),
                mode=self._config.upsample_mode,
                align_corners=self._config.align_corners if self._config.upsample_mode in ("bilinear", "bicubic") else None,
            ).squeeze(0)
            low_up = torch.cat([low_up_rgb, aux_up], dim=0)
        else:
            low_up = low_up_rgb

        # Optionally upsample normals and append them after the scalar auxiliary channels.
        if normals is not None:
            normals_up = F.interpolate(
                normals.unsqueeze(0),
                size=(target_h, target_w),
                mode=self._config.upsample_mode,
                align_corners=self._config.align_corners if self._config.upsample_mode in ("bilinear", "bicubic") else None,
            ).squeeze(0)
            low_up = torch.cat([low_up, normals_up], dim=0)

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


