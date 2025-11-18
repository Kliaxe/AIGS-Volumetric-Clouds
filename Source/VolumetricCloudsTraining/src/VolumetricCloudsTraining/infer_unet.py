import os
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .models.unet import UNet, UNetConfig
from .pfm_io import read_pfm, write_pfm


# ================================================================
#                          INFERENCE CONFIG
# ================================================================


@dataclass
class InferConfig:
    """
    Configuration for running inference with a trained U‑Net.

    We keep this structurally similar to TrainConfig so that the model
    architecture (base_channels, bilinear, learn_residual) is always
    explicit and under config control. This avoids accidentally trying
    to load a checkpoint trained with one architecture into a different
    one at inference time.
    """

    # Paths ---------------------------------------------------------------------
    checkpoint: str                   # checkpoint .pt file to load
    input_path: str                   # file or directory
    output_dir: str                   # directory to write results
    input_glob: str = "*_low.pfm"     # used if input_path is a directory

    # Model configuration -------------------------------------------------------
    model_base_channels: int = 64
    model_bilinear: bool = True
    model_learn_residual: bool = True

    # Runtime behaviour ---------------------------------------------------------
    device: str = "auto"              # 'auto' | 'cuda' | 'cpu'
    upsample_mode: str = "bicubic"    # 'bilinear'|'bicubic'
    align_corners: bool = False
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    scale_factor: int = 4             # expected 320x180 -> 1280x720
    recursive: bool = False           # recurse when input_path is directory
    output_suffix: str = "_pred"      # appended before extension


# ================================================================
#                              HELPERS
# ================================================================


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if spec in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: Requested CUDA but it's unavailable or not compiled. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(spec)


def _gather_inputs(path: str, pattern: str, recursive: bool) -> List[str]:
    if os.path.isdir(path):
        search = os.path.join(path, "**", pattern) if recursive else os.path.join(path, pattern)
        return sorted(glob.glob(search, recursive=recursive))
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _derive_output_path(output_dir: str, input_file: str, suffix: str) -> str:
    base = os.path.basename(input_file)
    name, ext = os.path.splitext(base)
    return os.path.join(output_dir, f"{name}{suffix}{ext}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ================================================================
#                            INFERENCE
# ================================================================


def infer_unet(config: InferConfig) -> None:
    # ----------------- Setup -----------------
    device = _select_device(config.device)
    _ensure_dir(config.output_dir)

    # ----------------- Model -----------------
    # Mirror the way the training script constructs UNet: derive a UNetConfig
    # from the model_* fields so that architecture exactly matches training.
    unet_config = UNetConfig(
        in_channels=3,
        out_channels=3,
        base_channels=config.model_base_channels,
        bilinear=config.model_bilinear,
        learn_residual=config.model_learn_residual,
    )
    model = UNet(unet_config)
    state = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    # ----------------- Inputs -----------------
    inputs = _gather_inputs(config.input_path, config.input_glob, config.recursive)
    if len(inputs) == 0:
        raise FileNotFoundError(f"No inputs found using path={config.input_path!r} and pattern={config.input_glob!r}")

    # ----------------- Run -----------------
    with torch.no_grad():
        for in_path in inputs:
            # Load PFM (H, W[, 3]) -> ensure 3 channels
            np_img = read_pfm(in_path)
            if np_img.ndim == 2:
                np_img = np.repeat(np_img[..., None], 3, axis=2)

            # Clamp to expected range
            np_img = np.clip(np_img, config.clamp_min, config.clamp_max).astype(np.float32, copy=False)
            h, w, _ = np_img.shape

            # To torch, CHW
            t_low = torch.from_numpy(np_img).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)

            # Upsample to target size (scale factor, e.g., 4x)
            target_h = int(h * config.scale_factor)
            target_w = int(w * config.scale_factor)
            t_low_up = F.interpolate(
                t_low,
                size=(target_h, target_w),
                mode=config.upsample_mode,
                align_corners=config.align_corners if config.upsample_mode in ("bilinear", "bicubic") else None,
            )

            # Predict
            pred = model(t_low_up)  # [1, 3, H', W']
            pred = torch.clamp(pred, 0.0, 1.0)

            # Save
            out_path = _derive_output_path(config.output_dir, in_path, config.output_suffix)
            out_np = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
            write_pfm(out_path, out_np)
            print(f"Wrote: {out_path}")


