import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Ensure the local package under ./src is importable when running as a script.
# This keeps the workflow simple on Windows (no need to install editable packages).
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from VolumetricCloudsTraining.dataset_cloud_pairs import (
    CloudPairDataset,
    CloudPairDatasetConfig,
    compute_dataset_index_splits,
)
from VolumetricCloudsTraining.models.unet import UNet, UNetConfig


# ================================================================
#                 PAPER QUALITATIVE FIGURE GENERATOR
# ================================================================
#
# This script generates a small number of *paper-ready* qualitative figures:
# - A buffer visualization (RGB + aux channels shown separately, not "squashed")
# - A model comparison grid (input / prediction / ground truth / abs error)
#
# Design goals:
# - Be explicit and reproducible (fixed split seed and fixed test sample index)
# - Avoid cherry-picking by default (we pick the first test sample)
# - Avoid flooding the paper with checkpoint curves (we show 1 checkpoint per model)
#


@dataclass
class _SplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    split_seed: int


@dataclass
class _InferRuntimeConfig:
    clamp_min: float
    clamp_max: float
    upsample_mode: str
    align_corners: bool
    device: str
    depth_normalization_max: float


@dataclass
class _ToneMapConfig:
    # Visualization-only tonemapping for exported PNGs.
    mode: str
    exposure: float
    gamma: float


@dataclass
class _ExperimentSpec:
    name: str
    use_view_transmittance: bool
    use_light_transmittance: bool
    use_linear_depth: bool
    use_normals: bool


def _project_root() -> str:
    # This script lives in Source/VolumetricCloudsTraining/scripts/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(_project_root(), "..", ".."))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML root to be a dict, got {type(data)}")
    return data


def _select_device(spec: str) -> torch.device:
    if spec.lower() in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if spec.lower() == "cpu":
        return torch.device("cpu")
    # Default: auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_common_from_config(config_path: str) -> Tuple[str, _SplitConfig, _InferRuntimeConfig]:
    cfg = _read_yaml(config_path)

    # Dataset root
    infer = cfg.get("infer", {})
    if not isinstance(infer, dict):
        raise TypeError("Expected 'infer' section to be a dict.")

    dataset_root_rel = str(infer.get("input", "")).strip()
    if not dataset_root_rel:
        raise ValueError("infer.input is missing or empty in config.")

    dataset_root = os.path.join(_project_root(), dataset_root_rel)

    # Split config
    split_cfg = _SplitConfig(
        train_fraction=float(infer.get("train_fraction", 0.8)),
        val_fraction=float(infer.get("val_fraction", 0.15)),
        test_fraction=float(infer.get("test_fraction", 0.05)),
        split_seed=int(infer.get("split_seed", 12345)),
    )

    # Inference runtime config
    runtime_cfg = _InferRuntimeConfig(
        clamp_min=float(infer.get("clamp_min", 0.0)),
        clamp_max=float(infer.get("clamp_max", 1.0)),
        upsample_mode=str(infer.get("upsample_mode", "bicubic")),
        align_corners=bool(infer.get("align_corners", False)),
        device=str(infer.get("device", "auto")),
        depth_normalization_max=float(infer.get("depth_normalization_max", 70000.0)),
    )

    return dataset_root, split_cfg, runtime_cfg


def _infer_experiment_from_name(name: str) -> _ExperimentSpec:
    # Keep mapping explicit and predictable.
    # This matches the naming used in Exp1 outputs.
    if name == "rgb_only":
        return _ExperimentSpec(
            name=name,
            use_view_transmittance=False,
            use_light_transmittance=False,
            use_linear_depth=False,
            use_normals=False,
        )
    if name == "rgb_plus_depth":
        return _ExperimentSpec(
            name=name,
            use_view_transmittance=False,
            use_light_transmittance=False,
            use_linear_depth=True,
            use_normals=False,
        )
    if name == "rgb_plus_normals":
        return _ExperimentSpec(
            name=name,
            use_view_transmittance=False,
            use_light_transmittance=False,
            use_linear_depth=False,
            use_normals=True,
        )
    if name == "rgb_plus_viewT":
        return _ExperimentSpec(
            name=name,
            use_view_transmittance=True,
            use_light_transmittance=False,
            use_linear_depth=False,
            use_normals=False,
        )
    if name == "rgb_plus_viewT_normals":
        return _ExperimentSpec(
            name=name,
            use_view_transmittance=True,
            use_light_transmittance=False,
            use_linear_depth=False,
            use_normals=True,
        )

    raise ValueError(f"Unsupported experiment name: {name!r}")


def _resolve_checkpoint(outputs_root: str, run_name: str, exp_name: str, epoch: int) -> str:
    ckpt = os.path.join(
        outputs_root,
        run_name,
        "experiments",
        exp_name,
        "checkpoints",
        f"unet_epoch_{epoch}.pt",
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _get_test_indices(dataset_root: str, split: _SplitConfig) -> List[int]:
    # Use the same deterministic pairing logic as training/infer.
    pairs = CloudPairDataset._discover_pairs(dataset_root, limit=None)
    num_samples = len(pairs)
    _, _, test_indices = compute_dataset_index_splits(
        num_samples=num_samples,
        train_fraction=split.train_fraction,
        val_fraction=split.val_fraction,
        test_fraction=split.test_fraction,
        split_seed=split.split_seed,
    )
    return test_indices


def _load_sample_fullres(
    dataset_root: str,
    split: _SplitConfig,
    runtime: _InferRuntimeConfig,
    exp: _ExperimentSpec,
    test_local_index: int,
) -> Dict[str, torch.Tensor]:
    # We disable cropping for paper figures so we show full-resolution images.
    ds_cfg = CloudPairDatasetConfig(
        root_dir=dataset_root,
        crop_size=None,
        clamp_min=runtime.clamp_min,
        clamp_max=runtime.clamp_max,
        upsample_mode=runtime.upsample_mode,
        align_corners=runtime.align_corners,
        limit_pairs=None,
        use_view_transmittance=exp.use_view_transmittance,
        use_light_transmittance=exp.use_light_transmittance,
        use_linear_depth=exp.use_linear_depth,
        use_normals=exp.use_normals,
        depth_normalization_max=runtime.depth_normalization_max,
    )
    dataset = CloudPairDataset(ds_cfg)
    test_indices = _get_test_indices(dataset_root, split)
    if len(test_indices) == 0:
        raise ValueError("Test split is empty; cannot generate qualitative figures.")
    if test_local_index < 0 or test_local_index >= len(test_indices):
        raise IndexError(f"test_local_index={test_local_index} out of bounds for test split length {len(test_indices)}.")

    sample_global_index = test_indices[test_local_index]
    sample = dataset[sample_global_index]
    low_up = sample["low_up"]  # [C, H, W]
    high = sample["high"]      # [3, H, W]
    return {
        "low_up": low_up,
        "high": high,
    }


def _tensor_to_hwc_rgb01(t: torch.Tensor) -> np.ndarray:
    # Expect [3, H, W] in linear space (may be HDR); do not clip here.
    np_img = t.detach().cpu().numpy().astype(np.float32, copy=False)
    np_img = np.transpose(np_img, (1, 2, 0))
    return np_img


def _tensor_to_hw01(t: torch.Tensor) -> np.ndarray:
    # Expect [1, H, W] or [H, W]
    if t.ndim == 3:
        t = t[0:1, ...]
    np_img = t.detach().cpu().numpy().astype(np.float32, copy=False)
    np_img = np.squeeze(np_img)
    return np.clip(np_img, 0.0, 1.0)


def _tensor_normals_to_rgb01(t: torch.Tensor) -> np.ndarray:
    # Expect [3, H, W] in approximately [-1,1], map to [0,1] for display.
    np_img = t.detach().cpu().numpy().astype(np.float32, copy=False)
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = 0.5 * (np_img + 1.0)
    return np.clip(np_img, 0.0, 1.0)


def _tonemap_rgb_to_rgb01(rgb_linear: np.ndarray, cfg: _ToneMapConfig, enable: bool) -> np.ndarray:
    # Keep this purely for visualization; training/inference uses the (linear) RGB range
    # produced by the dataset/runtime configuration (often clamped to [0,1] in our experiments).
    rgb = np.maximum(rgb_linear, 0.0) * float(cfg.exposure)

    mode = str(cfg.mode).strip().lower()
    if not enable or mode == "off":
        return np.clip(rgb, 0.0, 1.0)

    # Auto: prefer ACES for HDR-like ranges (matches renderer tonemapping style).
    if mode == "auto":
        mode = "aces"

    if mode == "aces":
        # Ported from `Source/VolumetricCloudsRender/Shaders/tonemapping.frag`.
        # Source references (as in shader):
        # - https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
        # - https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl

        # sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
        aces_in = np.array(
            [
                [0.59719, 0.35458, 0.04823],
                [0.07600, 0.90834, 0.01566],
                [0.02840, 0.13383, 0.83777],
            ],
            dtype=np.float32,
        )

        # ODT_SAT => XYZ => D60_2_D65 => sRGB
        aces_out = np.array(
            [
                [1.60475, -0.53108, -0.07367],
                [-0.10208, 1.10813, -0.00605],
                [-0.00327, -0.07276, 1.07602],
            ],
            dtype=np.float32,
        )

        # Apply matrix multiply per pixel in the same order as GLSL: color = color * mat3.
        rgb = rgb @ aces_in

        # Apply RRT and ODT fit.
        a = rgb * (rgb + 0.0245786) - 0.000090537
        b = rgb * (0.983729 * rgb + 0.4329510) + 0.238081
        rgb = a / b

        rgb = rgb @ aces_out
        rgb = np.clip(rgb, 0.0, 1.0)
    elif mode == "reinhard":
        # Classic Reinhard: x / (1 + x)
        rgb = rgb / (1.0 + rgb)
    elif mode == "clamp":
        # Simple clamp (legacy behavior).
        rgb = rgb
    else:
        raise ValueError(f"Unsupported tonemap mode: {cfg.mode!r} (expected: auto|aces|reinhard|clamp|off)")

    gamma = float(cfg.gamma)
    if gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        # Display gamma (approx sRGB) so highlights don't look harsh.
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)

    return np.clip(rgb, 0.0, 1.0)


def _run_model(
    low_up: torch.Tensor,
    checkpoint_path: str,
    runtime: _InferRuntimeConfig,
    model_base_channels: int,
    model_bilinear: bool,
    model_learn_residual: bool,
) -> np.ndarray:
    device = _select_device(runtime.device)

    in_channels = int(low_up.shape[0])
    model = UNet(
        UNetConfig(
            in_channels=in_channels,
            out_channels=3,
            base_channels=model_base_channels,
            bilinear=model_bilinear,
            learn_residual=model_learn_residual,
        )
    ).to(device)
    model.eval()

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)

    with torch.no_grad():
        pred = model(low_up.unsqueeze(0).to(device))
        # Clamp to the configured training/infer range (supports HDR if clamp_max > 1).
        pred = torch.clamp(pred, float(runtime.clamp_min), float(runtime.clamp_max))
        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        pred_np = np.transpose(pred_np, (1, 2, 0))
        return pred_np


def _save_fig(fig: plt.Figure, out_path: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _compute_integral_image(image: np.ndarray) -> np.ndarray:
    # Integral image for fast sliding-window sums.
    # Output shape matches input, where ii[y,x] = sum_{i<=y, j<=x} image[i,j].
    return np.cumsum(np.cumsum(image, axis=0), axis=1)


def _window_sum(ii: np.ndarray, x0: int, y0: int, w: int, h: int) -> float:
    # Sum over rectangle [x0, x0+w) x [y0, y0+h) using integral image.
    x1 = x0 + w - 1
    y1 = y0 + h - 1

    a = ii[y1, x1]
    b = ii[y1, x0 - 1] if x0 > 0 else 0.0
    c = ii[y0 - 1, x1] if y0 > 0 else 0.0
    d = ii[y0 - 1, x0 - 1] if (x0 > 0 and y0 > 0) else 0.0
    return float(a - b - c + d)


def _auto_select_crop(input_rgb: np.ndarray, gt_rgb: np.ndarray, crop_size: int) -> Tuple[int, int, int]:
    # Pick a visually "interesting" crop by maximizing the mean absolute error of bicubic vs GT.
    # This is deterministic for a given sample and avoids manual cherry-picking.
    h, w, _ = input_rgb.shape
    cs = int(crop_size)
    cs = max(8, min(cs, h, w))

    err = np.mean(np.abs(input_rgb - gt_rgb), axis=2)
    ii = _compute_integral_image(err)

    best_x = 0
    best_y = 0
    best_score = -1.0
    for y in range(0, h - cs + 1):
        # Small speed: compute row sums by scanning x in inner loop.
        for x in range(0, w - cs + 1):
            score = _window_sum(ii, x, y, cs, cs) / float(cs * cs)
            if score > best_score:
                best_score = score
                best_x = x
                best_y = y

    return best_x, best_y, cs


def _plot_teaser(
    input_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    out_path: str,
) -> None:
    # Simple "at a glance" intent figure: low-res input (upsampled) -> U-Net output.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8))
    axes[0].imshow(input_rgb)
    axes[0].set_title("Input (low-res RGB, bicubic upsample)")
    axes[0].axis("off")

    axes[1].imshow(pred_rgb)
    axes[1].set_title("U-Net output (4× upsampled)")
    axes[1].axis("off")

    _save_fig(fig, out_path)


def _plot_zoom_inset(
    input_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    crop: Tuple[int, int, int],
    out_path: str,
    error_vmax: float = 0.15,
) -> None:
    # Two-row layout:
    # - Top: full images with a marked crop region.
    # - Bottom: zoomed crops (same region) for easy visual comparison.
    x, y, cs = crop
    h, w, _ = input_rgb.shape
    x = max(0, min(x, w - cs))
    y = max(0, min(y, h - cs))

    def _draw_rect(ax) -> None:
        # Rectangle outline for the crop region.
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                cs,
                cs,
                fill=False,
                edgecolor="cyan",
                linewidth=2.0,
            )
        )

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))

    # --- Full images ---
    axes[0, 0].imshow(input_rgb)
    axes[0, 0].set_title("Input (bicubic)")
    _draw_rect(axes[0, 0])
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_rgb)
    axes[0, 1].set_title("Prediction")
    _draw_rect(axes[0, 1])
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gt_rgb)
    axes[0, 2].set_title("Ground truth")
    _draw_rect(axes[0, 2])
    axes[0, 2].axis("off")

    full_err = np.mean(np.abs(pred_rgb - gt_rgb), axis=2)
    im0 = axes[0, 3].imshow(full_err, cmap="inferno", vmin=0.0, vmax=error_vmax)
    axes[0, 3].set_title("Abs error (mean RGB)")
    _draw_rect(axes[0, 3])
    axes[0, 3].axis("off")
    fig.colorbar(im0, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # --- Zoom crops ---
    crop_in = input_rgb[y : y + cs, x : x + cs, :]
    crop_pred = pred_rgb[y : y + cs, x : x + cs, :]
    crop_gt = gt_rgb[y : y + cs, x : x + cs, :]
    crop_err = np.mean(np.abs(crop_pred - crop_gt), axis=2)

    axes[1, 0].imshow(crop_in)
    axes[1, 0].set_title(f"Zoom: input ({cs}×{cs})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(crop_pred)
    axes[1, 1].set_title("Zoom: prediction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(crop_gt)
    axes[1, 2].set_title("Zoom: ground truth")
    axes[1, 2].axis("off")

    im1 = axes[1, 3].imshow(crop_err, cmap="inferno", vmin=0.0, vmax=error_vmax)
    axes[1, 3].set_title("Zoom: abs error")
    axes[1, 3].axis("off")
    fig.colorbar(im1, ax=axes[1, 3], fraction=0.046, pad=0.04)

    _save_fig(fig, out_path)


def _plot_aux_buffers(
    low_up: torch.Tensor,
    exp: _ExperimentSpec,
    out_path: str,
    tonemap: _ToneMapConfig,
    tonemap_enabled: bool,
) -> None:
    # Channel layout in CloudPairDataset:
    # - 0..2: RGB
    # - +viewT (1ch), +lightT (1ch), +depth (1ch) in that order, when enabled
    # - +normals (3ch) appended last, when enabled

    c = 3
    rgb = _tonemap_rgb_to_rgb01(_tensor_to_hwc_rgb01(low_up[0:3, ...]), tonemap, enable=tonemap_enabled)

    view_t = None
    light_t = None
    depth = None
    normals = None

    if exp.use_view_transmittance:
        view_t = _tensor_to_hw01(low_up[c : c + 1, ...])
        c += 1
    if exp.use_light_transmittance:
        light_t = _tensor_to_hw01(low_up[c : c + 1, ...])
        c += 1
    if exp.use_linear_depth:
        depth = _tensor_to_hw01(low_up[c : c + 1, ...])
        c += 1
    if exp.use_normals:
        normals = _tensor_normals_to_rgb01(low_up[c : c + 3, ...])
        c += 3

    # Keep a consistent panel set for the paper:
    # - Always show RGB
    # - Then show viewT / lightT / depth if present
    # - Then show normals if present
    panels: List[Tuple[str, str]] = []
    panels.append(("Input RGB (bicubic upsample)", "rgb"))
    if view_t is not None:
        panels.append(("View transmittance (upsampled)", "view_t"))
    if light_t is not None:
        panels.append(("Light transmittance (upsampled)", "light_t"))
    if depth is not None:
        panels.append(("Linear depth (normalized, upsampled)", "depth"))
    if normals is not None:
        panels.append(("Normals (world-space, mapped, upsampled)", "normals"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, key) in zip(axes, panels):
        if key == "rgb":
            ax.imshow(rgb)
        elif key == "view_t":
            ax.imshow(view_t, cmap="gray", vmin=0.0, vmax=1.0)
        elif key == "light_t":
            ax.imshow(light_t, cmap="gray", vmin=0.0, vmax=1.0)
        elif key == "depth":
            ax.imshow(depth, cmap="magma", vmin=0.0, vmax=1.0)
        elif key == "normals":
            ax.imshow(normals)
        ax.set_title(title)
        ax.axis("off")

    _save_fig(fig, out_path)


def _plot_prediction_grid(
    rows: List[Tuple[str, np.ndarray]],
    input_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    out_path: str,
    error_vmax: float = 0.15,
) -> None:
    # Grid columns:
    # - Input (bicubic upsample)
    # - Prediction
    # - Ground truth
    # - Mean absolute error heatmap

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14.5, 3.3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (label, pred_rgb) in enumerate(rows):
        err = np.mean(np.abs(pred_rgb - gt_rgb), axis=2)

        axes[r, 0].imshow(input_rgb)
        axes[r, 0].set_title("Input (bicubic)")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(pred_rgb)
        axes[r, 1].set_title(f"Prediction ({label})")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(gt_rgb)
        axes[r, 2].set_title("Ground truth")
        axes[r, 2].axis("off")

        im = axes[r, 3].imshow(err, cmap="inferno", vmin=0.0, vmax=error_vmax)
        axes[r, 3].set_title("Abs error (mean RGB)")
        axes[r, 3].axis("off")
        fig.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)

    _save_fig(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a training config YAML (exp1/exp2).")
    parser.add_argument("--run_name", type=str, required=True, help="Outputs/<run_name>/... folder name (e.g. exp1_buffers_general).")
    parser.add_argument("--test_index", type=int, default=0, help="Local index within the test split (default: 0).")
    parser.add_argument("--epoch", type=int, default=100, help="Checkpoint epoch to visualize (default: 100).")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to write PNGs (default: Report/figures).")
    parser.add_argument("--crop_size", type=int, default=160, help="Zoom crop size in pixels (default: 160).")
    # HDR-friendly visualization controls.
    parser.add_argument(
        "--tonemap",
        type=str,
        default="auto",
        help="Tonemap for exported PNGs: auto|aces|reinhard|clamp|off (default: auto).",
    )
    parser.add_argument("--exposure", type=float, default=1.0, help="Exposure multiplier before tonemapping (default: 1.0).")
    parser.add_argument("--gamma", type=float, default=2.2, help="Display gamma after tonemapping (default: 2.2).")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["rgb_only", "rgb_plus_viewT_normals"],
        help="Experiment names to compare (default: rgb_only rgb_plus_viewT_normals).",
    )
    args = parser.parse_args()

    dataset_root, split_cfg, runtime_cfg = _load_common_from_config(args.config)

    # Determine output directory.
    output_dir = args.output_dir.strip()
    if not output_dir:
        output_dir = os.path.join(_repo_root(), "Report", "figures")
    output_dir = os.path.abspath(output_dir)
    _ensure_dir(output_dir)

    # Model settings: read from config (fall back to your paper defaults).
    cfg = _read_yaml(args.config)
    infer = cfg.get("infer", {})
    if not isinstance(infer, dict):
        raise TypeError("Expected 'infer' section to be a dict.")
    model_base_channels = int(infer.get("model_base_channels", 32))
    model_bilinear = bool(infer.get("model_bilinear", True))
    model_learn_residual = bool(infer.get("model_learn_residual", True))

    # Outputs folder: configs store Outputs relative to project root.
    outputs_root = os.path.join(_project_root(), "Outputs")

    # Use the first experiment to load a sample (input/GT). This defines the input RGB and GT.
    # For other experiments the input may include more channels, but the *RGB upsample* is always first.
    base_exp = _infer_experiment_from_name(args.experiments[0])
    base_sample = _load_sample_fullres(
        dataset_root=dataset_root,
        split=split_cfg,
        runtime=runtime_cfg,
        exp=base_exp,
        test_local_index=int(args.test_index),
    )
    base_low_up = base_sample["low_up"]
    gt = base_sample["high"]

    tonemap_cfg = _ToneMapConfig(mode=str(args.tonemap), exposure=float(args.exposure), gamma=float(args.gamma))
    # For paper figures we treat tonemapping as a display concern: enable it unless explicitly disabled.
    tonemap_enabled = str(args.tonemap).strip().lower() != "off"

    input_rgb = _tonemap_rgb_to_rgb01(_tensor_to_hwc_rgb01(base_low_up[0:3, ...]), tonemap_cfg, enable=tonemap_enabled)
    gt_rgb = _tonemap_rgb_to_rgb01(_tensor_to_hwc_rgb01(gt), tonemap_cfg, enable=tonemap_enabled)

    # Export an aux buffer visualization for the richer experiment (if present).
    # This figure answers: "what are these channels, visually?"
    rich_exp_name = args.experiments[-1]
    rich_exp = _infer_experiment_from_name(rich_exp_name)
    rich_sample = _load_sample_fullres(
        dataset_root=dataset_root,
        split=split_cfg,
        runtime=runtime_cfg,
        exp=rich_exp,
        test_local_index=int(args.test_index),
    )
    aux_out = os.path.join(output_dir, f"qual_{args.run_name}_test{args.test_index:02d}_aux_{rich_exp_name}.png")
    _plot_aux_buffers(rich_sample["low_up"], rich_exp, aux_out, tonemap=tonemap_cfg, tonemap_enabled=tonemap_enabled)

    # Run inference for each experiment and build the comparison grid.
    rows: List[Tuple[str, np.ndarray]] = []
    hero_pred: Optional[np.ndarray] = None
    for exp_name in args.experiments:
        exp = _infer_experiment_from_name(exp_name)
        sample = _load_sample_fullres(
            dataset_root=dataset_root,
            split=split_cfg,
            runtime=runtime_cfg,
            exp=exp,
            test_local_index=int(args.test_index),
        )
        ckpt = _resolve_checkpoint(outputs_root, args.run_name, exp_name, int(args.epoch))
        pred = _run_model(
            low_up=sample["low_up"],
            checkpoint_path=ckpt,
            runtime=runtime_cfg,
            model_base_channels=model_base_channels,
            model_bilinear=model_bilinear,
            model_learn_residual=model_learn_residual,
        )
        pred_disp = _tonemap_rgb_to_rgb01(pred, tonemap_cfg, enable=tonemap_enabled)
        rows.append((exp_name, pred_disp))
        # Use the last experiment as the default "hero" (usually the richer input).
        if exp_name == args.experiments[-1]:
            hero_pred = pred_disp

    grid_out = os.path.join(output_dir, f"qual_{args.run_name}_test{args.test_index:02d}_compare.png")
    _plot_prediction_grid(rows=rows, input_rgb=input_rgb, gt_rgb=gt_rgb, out_path=grid_out)

    # Teaser figure (input -> prediction).
    if hero_pred is not None:
        teaser_out = os.path.join(output_dir, f"qual_{args.run_name}_test{args.test_index:02d}_teaser.png")
        _plot_teaser(input_rgb=input_rgb, pred_rgb=hero_pred, out_path=teaser_out)

        # Zoom inset figure (auto-selected ROI).
        crop = _auto_select_crop(input_rgb=input_rgb, gt_rgb=gt_rgb, crop_size=int(args.crop_size))
        zoom_out = os.path.join(
            output_dir,
            f"qual_{args.run_name}_test{args.test_index:02d}_zoom_{args.experiments[-1]}.png",
        )
        _plot_zoom_inset(
            input_rgb=input_rgb,
            pred_rgb=hero_pred,
            gt_rgb=gt_rgb,
            crop=crop,
            out_path=zoom_out,
        )


if __name__ == "__main__":
    main()

