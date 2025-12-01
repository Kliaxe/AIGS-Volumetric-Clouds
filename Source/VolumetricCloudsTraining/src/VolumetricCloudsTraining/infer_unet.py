import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .models.unet import UNet, UNetConfig
from .pfm_io import read_pfm, write_pfm
from .dataset_cloud_pairs import compute_dataset_index_splits, CloudPairDataset


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

    # Paths / dataset split selection ------------------------------------------
    checkpoint: str                   # checkpoint .pt file to load
    input_path: str                   # file or directory, or dataset root when using splits
    output_dir: str                   # directory to write results
    input_glob: str = "*_low.pfm"     # used if input_path is a directory and split_mode="custom"
    split_mode: str = "custom"        # "custom" | "train" | "val" | "test" | "all"

    # Dataset split configuration (should mirror TrainConfig when using splits)
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 12345
    # Optional: when not None, restrict the selected split ("train"/"val"/"test"/"all")
    # to a single sample at this index within the chosen split. For example,
    # split_mode="test" and split_sample_index=0 will run on the first test sample only.
    split_sample_index: Optional[int] = None

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

    # Experiment-aware checkpoint handling --------------------------------------
    # When run_all_checkpoints is True, checkpoints are discovered in
    # checkpoint_dir using checkpoint_glob and each checkpoint is evaluated on
    # the same input(s). Outputs are written under:
    #   output_dir/<experiment_name>/<checkpoint_tag>/<input>_pred.pfm
    experiment_name: str = ""
    checkpoint_dir: str = ""
    checkpoint_glob: str = "unet_epoch_*.pt"
    run_all_checkpoints: bool = False

    # Auxiliary input feature toggles (must match training) ---------------------
    use_view_transmittance: bool = True
    use_light_transmittance: bool = True
    use_linear_depth: bool = True
    # When enabled, *_low_normals.pfm is loaded and three RGB normal channels
    # are concatenated to the input after any scalar auxiliary channels.
    use_normals: bool = False

    # Normalisation scale for the linear depth channel (world units).
    depth_normalization_max: float = 40000.0


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


def _resolve_checkpoints(config: InferConfig) -> List[Tuple[str, str]]:
    """
    Determine which checkpoints to evaluate and derive a short tag for each.

    Returns a list of (checkpoint_path, checkpoint_tag) pairs. For the
    single-checkpoint case, checkpoint_tag is an empty string.
    """
    # Single checkpoint: preserve original behaviour.
    if not config.run_all_checkpoints:
        return [(config.checkpoint, "")]

    if not config.checkpoint_dir:
        raise ValueError("run_all_checkpoints=True requires 'checkpoint_dir' to be set in the config.")

    pattern = os.path.join(config.checkpoint_dir, config.checkpoint_glob or "*.pt")
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in directory {config.checkpoint_dir!r} "
            f"with pattern {config.checkpoint_glob!r}."
        )

    resolved: List[Tuple[str, str]] = []
    for path in paths:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        # Use the checkpoint filename (without extension) as a human-readable tag.
        resolved.append((path, name))
    return resolved


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
    in_channels = 3
    if config.use_view_transmittance:
        in_channels += 1
    if config.use_light_transmittance:
        in_channels += 1
    if config.use_linear_depth:
        in_channels += 1
    if config.use_normals:
        # RGB normals contribute three additional channels.
        in_channels += 3

    unet_config = UNetConfig(
        in_channels=in_channels,
        out_channels=3,
        base_channels=config.model_base_channels,
        bilinear=config.model_bilinear,
        learn_residual=config.model_learn_residual,
    )
    model = UNet(unet_config)
    model = model.to(device)
    model.eval()

    # ----------------- Inputs -----------------
    # Two modes:
    #  - split_mode="custom": behave like a generic PFM super‑resolution tool, using
    #    input_path + input_glob (existing behaviour).
    #  - split_mode in {"train","val","test","all"}: treat input_path as the dataset
    #    root (e.g. TrainingCaptures), reproduce the same index split as training
    #    and run inference on that subset.
    if config.split_mode.lower() == "custom":
        inputs = _gather_inputs(config.input_path, config.input_glob, config.recursive)
    else:
        split_mode = config.split_mode.lower()
        if split_mode not in ("train", "val", "test", "all"):
            raise ValueError(
                f"Unsupported split_mode={config.split_mode!r}. "
                "Expected 'custom', 'train', 'val', 'test' or 'all'."
            )

        pairs = CloudPairDataset._discover_pairs(config.input_path, limit=None)
        num_samples = len(pairs)
        train_indices, val_indices, test_indices = compute_dataset_index_splits(
            num_samples=num_samples,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
            split_seed=config.split_seed,
        )

        selected_indices: List[int]
        if split_mode == "train":
            selected_indices = train_indices
        elif split_mode == "val":
            selected_indices = val_indices
        elif split_mode == "test":
            selected_indices = test_indices
        else:  # "all"
            selected_indices = train_indices + val_indices + test_indices

        # Optionally restrict the chosen split to a single sample by index so
        # that, for example, we can run only one test image across all
        # experiments / checkpoints.
        if config.split_sample_index is not None:
            if len(selected_indices) == 0:
                raise ValueError(
                    "split_sample_index was provided but the selected split contains no samples."
                )
            local_index = config.split_sample_index
            if local_index < 0 or local_index >= len(selected_indices):
                raise IndexError(
                    f"split_sample_index={local_index} is out of bounds for split "
                    f"of length {len(selected_indices)}."
                )
            selected_indices = [selected_indices[local_index]]

        inputs = [pairs[i][0] for i in selected_indices]

    if len(inputs) == 0:
        raise FileNotFoundError(f"No inputs found using path={config.input_path!r} and pattern={config.input_glob!r}")

    # Determine which checkpoints to evaluate.
    checkpoints = _resolve_checkpoints(config)

    # ----------------- Run -----------------
    with torch.no_grad():
        for ckpt_path, ckpt_tag in checkpoints:
            # Load checkpoint weights into the shared model instance.
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=True)
            print(f"Loaded checkpoint: {ckpt_path}")

            for in_path in inputs:
                # Load low-resolution colour PFM (H, W[, 3]) -> ensure 3 channels
                np_img = read_pfm(in_path)
                if np_img.ndim == 2:
                    np_img = np.repeat(np_img[..., None], 3, axis=2)

                # Clamp colour to expected range
                np_img = np.clip(np_img, config.clamp_min, config.clamp_max).astype(np.float32, copy=False)
                h, w, _ = np_img.shape

                # Derive corresponding low-data path (optional)
                base, ext = os.path.splitext(in_path)
                if base.endswith("_low"):
                    data_path = f"{base}_data{ext}"
                else:
                    data_path = base + "_data" + ext

                np_data = read_pfm(data_path) if os.path.isfile(data_path) else None

                if np_data is None:
                    # Build default data image: T_view=1, T_light=1, depth=0
                    np_data = np.zeros((h, w, 3), dtype=np.float32)
                    np_data[..., 0] = 1.0
                    np_data[..., 1] = 1.0
                elif np_data.ndim == 2:
                    np_data = np.repeat(np_data[..., None], 3, axis=2)

                if np_data.shape[2] < 3:
                    pad = 3 - np_data.shape[2]
                    pad_values = np.zeros((h, w, pad), dtype=np.float32)
                    if pad >= 1:
                        pad_values[..., 0] = 1.0
                    if pad >= 2:
                        pad_values[..., 1] = 1.0
                    np_data = np.concatenate([np_data, pad_values], axis=2)

                # Extract and normalise auxiliary channels
                t_view_np = np.clip(np_data[..., 0:1], 0.0, 1.0)
                t_light_np = np.clip(np_data[..., 1:2], 0.0, 1.0)
                depth_np = np_data[..., 2:3]
                depth_np = np.maximum(depth_np, 0.0)
                if config.depth_normalization_max > 0.0:
                    depth_np = depth_np / float(config.depth_normalization_max)
                depth_np = np.clip(depth_np, 0.0, 1.0)

                # Derive corresponding low-normals path (optional, RGB in approximately [-1, 1])
                if base.endswith("_low"):
                    normals_path = f"{base}_normals{ext}"
                else:
                    normals_path = base + "_normals" + ext

                np_normals = None
                if config.use_normals and os.path.isfile(normals_path):
                    np_normals = read_pfm(normals_path)

                if config.use_normals and np_normals is None:
                    # Fall back to a zero normal field when normals are requested but missing.
                    np_normals = np.zeros((h, w, 3), dtype=np.float32)

                if np_normals is not None:
                    if np_normals.ndim == 2:
                        np_normals = np.repeat(np_normals[..., None], 3, axis=2)
                    if np_normals.shape[2] < 3:
                        pad = 3 - np_normals.shape[2]
                        pad_values = np.zeros((h, w, pad), dtype=np.float32)
                        np_normals = np.concatenate([np_normals, pad_values], axis=2)
                    # Clamp to a reasonable signed range while preserving direction.
                    np_normals = np.clip(np_normals, -1.0, 1.0)

                # To torch, CHW
                t_low_rgb = torch.from_numpy(np_img).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
                t_view = torch.from_numpy(t_view_np).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
                t_light = torch.from_numpy(t_light_np).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
                depth = torch.from_numpy(depth_np).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
                t_normals: Optional[torch.Tensor]
                if np_normals is not None:
                    t_normals = (
                        torch.from_numpy(np_normals)
                        .permute(2, 0, 1)
                        .contiguous()
                        .unsqueeze(0)
                        .to(device)
                    )
                else:
                    t_normals = None

                # Upsample to target size (scale factor, e.g., 4x)
                target_h = int(h * config.scale_factor)
                target_w = int(w * config.scale_factor)
                t_low_up_rgb = F.interpolate(
                    t_low_rgb,
                    size=(target_h, target_w),
                    mode=config.upsample_mode,
                    align_corners=config.align_corners if config.upsample_mode in ("bilinear", "bicubic") else None,
                )

                aux_tensors = []
                if config.use_view_transmittance:
                    aux_tensors.append(t_view)
                if config.use_light_transmittance:
                    aux_tensors.append(t_light)
                if config.use_linear_depth:
                    aux_tensors.append(depth)

                if aux_tensors:
                    aux_stack = torch.cat(aux_tensors, dim=1)  # [1, C_aux, h, w]
                    aux_up = F.interpolate(
                        aux_stack,
                        size=(target_h, target_w),
                        mode=config.upsample_mode,
                        align_corners=config.align_corners if config.upsample_mode in ("bilinear", "bicubic") else None,
                    )
                    t_input = torch.cat([t_low_up_rgb, aux_up], dim=1)
                else:
                    t_input = t_low_up_rgb

                # Optionally upsample normals and append them after scalar auxiliary channels.
                if t_normals is not None:
                    normals_up = F.interpolate(
                        t_normals,
                        size=(target_h, target_w),
                        mode=config.upsample_mode,
                        align_corners=config.align_corners if config.upsample_mode in ("bilinear", "bicubic") else None,
                    )
                    t_input = torch.cat([t_input, normals_up], dim=1)

                # Predict
                pred = model(t_input)  # [1, 3, H', W']
                pred = torch.clamp(pred, 0.0, 1.0)

                # Derive output directory and filename for this experiment / checkpoint.
                # We keep a flat directory structure and encode experiment / checkpoint
                # information directly in the filename, e.g.:
                #   light_only-unet_epoch_10-<input_basename>_pred.pfm
                base_output_dir = config.output_dir
                exp_name = config.experiment_name.strip()
                ckpt_output_dir = base_output_dir
                _ensure_dir(ckpt_output_dir)

                base = os.path.basename(in_path)
                name, ext = os.path.splitext(base)

                prefix_parts: List[str] = []
                if exp_name:
                    prefix_parts.append(exp_name)
                if ckpt_tag:
                    prefix_parts.append(ckpt_tag)

                if prefix_parts:
                    prefix = "-".join(prefix_parts)
                    filename = f"{prefix}-{name}{config.output_suffix}{ext}"
                else:
                    filename = f"{name}{config.output_suffix}{ext}"

                out_path = os.path.join(ckpt_output_dir, filename)
                out_np = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                write_pfm(out_path, out_np)
                print(f"Wrote: {out_path}")


