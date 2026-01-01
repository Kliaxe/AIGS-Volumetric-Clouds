import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib


# Use a non-interactive backend for environments without a display (e.g., CI/headless)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader, Subset

from .dataset_cloud_pairs import CloudPairDataset, CloudPairDatasetConfig, compute_dataset_index_splits
from .models.unet import UNet, UNetConfig
from .pfm_io import write_pfm


# ================================================================
#                          TRAINING CONFIG
# ================================================================


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "outputs"
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-4
    crop_size: Optional[int] = 256
    limit_pairs: Optional[int] = None
    num_workers: int = 0  # Windows default safer
    device: str = "auto"  # 'auto' | 'cuda' | 'cpu'
    seed: int = 42
    # Dataset split configuration -----------------------------------------------
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 12345
    # Logging / export behaviour -------------------------------------------------
    # When > 0, loss curves (PNG + TXT) are exported every export_every_n_epochs
    # epochs as well as at the final epoch. When 0, export occurs only at the
    # final epoch.
    export_every_n_epochs: int = 0
    # When True, include the dense per-step training loss curve in loss_curve.png.
    # This is useful for debugging, but can be disabled for "hero" paper figures
    # to keep plots visually simple.
    plot_step_loss: bool = False
    # Epoch checkpoint stride: save a checkpoint every N epochs (always saves
    # the final epoch checkpoint regardless of this value).
    save_epoch_stride: int = 1
    save_every_n_steps: int = 200
    log_every_n_steps: int = 20
    # Checkpointing behaviour ----------------------------------------------------
    # When True, only the final epoch checkpoint is saved (no intermediate step
    # or per‑epoch checkpoints). Useful for very long runs where you only care
    # about the last model state.
    save_only_last_epoch: bool = False
    # Model configuration --------------------------------------------
    model_base_channels: int = 64
    model_bilinear: bool = True
    model_learn_residual: bool = True

    # Auxiliary input feature toggles --------------------------------
    # When enabled, the dataset will read *_low_data.pfm and concatenate the
    # corresponding channels to the RGB input before feeding it to the U-Net.
    use_view_transmittance: bool = True
    use_light_transmittance: bool = True
    use_linear_depth: bool = True
    # When enabled, the dataset will read *_low_normals.pfm and concatenate the
    # three RGB normal channels to the input after any scalar auxiliary channels.
    use_normals: bool = False

    # Normalisation scale for the linear depth channel (world units).
    depth_normalization_max: float = 40000.0

    # Loss configuration ---------------------------------------------
    # When True, auxiliary channels are used to weight the per‑pixel L1 loss.
    # Default is False, which uses a plain, unweighted L1 loss on RGB only.
    use_auxiliary_in_loss: bool = False


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    # Explicit selection with graceful fallback
    if spec in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: Requested CUDA but it's unavailable or not compiled. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(spec)


def _ensure_dirs(base: str) -> None:
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base, "logs"), exist_ok=True)


def _compute_auxiliary_weights(low_up: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    """
    Derive a simple per‑pixel weighting map from the auxiliary channels present in low_up.

    Channel layout in low_up:
      - [0:3]: RGB input
      - [3: ]: optional scalar auxiliary channels in the order:
               view_transmittance, light_transmittance, linear_depth (normalised)
      - [...]: optional RGB normal channels (when use_normals is enabled), appended
               after the scalar auxiliary channels.

    The returned tensor has shape [B, 1, H, W] and is used to weight the L1 loss.
    """
    b, c, h, w = low_up.shape
    device = low_up.device
    dtype = low_up.dtype

    # Start with uniform weights = 1 everywhere.
    weights = torch.ones((b, 1, h, w), device=device, dtype=dtype)

    # No auxiliary channels configured -> keep uniform weighting.
    if not (
        config.use_view_transmittance
        or config.use_light_transmittance
        or config.use_linear_depth
        or config.use_normals
    ):
        return weights

    # Extract auxiliary slices based on the same ordering used when building low_up.
    cursor = 3
    scalar_components: list[torch.Tensor] = []

    if config.use_view_transmittance and cursor < c:
        scalar_components.append(low_up[:, cursor : cursor + 1])
        cursor += 1
    if config.use_light_transmittance and cursor < c:
        scalar_components.append(low_up[:, cursor : cursor + 1])
        cursor += 1
    if config.use_linear_depth and cursor < c:
        scalar_components.append(low_up[:, cursor : cursor + 1])

    # Derive a difficulty map from scalar auxiliaries: low transmittance / high depth
    # -> larger difficulty values.
    difficulty_scalar: Optional[torch.Tensor]
    if scalar_components:
        aux_stack = torch.cat(scalar_components, dim=1)  # [B, C_aux, H, W]
        aux_mean = aux_stack.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        difficulty_scalar = 1.0 - aux_mean
    else:
        difficulty_scalar = None

    # Derive a structure map from normals when present: strong non-zero normals
    # inside clouds should contribute more than zero-valued background.
    normals_strength: Optional[torch.Tensor]
    if config.use_normals and cursor < c:
        normals_end = min(cursor + 3, c)
        normals_slice = low_up[:, cursor:normals_end, :, :]
        normals_strength = torch.abs(normals_slice).mean(dim=1, keepdim=True)  # [B, 1, H, W]
    else:
        normals_strength = None

    if difficulty_scalar is None and normals_strength is None:
        return weights

    # Example scheme:
    #  - difficulty_scalar emphasises pixels with low T / high depth.
    #  - normals_strength emphasises pixels with strong normal signal (clouds).
    #  - When both are present we combine them multiplicatively so that pixels
    #    that are both "difficult" and "structured" receive the highest weight.
    if difficulty_scalar is not None and normals_strength is not None:
        combined = difficulty_scalar * normals_strength
    elif difficulty_scalar is not None:
        combined = difficulty_scalar
    else:
        combined = normals_strength

    weights = 1.0 + combined
    weights = torch.clamp(weights, 0.5, 2.0)
    return weights


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Create a 2D Gaussian window for SSIM computation.
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) * 0.5
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(0)
    kernel_2d = (kernel_1d.t() @ kernel_1d).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _compute_batch_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # Compute mean SSIM over a batch of RGB images.
    # Inputs are expected in [0, 1]; clamp to be safe.
    pred_clamped = torch.clamp(pred, 0.0, 1.0)
    target_clamped = torch.clamp(target, 0.0, 1.0)

    b, c, h, w = pred_clamped.shape
    device = pred_clamped.device
    dtype = pred_clamped.dtype

    window = _gaussian_window(window_size, sigma, c, device, dtype)

    # Constants from the original SSIM paper for L=1 images.
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    mu_x = F.conv2d(pred_clamped, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(target_clamped, window, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred_clamped * pred_clamped, window, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target_clamped * target_clamped, window, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred_clamped * target_clamped, window, padding=window_size // 2, groups=c) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_map = numerator / (denominator + 1e-12)
    ssim_value = ssim_map.mean()
    return ssim_value


def _export_loss_curves(
    step_indices: list[int],
    step_losses: list[float],
    epoch_steps: list[int],
    epoch_train_losses: list[float],
    epoch_val_losses: list[float],
    logs_dir: str,
    plot_step_loss: bool,
) -> None:
    """
    Export training loss history both as a PNG plot and as a plain text file.

    The text file stores one "step loss" pair per line so that external tools
    can easily parse and re-plot the curve.
    """
    if len(step_losses) == 0:
        return

    # Plot and save image -------------------------------------------------------
    fig = plt.figure(figsize=(8, 4.5), dpi=120)

    # Convert to NumPy for easier analysis of scales.
    step_losses_np = np.asarray(step_losses, dtype=np.float64)
    epoch_train_np = np.asarray(epoch_train_losses, dtype=np.float64)
    epoch_val_np = np.asarray(epoch_val_losses, dtype=np.float64)

    # Per-step training loss (dense curve) ----------------------------
    if plot_step_loss:
        plt.plot(step_indices, step_losses_np, label="Train step loss", color="tab:blue", linewidth=1.25)

    # Per-epoch smoothed averages for readability --------------------
    if len(epoch_steps) > 0 and len(epoch_train_losses) == len(epoch_steps):
        # Smoothed train curve (moving average over epochs).
        window = 5  # window size for epoch smoothing
        if len(epoch_train_np) >= window:
            kernel = np.ones(window, dtype=np.float64) / float(window)
            smooth_train = np.convolve(epoch_train_np, kernel, mode="same")
        plt.plot(
            epoch_steps,
                smooth_train,
                label="Train (smooth, 5-epoch)",
                color="tab:red",
                linewidth=2.0,
                alpha=1.0,
        )

    if len(epoch_steps) > 0 and len(epoch_val_losses) > 0:
        count = min(len(epoch_steps), len(epoch_val_losses))
        if count > 0:
            # Smoothed validation curve (moving average over epochs).
            window = 5
            if count >= window:
                kernel = np.ones(window, dtype=np.float64) / float(window)
                smooth_val = np.convolve(epoch_val_np[:count], kernel, mode="same")
            plt.plot(
                epoch_steps[:count],
                    smooth_val,
                    label="Val (smooth, 5-epoch)",
                color="tab:orange",
                    linewidth=2.0,
                    alpha=1.0,
            )

    # Use a logarithmic y-axis so that early large losses and late small losses
    # are visible in the same plot. For easier visual comparison across runs,
    # keep the y‑axis limits deterministic instead of data‑dependent.
    plt.yscale("log")
    # Fixed limits can be tweaked here if needed; they apply to all experiments.
    plt.ylim(bottom=5e-3, top=1e-1)

    plt.xlabel("Step")
    plt.ylabel("L1 Loss (log scale)")
    plt.title("Training / validation loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    png_path = os.path.join(logs_dir, "loss_curve.png")
    plt.savefig(png_path)
    plt.close(fig)

    # Plain text export:
    #  - Section 1: per-step training loss ("step train_loss")
    #  - Section 2: per-epoch averages ("epoch_step train_epoch_avg val_epoch_avg")
    txt_path = os.path.join(logs_dir, "loss_curve.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("# Per-step training loss\n")
        handle.write("# step train_loss\n")
        for step, loss in zip(step_indices, step_losses):
            handle.write(f"{step} {loss:.8f}\n")

        handle.write("\n# Per-epoch average losses\n")
        handle.write("# epoch_step train_epoch_avg val_epoch_avg\n")

        max_count = max(len(epoch_steps), len(epoch_train_losses), len(epoch_val_losses))
        for i in range(max_count):
            step_val = epoch_steps[i] if i < len(epoch_steps) else -1
            train_val = epoch_train_losses[i] if i < len(epoch_train_losses) else float("nan")
            val_val = epoch_val_losses[i] if i < len(epoch_val_losses) else float("nan")
            handle.write(f"{step_val} {train_val:.8f} {val_val:.8f}\n")

    print(f"Saved training loss plot: {png_path}")
    print(f"Saved training loss txt:  {txt_path}")


def _export_ssim_curve(
    epoch_val_ssim: list[float],
    logs_dir: str,
) -> None:
    # Export validation SSIM history as a PNG plot and as a plain text file.
    if len(epoch_val_ssim) == 0:
        return

    fig = plt.figure(figsize=(8, 4.5), dpi=120)

    epochs = np.arange(1, len(epoch_val_ssim) + 1, dtype=np.int32)
    ssim_np = np.asarray(epoch_val_ssim, dtype=np.float64)

    # Smoothed validation SSIM (moving average over epochs) for readability.
    window = 5
    if len(ssim_np) >= window:
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smooth_ssim = np.convolve(ssim_np, kernel, mode="same")
        plt.plot(
            epochs,
            smooth_ssim,
            label="Val SSIM (smooth, 5-epoch)",
            color="tab:purple",
            linewidth=2.0,
            alpha=1.0,
        )
    else:
        # For very short runs, fall back to plotting the raw values.
        plt.plot(
            epochs,
            ssim_np,
            label="Val SSIM",
            color="tab:purple",
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

    plt.ylim(0.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM per epoch")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()

    png_path = os.path.join(logs_dir, "ssim_curve.png")
    plt.savefig(png_path)
    plt.close(fig)

    txt_path = os.path.join(logs_dir, "ssim_curve.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("# Per-epoch validation SSIM\n")
        handle.write("# epoch ssim\n")
        for idx, value in enumerate(epoch_val_ssim, start=1):
            handle.write(f"{idx} {value:.6f}\n")

    print(f"Saved validation SSIM plot: {png_path}")
    print(f"Saved validation SSIM txt:  {txt_path}")

def train_unet(config: TrainConfig) -> None:
    # ----------------- Setup -----------------
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = _select_device(config.device)

    _ensure_dirs(config.output_dir)
    ckpt_dir = os.path.join(config.output_dir, "checkpoints")
    logs_dir = os.path.join(config.output_dir, "logs")

    # ----------------- Data -----------------
    ds_conf = CloudPairDatasetConfig(
        root_dir=config.data_dir,
        crop_size=config.crop_size,
        limit_pairs=config.limit_pairs,
        upsample_mode="bicubic",
        use_view_transmittance=config.use_view_transmittance,
        use_light_transmittance=config.use_light_transmittance,
        use_linear_depth=config.use_linear_depth,
        use_normals=config.use_normals,
        depth_normalization_max=config.depth_normalization_max,
    )
    dataset = CloudPairDataset(ds_conf)

    # Split underlying dataset indices into train/val/test so that we can
    # monitor generalisation and later run inference on the exact same splits.
    num_samples = len(dataset)
    train_indices, val_indices, test_indices = compute_dataset_index_splits(
        num_samples=num_samples,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        split_seed=config.split_seed,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if len(val_indices) > 0 else None
    test_dataset = Subset(dataset, test_indices) if len(test_indices) > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if val_dataset is not None
        else None
    )

    train_num_samples = len(train_dataset)
    train_num_batches = (train_num_samples + config.batch_size - 1) // max(1, config.batch_size)
    total_steps = train_num_batches * max(1, config.epochs)  # total training steps across all epochs
    print("===============================================================")
    print(
        f"Dataset: {num_samples} pairs "
        f"(train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}) | "
        f"batch_size={config.batch_size} | train_batches/epoch={train_num_batches}"
    )
    print(f"Crop size: {config.crop_size} | Upsample: {ds_conf.upsample_mode}")
    print(f"Output: {config.output_dir} | Checkpoints: {ckpt_dir}")
    print("===============================================================")

    # ----------------- Model & Optim -----------------
    # Build a U‑Net that takes an N‑channel input at high resolution
    # (upsampled low-res frame plus optional auxiliary channels) and predicts
    # a refined 3‑channel RGB output.
    # learn_residual=True makes the model add a learned residual on top of the
    # upsampled input, which stabilizes training for super‑resolution/refinement.
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

    model = UNet(
        UNetConfig(
            in_channels=in_channels,
            out_channels=3,
            base_channels=config.model_base_channels,
            bilinear=config.model_bilinear,
            learn_residual=config.model_learn_residual,
        )
    )
    model = model.to(device)

    # Optimizer and loss
    # - AdamW: Adam with decoupled weight decay (good general default)
    # - L1 loss: robust to outliers; favors sharper results than L2
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Per‑pixel L1 loss; we apply optional weighting ourselves below.
    criterion = torch.nn.L1Loss(reduction="none")

    num_params = sum(p.numel() for p in model.parameters())

    # Model summary: channels per level, depth, conv/bn counts
    b = config.model_base_channels
    factor = 2 if config.model_bilinear else 1
    channels_per_level = [b, b * 2, b * 4, b * 8, (b * 16) // factor]
    depth_levels = len(channels_per_level)
    num_convs = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    num_bns = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    print(
        "Model: UNet | "
        f"params={num_params/1e6:.2f}M | "
        f"depth={depth_levels} | "
        f"levels={channels_per_level} | "
        f"convs={num_convs} | bns={num_bns} | "
        f"in_ch={in_channels} | lr={config.learning_rate:.2e} | device={device.type} | "
        f"bilinear_up={config.model_bilinear} | residual={config.model_learn_residual}"
    )

    # Global step counter and arrays for plotting loss after training
    global_step = 0
    step_indices: list[int] = []
    step_losses: list[float] = []
    epoch_steps: list[int] = []
    epoch_train_losses: list[float] = []
    epoch_val_losses: list[float] = []
    epoch_val_ssim: list[float] = []

    # Simple ETA tracking between log events
    last_log_step: Optional[int] = None
    last_log_time: Optional[float] = None

    # ----------------- Training Loop -----------------
    for epoch in range(1, config.epochs + 1):
        model.train()  # enable dropout/batchnorm update if present
        epoch_loss = 0.0  # running sum to compute average epoch loss
        print(f"==> Epoch {epoch}/{config.epochs}")

        for batch_idx, batch in enumerate(train_loader):
            # Fetch batch
            # low_up: upsampled low‑res to high‑res input  [B, C_in, H, W]
            # high:   ground truth high‑res target         [B, 3, H, W]
            low_up = batch["low_up"].to(device, non_blocking=True)   # [B, C_in, H, W]
            high = batch["high"].to(device, non_blocking=True)       # [B, 3, H, W]

            # Forward pass: predict refined image (residual added inside the model)
            pred = model(low_up)

            # Compute loss and update weights
            per_pixel_loss = criterion(pred, high)  # [B, 3, H, W]

            if config.use_auxiliary_in_loss:
                # Derive a weighting map from auxiliary channels present in low_up.
                weights = _compute_auxiliary_weights(low_up, config)  # [B, 1, H, W]
                weighted = per_pixel_loss * weights
                loss = weighted.mean()
            else:
                # Plain unweighted L1 loss on RGB.
                loss = per_pixel_loss.mean()

            optimizer.zero_grad(set_to_none=True) # zero_grad: clear old gradients on parameters before accumulating new ones.
            loss.backward() # backward: run backpropagation to compute gradients d(loss)/d(parameter).
            optimizer.step() # step: apply the optimizer update rule (AdamW) using the computed gradients.

            # Bookkeeping
            epoch_loss += loss.item()
            global_step += 1
            step_indices.append(global_step)
            step_losses.append(float(loss.item()))

            # Logging
            epoch_processed = min((batch_idx + 1) * config.batch_size, train_num_samples)
            if global_step % max(1, config.log_every_n_steps) == 0 or batch_idx == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                # ------------------------------------------------------------
                # ETA computation based on time between the last and current
                # log events. This keeps the estimate local in time so it
                # adapts if training speed changes.
                # ------------------------------------------------------------
                now = time.perf_counter()
                eta_str = "ETA --:--:--"

                if last_log_step is not None and last_log_time is not None:
                    step_delta = global_step - last_log_step
                    time_delta = now - last_log_time

                    if step_delta > 0 and time_delta > 0.0:
                        steps_per_second = step_delta / time_delta
                        remaining_steps = max(0, total_steps - global_step)

                        if steps_per_second > 0.0:
                            remaining_seconds = remaining_steps / steps_per_second

                            if np.isfinite(remaining_seconds) and remaining_seconds >= 0.0:
                                hours = int(remaining_seconds // 3600)
                                minutes = int((remaining_seconds % 3600) // 60)
                                seconds = int(remaining_seconds % 60)
                                eta_str = f"ETA {hours:02d}:{minutes:02d}:{seconds:02d}"

                last_log_step = global_step
                last_log_time = now

                print(
                    f"[epoch {epoch}/{config.epochs}] "
                    f"batch {batch_idx + 1}/{train_num_batches} "
                    f"(step {global_step}) | "
                    f"samples {epoch_processed}/{num_samples} | "
                    f"loss {loss.item():.6f} | avg {(epoch_loss / (batch_idx + 1)):.6f} | "
                    f"lr {current_lr:.2e} | {eta_str}"
                )

            # Periodically save a checkpoint (disabled when we only want the
            # final epoch checkpoint to avoid filling disk with intermediates).
            if (not config.save_only_last_epoch) and config.save_every_n_steps > 0:
                if global_step % config.save_every_n_steps == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"unet_step_{global_step}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")

        # End‑of‑epoch reporting on training set
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{config.epochs} - train avg loss: {avg_loss:.6f}")

        # Record epoch-level training statistics at the step reached at end of epoch.
        epoch_steps.append(global_step)
        epoch_train_losses.append(avg_loss)

        # ----------------- Validation Loop -----------------
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_ssim_sum = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_low_up = val_batch["low_up"].to(device, non_blocking=True)
                    val_high = val_batch["high"].to(device, non_blocking=True)
                    val_pred = model(val_low_up)

                    val_per_pixel = criterion(val_pred, val_high)  # [B, 3, H, W]
                    if config.use_auxiliary_in_loss:
                        val_weights = _compute_auxiliary_weights(val_low_up, config)  # [B, 1, H, W]
                        val_weighted = val_per_pixel * val_weights
                        val_loss = val_weighted.mean()
                    else:
                        val_loss = val_per_pixel.mean()

                    val_loss_sum += float(val_loss.item())

                    # Compute SSIM for this batch and accumulate.
                    batch_ssim = _compute_batch_ssim(val_pred, val_high)
                    val_ssim_sum += float(batch_ssim.item())

            val_avg_loss = val_loss_sum / max(1, len(val_loader))
            val_avg_ssim = val_ssim_sum / max(1, len(val_loader))
            epoch_val_losses.append(val_avg_loss)
            epoch_val_ssim.append(val_avg_ssim)
            print(
                f"Epoch {epoch}/{config.epochs} - val   avg loss: {val_avg_loss:.6f} | "
                f"val SSIM: {val_avg_ssim:.4f}"
            )

            # Append SSIM history to a simple text log for later analysis.
            ssim_log_path = os.path.join(logs_dir, "val_ssim.txt")
            with open(ssim_log_path, "a", encoding="utf-8") as ssim_handle:
                ssim_handle.write(f"{epoch} {val_avg_ssim:.6f}\n")

        # Save checkpoint at end of epoch.
        # - Default: save every epoch.
        # - save_epoch_stride>1: save only every Nth epoch plus the final epoch.
        # - save_only_last_epoch=True: only write a checkpoint for the final epoch.
        save_epoch_ckpt = False
        if not config.save_only_last_epoch:
            if config.save_epoch_stride <= 0:
                # Non‑positive stride interpreted as "save every epoch".
                save_epoch_ckpt = True
            elif (epoch % config.save_epoch_stride) == 0:
                save_epoch_ckpt = True
        if epoch == config.epochs:
            save_epoch_ckpt = True

        if save_epoch_ckpt:
            epoch_ckpt_path = os.path.join(ckpt_dir, f"unet_epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_ckpt_path)
            print(f"Saved epoch checkpoint: {epoch_ckpt_path}")

        # Periodic export of loss history (PNG + TXT). Always export at the final epoch,
        # and also every config.export_every_n_epochs when that value is > 0.
        should_export = (epoch == config.epochs)
        if config.export_every_n_epochs > 0 and (epoch % config.export_every_n_epochs == 0):
            should_export = True

        if should_export:
            _export_loss_curves(
                step_indices=step_indices,
                step_losses=step_losses,
                epoch_steps=epoch_steps,
                epoch_train_losses=epoch_train_losses,
                epoch_val_losses=epoch_val_losses,
                logs_dir=logs_dir,
                plot_step_loss=config.plot_step_loss,
            )
            _export_ssim_curve(
                epoch_val_ssim=epoch_val_ssim,
                logs_dir=logs_dir,
            )
