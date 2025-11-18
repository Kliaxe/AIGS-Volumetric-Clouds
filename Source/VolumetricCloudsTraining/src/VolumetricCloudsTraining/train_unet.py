import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib


# Use a non-interactive backend for environments without a display (e.g., CI/headless)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader

from .dataset_cloud_pairs import CloudPairDataset, CloudPairDatasetConfig
from .models.unet import UNet, UNetConfig
from .pfm_io import write_pfm


# ================================================================
#                          TRAINING CONFIG
# ================================================================


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "Outputs"
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-4
    crop_size: Optional[int] = 256
    limit_pairs: Optional[int] = None
    num_workers: int = 0  # Windows default safer
    device: str = "auto"  # 'auto' | 'cuda' | 'cpu'
    seed: int = 42
    save_every_n_steps: int = 200
    log_every_n_steps: int = 20
    # Model configuration --------------------------------------------
    model_base_channels: int = 64
    model_bilinear: bool = True
    model_learn_residual: bool = True


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
    )
    dataset = CloudPairDataset(ds_conf)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_samples = len(dataset)
    num_batches = (num_samples + config.batch_size - 1) // max(1, config.batch_size)
    print("===============================================================")
    print(f"Dataset: {num_samples} pairs | batch_size={config.batch_size} | batches/epoch={num_batches}")
    print(f"Crop size: {config.crop_size} | Upsample: {ds_conf.upsample_mode}")
    print(f"Output: {config.output_dir} | Checkpoints: {ckpt_dir}")
    print("===============================================================")

    # ----------------- Model & Optim -----------------
    # Build a U‑Net that takes a 3‑channel RGB input at high resolution
    # (upsampled low-res frame) and predicts a refined 3‑channel RGB output.
    # learn_residual=True makes the model add a learned residual on top of the
    # upsampled input, which stabilizes training for super‑resolution/refinement.
    model = UNet(
        UNetConfig(
            in_channels=3,
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
    criterion = torch.nn.L1Loss()

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
        f"lr={config.learning_rate:.2e} | device={device.type} | "
        f"bilinear_up={config.model_bilinear} | residual={config.model_learn_residual}"
    )

    # Global step counter and arrays for plotting loss after training
    global_step = 0
    step_indices: list[int] = []
    step_losses: list[float] = []

    # ----------------- Training Loop -----------------
    for epoch in range(1, config.epochs + 1):
        model.train()  # enable dropout/batchnorm update if present
        epoch_loss = 0.0  # running sum to compute average epoch loss
        print(f"==> Epoch {epoch}/{config.epochs}")

        for batch_idx, batch in enumerate(loader):
            # Fetch batch
            # low_up: upsampled low‑res to high‑res input  [B, 3, H, W]
            # high:   ground truth high‑res target         [B, 3, H, W]
            low_up = batch["low_up"].to(device, non_blocking=True)   # [B, 3, H, W]
            high = batch["high"].to(device, non_blocking=True)       # [B, 3, H, W]

            # Forward pass: predict refined image (residual added inside the model)
            pred = model(low_up)

            # Compute loss and update weights
            loss = criterion(pred, high) # loss = L1(prediction, target). Scalar measuring current batch error.
            optimizer.zero_grad(set_to_none=True) # zero_grad: clear old gradients on parameters before accumulating new ones.
            loss.backward() # backward: run backpropagation to compute gradients d(loss)/d(parameter).
            optimizer.step() # step: apply the optimizer update rule (AdamW) using the computed gradients.

            # Bookkeeping
            epoch_loss += loss.item()
            global_step += 1
            step_indices.append(global_step)
            step_losses.append(float(loss.item()))

            # Logging
            epoch_processed = min((batch_idx + 1) * config.batch_size, num_samples)
            if global_step % max(1, config.log_every_n_steps) == 0 or batch_idx == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[epoch {epoch}/{config.epochs}] "
                    f"batch {batch_idx + 1}/{num_batches} "
                    f"(step {global_step}) | "
                    f"samples {epoch_processed}/{num_samples} | "
                    f"loss {loss.item():.6f} | avg {(epoch_loss / (batch_idx + 1)):.6f} | "
                    f"lr {current_lr:.2e}"
                )

            # Periodically save a checkpoint
            if global_step % config.save_every_n_steps == 0:
                ckpt_path = os.path.join(ckpt_dir, f"unet_step_{global_step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # End‑of‑epoch reporting
        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{config.epochs} - avg loss: {avg_loss:.6f}")

        # Save checkpoint at end of epoch (always have a recent stable checkpoint)
        epoch_ckpt_path = os.path.join(ckpt_dir, f"unet_epoch_{epoch}.pt")
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"Saved epoch checkpoint: {epoch_ckpt_path}")

    # ----------------- Post Training: Save Loss Curve & CSV -----------------
    if len(step_losses) > 0:
        # Save plot
        fig = plt.figure(figsize=(8, 4.5), dpi=120)
        plt.plot(step_indices, step_losses, label="Step Loss", color="tab:blue", linewidth=1.25)
        plt.xlabel("Step")
        plt.ylabel("L1 Loss")
        plt.title("Training Loss")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend(loc="best")
        plt.tight_layout()
        png_path = os.path.join(logs_dir, "loss_curve.png")
        plt.savefig(png_path)
        plt.close(fig)
        print(f"Saved training loss plot: {png_path}")
