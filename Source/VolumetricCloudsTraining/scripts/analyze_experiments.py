import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ExperimentMetrics:
    """Container for parsed metrics for a single experiment."""

    family: str                         # e.g. "exp1_buffers_general"
    name: str                           # e.g. "rgb_plus_viewT_normals"
    num_epochs: int

    # Losses
    best_val_loss: float
    best_val_loss_epoch: int            # 1-based epoch index
    final_val_loss: float

    # Optional SSIM
    best_ssim: Optional[float]
    best_ssim_epoch: Optional[int]      # 1-based epoch index
    final_ssim: Optional[float]

    # SSIM trajectory helpers
    ssim_95pct_epoch: Optional[int]     # First epoch where SSIM >= 95% of best, if any
    ssim_late_gain: Optional[float]     # SSIM(final) - SSIM at 60% of training, if available

    # Simple overfitting heuristic
    val_overfits: bool                  # True if final val loss is noticeably worse than best


def _parse_loss_curve(loss_txt_path: str) -> Tuple[List[float], List[float]]:
    """Parse per-epoch averages from a loss_curve.txt file.

    Returns:
        train_epoch_avgs, val_epoch_avgs
    """
    train_epoch_avgs: List[float] = []
    val_epoch_avgs: List[float] = []

    in_epoch_section: bool = False

    with open(loss_txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line: str = raw_line.strip()

            # --------------------------------------------------------------
            # Skip empty / comment lines until we hit the epoch section
            # --------------------------------------------------------------
            if not line:
                continue

            if line.startswith("# Per-epoch average losses"):
                in_epoch_section = True
                continue

            if not in_epoch_section:
                continue

            if line.startswith("#"):
                continue

            # --------------------------------------------------------------
            # Expect: "<epoch_step> <train_epoch_avg> <val_epoch_avg>"
            # We treat the row index as the epoch number (1-based).
            # --------------------------------------------------------------
            parts: List[str] = line.split()
            if len(parts) != 3:
                # Be robust to any stray formatting
                continue

            _, train_str, val_str = parts

            try:
                train_val: float = float(train_str)
                val_val: float = float(val_str)
            except ValueError:
                continue

            train_epoch_avgs.append(train_val)
            val_epoch_avgs.append(val_val)

    return train_epoch_avgs, val_epoch_avgs


def _parse_ssim_curve(ssim_txt_path: str) -> List[float]:
    """Parse per-epoch validation SSIM from ssim_curve.txt."""
    ssim_values: List[float] = []

    if not os.path.isfile(ssim_txt_path):
        return ssim_values

    with open(ssim_txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line: str = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts: List[str] = line.split()
            if len(parts) != 2:
                continue

            _, ssim_str = parts

            try:
                ssim_val: float = float(ssim_str)
            except ValueError:
                continue

            ssim_values.append(ssim_val)

    return ssim_values


def _summarize_experiment(family: str, name: str, loss_txt_path: str, ssim_txt_path: Optional[str]) -> Optional[ExperimentMetrics]:
    """Parse logs for one experiment and build a metrics summary."""
    train_epoch_avgs, val_epoch_avgs = _parse_loss_curve(loss_txt_path)

    if not val_epoch_avgs:
        # Nothing to summarize
        return None

    num_epochs: int = len(val_epoch_avgs)

    # --------------------------------------------------------------
    # Loss statistics
    # --------------------------------------------------------------
    best_val_loss: float = min(val_epoch_avgs)
    best_val_loss_epoch: int = 1 + val_epoch_avgs.index(best_val_loss)
    final_val_loss: float = val_epoch_avgs[-1]

    # Simple overfitting detection: flag if final val loss is > 5% worse than the best
    val_overfits: bool = final_val_loss > best_val_loss * 1.05

    # --------------------------------------------------------------
    # SSIM statistics (optional)
    # --------------------------------------------------------------
    best_ssim: Optional[float] = None
    best_ssim_epoch: Optional[int] = None
    final_ssim: Optional[float] = None
    ssim_95pct_epoch: Optional[int] = None
    ssim_late_gain: Optional[float] = None

    if ssim_txt_path is not None and os.path.isfile(ssim_txt_path):
        ssim_values: List[float] = _parse_ssim_curve(ssim_txt_path)
        if ssim_values:
            best_ssim = max(ssim_values)
            best_ssim_epoch = 1 + ssim_values.index(best_ssim)
            final_ssim = ssim_values[-1]

            # ------------------------------------------------------
            # Epoch where we first reach 95% of the best SSIM
            # ------------------------------------------------------
            threshold_95: float = best_ssim * 0.95
            for idx, ssim_val in enumerate(ssim_values):
                if ssim_val >= threshold_95:
                    ssim_95pct_epoch = idx + 1
                    break

            # ------------------------------------------------------
            # SSIM gain over the last 40% of training
            # ------------------------------------------------------
            late_start_epoch: int = max(1, int(0.6 * float(num_epochs)))
            if late_start_epoch <= len(ssim_values):
                ssim_at_late_start: float = ssim_values[late_start_epoch - 1]
                ssim_late_gain = final_ssim - ssim_at_late_start

    return ExperimentMetrics(
        family=family,
        name=name,
        num_epochs=num_epochs,
        best_val_loss=best_val_loss,
        best_val_loss_epoch=best_val_loss_epoch,
        final_val_loss=final_val_loss,
        best_ssim=best_ssim,
        best_ssim_epoch=best_ssim_epoch,
        final_ssim=final_ssim,
        ssim_95pct_epoch=ssim_95pct_epoch,
        ssim_late_gain=ssim_late_gain,
        val_overfits=val_overfits,
    )


def _find_experiments(outputs_root: str) -> List[ExperimentMetrics]:
    """Walk outputs/*/experiments/*/logs and collect summaries."""
    all_metrics: List[ExperimentMetrics] = []

    if not os.path.isdir(outputs_root):
        return all_metrics

    for family in sorted(os.listdir(outputs_root)):
        family_dir: str = os.path.join(outputs_root, family)
        if not os.path.isdir(family_dir):
            continue

        experiments_root: str = os.path.join(family_dir, "experiments")
        if not os.path.isdir(experiments_root):
            continue

        for exp_name in sorted(os.listdir(experiments_root)):
            exp_dir: str = os.path.join(experiments_root, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            logs_dir: str = os.path.join(exp_dir, "logs")
            if not os.path.isdir(logs_dir):
                continue

            loss_txt: str = os.path.join(logs_dir, "loss_curve.txt")
            if not os.path.isfile(loss_txt):
                continue

            ssim_txt: str = os.path.join(logs_dir, "ssim_curve.txt")
            if not os.path.isfile(ssim_txt):
                ssim_txt = None

            metrics: Optional[ExperimentMetrics] = _summarize_experiment(
                family=family,
                name=exp_name,
                loss_txt_path=loss_txt,
                ssim_txt_path=ssim_txt,
            )

            if metrics is not None:
                all_metrics.append(metrics)

    return all_metrics


def _format_float(value: float, decimals: int = 6) -> str:
    """Format float with a fixed number of decimals."""
    fmt: str = f"{{:.{decimals}f}}"
    return fmt.format(value)


def print_summary(outputs_root: str = "outputs") -> None:
    """Print a human-readable summary of all experiments under outputs/.

    For each experiment, we show:
      - Number of epochs
      - Best validation loss and epoch
      - Final validation loss
      - (If available) best validation SSIM and epoch
      - (If available) final validation SSIM
      - (If available) a couple of simple SSIM trajectory indicators
      - A simple overfitting flag based on validation loss
    """
    metrics_list: List[ExperimentMetrics] = _find_experiments(outputs_root)

    if not metrics_list:
        print(f"No experiments found under '{outputs_root}'.")
        return

    # Group by family for readability
    metrics_list.sort(key=lambda m: (m.family, m.name))

    current_family: Optional[str] = None

    for m in metrics_list:
        if m.family != current_family:
            current_family = m.family
            print()
            print("=" * 80)
            print(f"Experiment family: {current_family}")
            print("=" * 80)

        print(f"- {m.name}")
        print(f"  epochs: {m.num_epochs}")

        # Loss summary
        print(
            "  val L1: best="
            f"{_format_float(m.best_val_loss, 6)} (epoch {m.best_val_loss_epoch}), "
            f"final={_format_float(m.final_val_loss, 6)}"
        )

        if m.val_overfits:
            print("  note : validation loss worsened after its best epoch (possible overfitting).")

        # SSIM summary (if present)
        if m.best_ssim is not None and m.best_ssim_epoch is not None and m.final_ssim is not None:
            print(
                "  val SSIM: best="
                f"{_format_float(m.best_ssim, 6)} (epoch {m.best_ssim_epoch}), "
                f"final={_format_float(m.final_ssim, 6)}"
            )

            # Optional SSIM trajectory details
            if m.ssim_95pct_epoch is not None:
                print(f"  traj : 95% of best SSIM reached by epoch {m.ssim_95pct_epoch}.")

            if m.ssim_late_gain is not None:
                gain_str: str = _format_float(m.ssim_late_gain, 4)
                print(f"         ΔSSIM over last 40% of training: {gain_str}")

        print()


if __name__ == "__main__":
    # Run from the VolumetricCloudsTraining root:
    #   uv run python scripts/analyze_experiments.py
    # or:
    #   python scripts/analyze_experiments.py
    print_summary()


