import os
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend suitable for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_experiments import _parse_loss_curve, _parse_ssim_curve


# Resolve a shared Figures/ directory at the project root, regardless of the
# current working directory when this script is invoked.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "Figures")


def _ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _save_figure(fig: matplotlib.figure.Figure, filename: str) -> None:
    """Save a matplotlib figure as a PNG into the shared Figures/ directory."""
    _ensure_dir(FIGURES_DIR)
    png_path = os.path.join(FIGURES_DIR, f"{filename}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {png_path}")


def _moving_average(values: List[float], window: int = 5) -> List[float]:
    """Simple centered moving average for 1D sequences.

    For each index i, we average over values in [i - r, i + r] where r = window // 2,
    clamped to the valid range. This keeps the sequence length unchanged while
    smoothing out short-term noise.
    """
    if window <= 1 or len(values) == 0:
        return list(values)

    radius = window // 2
    n = len(values)
    smoothed: List[float] = []
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        segment = values[start:end]
        smoothed.append(sum(segment) / float(len(segment)))
    return smoothed


def plot_exp0_overfit(outputs_root: str = "outputs") -> None:
    """Plot loss and SSIM curves for Experiment 0 (single-view overfitting)."""
    family = "exp0_overfit_singleview"
    experiments = {
        "overfit_pairs_2": "2 pairs",
        "overfit_pairs_4": "4 pairs",
    }

    base_dir = os.path.join(outputs_root, family, "experiments")

    # Loss vs epoch (validation only, training loss is typically much lower and
    # visually less informative for these tiny overfitting runs).
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))

    for exp_name, label in experiments.items():
        logs_dir = os.path.join(base_dir, exp_name, "logs")
        loss_txt = os.path.join(logs_dir, "loss_curve.txt")
        if not os.path.isfile(loss_txt):
            continue

        train_epoch, val_epoch = _parse_loss_curve(loss_txt)
        if not val_epoch:
            continue

        # Smooth over a small temporal window to reduce visual spikiness on
        # tiny overfitting datasets.
        val_smooth = _moving_average(val_epoch, window=5)
        epochs = list(range(1, len(val_smooth) + 1))

        ax_loss.plot(
            epochs,
            val_smooth,
            linestyle="-",
            linewidth=1.5,
            label=label,
        )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation L1 loss (epoch average)")
    ax_loss.set_title("Experiment 0 (single-view overfitting): validation loss vs epoch")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(loc="best")

    _save_figure(fig_loss, "exp0_overfit_loss")

    # Validation SSIM vs epoch
    fig_ssim, ax_ssim = plt.subplots(figsize=(6, 4))

    for exp_name, label in experiments.items():
        logs_dir = os.path.join(base_dir, exp_name, "logs")
        ssim_txt = os.path.join(logs_dir, "ssim_curve.txt")
        if not os.path.isfile(ssim_txt):
            continue

        ssim_values = _parse_ssim_curve(ssim_txt)
        if not ssim_values:
            continue

        ssim_smooth = _moving_average(ssim_values, window=5)
        epochs = list(range(1, len(ssim_smooth) + 1))
        ax_ssim.plot(
            epochs,
            ssim_smooth,
            linewidth=1.5,
            label=label,
        )

    ax_ssim.set_xlabel("Epoch")
    ax_ssim.set_ylabel("Validation SSIM")
    ax_ssim.set_title("Experiment 0 (single-view overfitting): SSIM vs epoch")
    ax_ssim.set_ylim(0.0, 1.0)
    ax_ssim.grid(True, linestyle="--", alpha=0.4)
    ax_ssim.legend(loc="best")

    _save_figure(fig_ssim, "exp0_overfit_ssim")


def plot_exp1_buffers(outputs_root: str = "outputs") -> None:
    """Plot buffer ablation SSIM curves for Experiment 1."""
    # General-view: RGB vs richest configuration
    family_general = "exp1_buffers_general"
    base_general = os.path.join(outputs_root, family_general, "experiments")
    general_experiments = {
        "rgb_only": "RGB only",
        "rgb_plus_viewT_normals": "RGB + viewT + normals",
    }

    fig_g, ax_g = plt.subplots(figsize=(6, 4))
    min_ssim_g = 1.0
    max_ssim_g = 0.0
    for exp_name, label in general_experiments.items():
        logs_dir = os.path.join(base_general, exp_name, "logs")
        ssim_txt = os.path.join(logs_dir, "ssim_curve.txt")
        if not os.path.isfile(ssim_txt):
            continue

        ssim_values = _parse_ssim_curve(ssim_txt)
        if not ssim_values:
            continue

        min_ssim_g = min(min_ssim_g, min(ssim_values))
        max_ssim_g = max(max_ssim_g, max(ssim_values))

        epochs = list(range(1, len(ssim_values) + 1))
        ax_g.plot(
            epochs,
            ssim_values,
            linewidth=1.5,
            label=label,
        )

    ax_g.set_xlabel("Epoch")
    ax_g.set_ylabel("Validation SSIM")
    ax_g.set_title("Experiment 1 (general-view): RGB vs rich buffers")
    if max_ssim_g > 0.0:
        # Zoom the y-axis to the region where the curves actually live, to avoid
        # unused white space while still keeping SSIM in a familiar linear scale.
        margin = 0.02
        y_min = max(0.0, min_ssim_g - margin)
        y_max = min(1.0, max_ssim_g + margin)
        # For these runs, nothing meaningful happens below ~0.6 SSIM, so clamp
        # to that range for readability.
        y_min = max(0.6, y_min)
        ax_g.set_ylim(y_min, y_max)
    ax_g.grid(True, linestyle="--", alpha=0.4)
    ax_g.legend(loc="best")

    _save_figure(fig_g, "exp1_buffers_general_ssim")

    # Single-view: RGB vs richest configuration
    family_single = "exp1_buffers_singleview"
    base_single = os.path.join(outputs_root, family_single, "experiments")
    single_experiments = {
        "rgb_only": "RGB only",
        "rgb_plus_viewT_normals": "RGB + viewT + normals",
    }

    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    min_ssim_s = 1.0
    max_ssim_s = 0.0
    for exp_name, label in single_experiments.items():
        logs_dir = os.path.join(base_single, exp_name, "logs")
        ssim_txt = os.path.join(logs_dir, "ssim_curve.txt")
        if not os.path.isfile(ssim_txt):
            continue

        ssim_values = _parse_ssim_curve(ssim_txt)
        if not ssim_values:
            continue

        min_ssim_s = min(min_ssim_s, min(ssim_values))
        max_ssim_s = max(max_ssim_s, max(ssim_values))

        epochs = list(range(1, len(ssim_values) + 1))
        ax_s.plot(
            epochs,
            ssim_values,
            linewidth=1.5,
            label=label,
        )

    ax_s.set_xlabel("Epoch")
    ax_s.set_ylabel("Validation SSIM")
    ax_s.set_title("Experiment 1 (single-view): RGB vs rich buffers")
    if max_ssim_s > 0.0:
        margin = 0.02
        y_min = max(0.0, min_ssim_s - margin)
        y_max = min(1.0, max_ssim_s + margin)
        y_min = max(0.6, y_min)
        ax_s.set_ylim(y_min, y_max)
    ax_s.grid(True, linestyle="--", alpha=0.4)
    ax_s.legend(loc="best")

    _save_figure(fig_s, "exp1_buffers_singleview_ssim")


def _collect_best_ssim_for_sizes(
    base_dir: str,
    rgb_prefix: str,
    rich_prefix: str,
    sizes: List[int],
) -> Tuple[List[float], List[float]]:
    """Helper to compute best SSIM per dataset size for RGB and rich configs."""
    best_rgb: List[float] = []
    best_rich: List[float] = []

    for n in sizes:
        rgb_name = f"{rgb_prefix}_{n}"
        rich_name = f"{rich_prefix}_{n}"

        # RGB
        rgb_logs = os.path.join(base_dir, rgb_name, "logs")
        rgb_ssim_path = os.path.join(rgb_logs, "ssim_curve.txt")
        if os.path.isfile(rgb_ssim_path):
            rgb_vals = _parse_ssim_curve(rgb_ssim_path)
            best_rgb.append(max(rgb_vals) if rgb_vals else float("nan"))
        else:
            best_rgb.append(float("nan"))

        # Rich
        rich_logs = os.path.join(base_dir, rich_name, "logs")
        rich_ssim_path = os.path.join(rich_logs, "ssim_curve.txt")
        if os.path.isfile(rich_ssim_path):
            rich_vals = _parse_ssim_curve(rich_ssim_path)
            best_rich.append(max(rich_vals) if rich_vals else float("nan"))
        else:
            best_rich.append(float("nan"))

    return best_rgb, best_rich


def plot_exp2_data_efficiency(outputs_root: str = "outputs") -> None:
    """Plot data efficiency curves for Experiment 2."""
    # General-view: SSIM vs epoch for selected configs, and best-SSIM vs size.
    family_general = "exp2_data_efficiency_general"
    base_general = os.path.join(outputs_root, family_general, "experiments")

    # SSIM vs epoch: RGB and rich at 100 pairs and at the full dataset size.
    # Note: The experiment folder names use the suffix "_1000" for the full dataset,
    # which in our captures corresponds to 1024 pairs.
    general_curve_experiments: Dict[str, str] = {
        "rgb_pairs_100": "RGB, 100 pairs",
        "rgb_pairs_1000": "RGB, 1024 pairs",
        "rich_pairs_100": "Rich, 100 pairs",
        "rich_pairs_1000": "Rich, 1024 pairs",
    }

    fig_g_curve, ax_g_curve = plt.subplots(figsize=(6, 4))
    min_ssim_curve = 1.0
    max_ssim_curve = 0.0
    for exp_name, label in general_curve_experiments.items():
        logs_dir = os.path.join(base_general, exp_name, "logs")
        ssim_txt = os.path.join(logs_dir, "ssim_curve.txt")
        if not os.path.isfile(ssim_txt):
            continue

        ssim_values = _parse_ssim_curve(ssim_txt)
        if not ssim_values:
            continue

        # Smooth over a small temporal window to reduce visual spikiness while
        # preserving the overall learning trends.
        ssim_smooth = _moving_average(ssim_values, window=5)
        min_ssim_curve = min(min_ssim_curve, min(ssim_smooth))
        max_ssim_curve = max(max_ssim_curve, max(ssim_smooth))

        epochs = list(range(1, len(ssim_smooth) + 1))
        ax_g_curve.plot(
            epochs,
            ssim_smooth,
            linewidth=1.0,
            label=label,
        )

    ax_g_curve.set_xlabel("Epoch")
    ax_g_curve.set_ylabel("Validation SSIM")
    ax_g_curve.set_title("Experiment 2 (general-view): SSIM vs epoch")
    if max_ssim_curve > 0.0:
        margin = 0.02
        y_min = max(0.0, min_ssim_curve - margin)
        y_max = min(1.0, max_ssim_curve + margin)
        # Clamp to the interesting range; nothing happens below ~0.6.
        y_min = max(0.6, y_min)
        ax_g_curve.set_ylim(y_min, y_max)
    ax_g_curve.grid(True, linestyle="--", alpha=0.4)
    ax_g_curve.legend(loc="best")

    _save_figure(fig_g_curve, "exp2_general_ssim_vs_epoch")

    # Best SSIM vs dataset size (general-view).
    # Folder names encode "1000" for the full dataset; we plot it at 1024 pairs
    # to match the actual capture count.
    sizes_folder = [100, 300, 1000]
    sizes_plot = [100, 300, 1024]

    best_rgb_g, best_rich_g = _collect_best_ssim_for_sizes(
        base_dir=base_general,
        rgb_prefix="rgb_pairs",
        rich_prefix="rich_pairs",
        sizes=sizes_folder,
    )

    fig_g_size, ax_g_size = plt.subplots(figsize=(6, 4))
    ax_g_size.plot(
        sizes_plot,
        best_rgb_g,
        marker="o",
        linewidth=1.5,
        label="RGB",
    )
    ax_g_size.plot(
        sizes_plot,
        best_rich_g,
        marker="s",
        linewidth=1.5,
        label="Rich",
    )

    ax_g_size.set_xlabel("Training pairs")
    ax_g_size.set_ylabel("Best validation SSIM")
    ax_g_size.set_title("Experiment 2 (general-view): data efficiency")
    ax_g_size.set_xticks(sizes_plot)
    # Zoom y-axis to where the best-SSIM points lie; for these experiments,
    # all interesting variation is above ~0.8.
    finite_vals_g = [v for v in (best_rgb_g + best_rich_g) if v == v]
    if finite_vals_g:
        margin = 0.005
        y_min = max(0.0, min(finite_vals_g) - margin)
        y_max = min(1.0, max(finite_vals_g) + margin)
        y_min = max(0.8, y_min)
        ax_g_size.set_ylim(y_min, y_max)
    ax_g_size.grid(True, linestyle="--", alpha=0.4)
    ax_g_size.legend(loc="best")

    _save_figure(fig_g_size, "exp2_general_best_ssim_vs_size")

    # Single-view: best SSIM vs dataset size (RGB vs rich).
    family_single = "exp2_data_efficiency_singleview"
    base_single = os.path.join(outputs_root, family_single, "experiments")

    best_rgb_s, best_rich_s = _collect_best_ssim_for_sizes(
        base_dir=base_single,
        rgb_prefix="rgb_pairs",
        rich_prefix="rich_pairs",
        sizes=sizes_folder,
    )

    fig_s_size, ax_s_size = plt.subplots(figsize=(6, 4))
    ax_s_size.plot(
        sizes_plot,
        best_rgb_s,
        marker="o",
        linewidth=1.5,
        label="RGB",
    )
    ax_s_size.plot(
        sizes_plot,
        best_rich_s,
        marker="s",
        linewidth=1.5,
        label="Rich",
    )

    ax_s_size.set_xlabel("Training pairs")
    ax_s_size.set_ylabel("Best validation SSIM")
    ax_s_size.set_title("Experiment 2 (single-view): data efficiency")
    ax_s_size.set_xticks(sizes_plot)
    finite_vals_s = [v for v in (best_rgb_s + best_rich_s) if v == v]
    if finite_vals_s:
        margin = 0.005
        y_min = max(0.0, min(finite_vals_s) - margin)
        y_max = min(1.0, max(finite_vals_s) + margin)
        y_min = max(0.8, y_min)
        ax_s_size.set_ylim(y_min, y_max)
    ax_s_size.grid(True, linestyle="--", alpha=0.4)
    ax_s_size.legend(loc="best")

    _save_figure(fig_s_size, "exp2_single_best_ssim_vs_size")


def plot_all(outputs_root: str = "outputs") -> None:
    """Generate all publication-style plots for the current experiments."""
    plot_exp0_overfit(outputs_root=outputs_root)
    plot_exp1_buffers(outputs_root=outputs_root)
    plot_exp2_data_efficiency(outputs_root=outputs_root)


if __name__ == "__main__":
    # Run from the VolumetricCloudsTraining root, for example:
    #   uv run python scripts/plot_experiments.py
    # or:
    #   python scripts/plot_experiments.py
    plot_all(outputs_root="outputs")

