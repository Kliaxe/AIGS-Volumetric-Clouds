# ================================================================
# Entry point for volumetric cloud training.
# Loads configuration from config/config.yaml and launches training.
# ================================================================

import argparse
import shutil
import sys
from dataclasses import MISSING, fields
from pathlib import Path

import numpy as np
import yaml
import matplotlib


# Use a non-interactive backend for environments without a display (e.g., CI/headless)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------
# Resolve project and source directories
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from VolumetricCloudsTraining.train_unet import TrainConfig, train_unet  # noqa: E402
from VolumetricCloudsTraining.infer_unet import InferConfig, infer_unet  # noqa: E402


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run volumetric cloud U-Net training using a YAML configuration file."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=("train", "infer"),
        help="Whether to train or run inference.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def _load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping/dictionary. Got {type(data)!r}.")

    return data


def _resolve_project_path(raw_path: str, base_dir: Path) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _build_train_config(config: dict, project_root: Path) -> TrainConfig:
    train_section = config.get("train", {})
    if not isinstance(train_section, dict):
        raise TypeError("The 'train' section of the configuration must be a dictionary.")

    field_defaults = {
        field.name: (field.default if field.default is not MISSING else None)
        for field in fields(TrainConfig)
    }

    data_dir = train_section.get("data_dir", "TrainingCaptures" if field_defaults["data_dir"] is None else field_defaults["data_dir"])
    output_dir = train_section.get("output_dir", field_defaults["output_dir"])

    resolved_data_dir = _resolve_project_path(data_dir, project_root)
    resolved_output_dir = _resolve_project_path(output_dir, project_root)

    kwargs = {
        "data_dir": resolved_data_dir,
        "output_dir": resolved_output_dir,
        "epochs": train_section.get("epochs", field_defaults["epochs"]),
        "batch_size": train_section.get("batch_size", field_defaults["batch_size"]),
        "learning_rate": train_section.get("learning_rate", field_defaults["learning_rate"]),
        "crop_size": train_section.get("crop_size", field_defaults["crop_size"]),
        "limit_pairs": train_section.get("limit_pairs", field_defaults["limit_pairs"]),
        "num_workers": train_section.get("num_workers", field_defaults["num_workers"]),
        "device": train_section.get("device", field_defaults["device"]),
        "seed": train_section.get("seed", field_defaults["seed"]),
        # Dataset split configuration ----------------------------------------
        "train_fraction": train_section.get("train_fraction", field_defaults.get("train_fraction", 0.8)),
        "val_fraction": train_section.get("val_fraction", field_defaults.get("val_fraction", 0.1)),
        "test_fraction": train_section.get("test_fraction", field_defaults.get("test_fraction", 0.1)),
        "split_seed": train_section.get("split_seed", field_defaults.get("split_seed", 12345)),
        "export_every_n_epochs": train_section.get(
            "export_every_n_epochs", field_defaults.get("export_every_n_epochs", 0)
        ),
        "save_epoch_stride": train_section.get(
            "save_epoch_stride", field_defaults.get("save_epoch_stride", 1)
        ),
        "save_every_n_steps": train_section.get("save_every_n_steps", field_defaults["save_every_n_steps"]),
        "log_every_n_steps": train_section.get("log_every_n_steps", field_defaults["log_every_n_steps"]),
        "save_only_last_epoch": train_section.get("save_only_last_epoch", field_defaults.get("save_only_last_epoch", False)),
        # Model configuration (controls parameter count / capacity)
        "model_base_channels": train_section.get("model_base_channels", field_defaults["model_base_channels"]),
        "model_bilinear": train_section.get("model_bilinear", field_defaults["model_bilinear"]),
        "model_learn_residual": train_section.get("model_learn_residual", field_defaults["model_learn_residual"]),
        # Auxiliary feature usage ------------------------------------------------
        "use_view_transmittance": train_section.get(
            "use_view_transmittance", field_defaults["use_view_transmittance"]
        ),
        "use_light_transmittance": train_section.get(
            "use_light_transmittance", field_defaults["use_light_transmittance"]
        ),
        "use_linear_depth": train_section.get(
            "use_linear_depth", field_defaults["use_linear_depth"]
        ),
        "use_normals": train_section.get(
            "use_normals", field_defaults.get("use_normals", False)
        ),
        "depth_normalization_max": train_section.get(
            "depth_normalization_max", field_defaults["depth_normalization_max"]
        ),
        # Loss configuration ------------------------------------------------------
        "use_auxiliary_in_loss": train_section.get(
            "use_auxiliary_in_loss", field_defaults.get("use_auxiliary_in_loss", False)
        ),
    }

    return TrainConfig(**kwargs)


def _get_run_name(config: dict) -> str:
    """
    Extract a short label for this configuration from the root 'name' field.
    Falls back to 'default' when missing or empty.
    """
    raw = config.get("name")
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            return stripped
    return "default"


def _snapshot_config(config_path: Path, project_root: Path, run_name: str) -> None:
    """
    Save a copy of the configuration file under Outputs/<run_name>/ so that
    each training run keeps an exact record of the settings used.
    """
    if not config_path.exists():
        return

    target_dir = project_root / "Outputs" / run_name
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / config_path.name
    shutil.copy2(str(config_path), str(target_path))
    print(f"Saved config snapshot to: {target_path}")


def _aggregate_experiment_loss_curves(experiments_root: Path) -> None:
    """
    Aggregate per-experiment loss curves into a single comparison plot under:
        <experiments_root>/result.png
    Each experiment contributes its epoch-average train/val curves.
    """
    if not experiments_root.exists():
        return

    experiment_dirs = [p for p in experiments_root.iterdir() if p.is_dir()]
    if not experiment_dirs:
        return

    # Predefined colours for known experiment names; fall back to a default cycle otherwise.
    base_colour_map = {
        "baseline_rgb": "tab:blue",
        "view_only": "tab:orange",
        "light_only": "tab:red",
        "depth_only": "tab:purple",
        "all_features": "tab:green",
    }

    def _parse_epoch_losses(log_path: Path) -> tuple[list[int], list[float], list[float]]:
        """Parse epoch-level train/val losses from a loss_curve.txt file."""
        steps: list[int] = []
        train_vals: list[float] = []
        val_vals: list[float] = []

        in_epoch_section = False
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    # Detect start of the epoch section by its header.
                    if stripped.startswith("# epoch_step"):
                        in_epoch_section = True
                    continue
                if not in_epoch_section:
                    continue

                parts = stripped.split()
                if len(parts) != 3:
                    continue
                try:
                    step_val = int(float(parts[0]))
                    train_val = float(parts[1])
                    val_val = float(parts[2])
                except ValueError:
                    continue

                steps.append(step_val)
                train_vals.append(train_val)
                val_vals.append(val_val)

        return steps, train_vals, val_vals

    # Collect per-experiment epoch curves.
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for exp_dir in experiment_dirs:
        logs_dir = exp_dir / "logs"
        log_txt = logs_dir / "loss_curve.txt"
        if not log_txt.exists():
            continue

        epoch_steps, train_vals, val_vals = _parse_epoch_losses(log_txt)
        if not train_vals:
            continue

        epochs = np.arange(1, len(train_vals) + 1, dtype=np.int32)
        train_arr = np.asarray(train_vals, dtype=np.float64)
        val_arr = np.asarray(val_vals, dtype=np.float64) if val_vals else np.full_like(train_arr, np.nan)

        series.append((exp_dir.name, epochs, train_arr, val_arr))

    if not series:
        return

    fig = plt.figure(figsize=(9, 5), dpi=120)

    for exp_name, epochs, train_arr, val_arr in series:
        base_colour = base_colour_map.get(exp_name, None)
        if base_colour is None:
            # Let matplotlib choose a default colour when not specified.
            base_colour = None

        # Train: solid line, base colour.
        plt.plot(
            epochs,
            train_arr,
            label=f"{exp_name} train",
            color=base_colour,
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

        # Val: same hue but lighter, or default if base colour is None.
        if np.any(np.isfinite(val_arr)):
            if base_colour is not None:
                rgb = np.array(matplotlib.colors.to_rgb(base_colour))
                lighter_rgb = 1.0 - (1.0 - rgb) * 0.5  # simple linear blend towards white.
                val_colour = lighter_rgb
            else:
                val_colour = None

            plt.plot(
                epochs,
                val_arr,
                label=f"{exp_name} val",
                color=val_colour,
                linewidth=1.5,
                linestyle="--",
                marker="s",
                markersize=3,
            )

    plt.yscale("log")
    plt.ylim(bottom=5e-3, top=1e-1)
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss (log scale)")
    plt.title("Experiment comparison – epoch average loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_path = experiments_root / "result.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved aggregated experiment comparison plot: {out_path}")

def _build_infer_config(config: dict, project_root: Path) -> InferConfig:
    infer_section = config.get("infer", {})
    if not isinstance(infer_section, dict):
        raise TypeError("The 'infer' section of the configuration must be a dictionary.")

    run_name = _get_run_name(config)

    # Resolve paths relative to project root
    checkpoint_path = _resolve_project_path(
        infer_section.get("checkpoint", "Outputs/checkpoints/unet_epoch_1.pt"), project_root
    )
    input_path = _resolve_project_path(infer_section.get("input", "TrainingCaptures"), project_root)

    # When the user does not specify an explicit infer.output_dir, group results
    # under Outputs/<run_name>/infer so that multiple configs can coexist cleanly.
    default_infer_output = f"Outputs/{run_name}/infer"
    output_dir = _resolve_project_path(infer_section.get("output_dir", default_infer_output), project_root)

    return InferConfig(
        checkpoint=checkpoint_path,
        input_path=input_path,
        output_dir=output_dir,
        input_glob=infer_section.get("input_glob", "*_low.pfm"),
        split_mode=infer_section.get("split_mode", "custom"),
        train_fraction=float(infer_section.get("train_fraction", 0.8)),
        val_fraction=float(infer_section.get("val_fraction", 0.1)),
        test_fraction=float(infer_section.get("test_fraction", 0.1)),
        split_seed=int(infer_section.get("split_seed", 12345)),
        split_sample_index=(
            int(infer_section["split_sample_index"])
            if "split_sample_index" in infer_section and infer_section["split_sample_index"] is not None
            else None
        ),
        # Model configuration (must match training checkpoint)
        model_base_channels=int(infer_section.get("model_base_channels", 64)),
        model_bilinear=bool(infer_section.get("model_bilinear", True)),
        model_learn_residual=bool(infer_section.get("model_learn_residual", True)),
        device=infer_section.get("device", "auto"),
        upsample_mode=infer_section.get("upsample_mode", "bicubic"),
        align_corners=bool(infer_section.get("align_corners", False)),
        clamp_min=float(infer_section.get("clamp_min", 0.0)),
        clamp_max=float(infer_section.get("clamp_max", 1.0)),
        scale_factor=int(infer_section.get("scale_factor", 4)),
        recursive=bool(infer_section.get("recursive", False)),
        output_suffix=str(infer_section.get("output_suffix", "_pred")),
        # Experiment-aware checkpoint selection --------------------------------
        experiment_name=str(infer_section.get("experiment_name", "")),
        checkpoint_dir=(
            _resolve_project_path(infer_section["checkpoint_dir"], project_root)
            if "checkpoint_dir" in infer_section and infer_section["checkpoint_dir"]
            else ""
        ),
        checkpoint_glob=str(infer_section.get("checkpoint_glob", "unet_epoch_*.pt")),
        run_all_checkpoints=bool(infer_section.get("run_all_checkpoints", False)),
        # Auxiliary feature usage (must match training)
        use_view_transmittance=bool(infer_section.get("use_view_transmittance", True)),
        use_light_transmittance=bool(infer_section.get("use_light_transmittance", True)),
        use_linear_depth=bool(infer_section.get("use_linear_depth", True)),
        use_normals=bool(infer_section.get("use_normals", False)),
        depth_normalization_max=float(infer_section.get("depth_normalization_max", 40000.0)),
    )


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config_map = _load_yaml_config(config_path)

    if args.mode == "train":
        run_name = _get_run_name(config_map)
        train_section = config_map.get("train", {})
        experiments_cfg = train_section.get("experiments")

        # No experiments specified: behave as a single training run.
        if not experiments_cfg:
            train_config = _build_train_config(config_map, PROJECT_ROOT)
            # For the single-run case, keep train_config.output_dir behaviour as-is
            # to avoid surprising existing usage.
            train_unet(train_config)
        else:
            if not isinstance(experiments_cfg, list):
                raise TypeError("The 'train.experiments' section must be a list of experiment dictionaries.")

            base_train_config = _build_train_config(config_map, PROJECT_ROOT)
            # Group all experiments for this configuration under:
            #   Outputs/<run_name>/experiments/<experiment_name>/
            base_output_dir = (PROJECT_ROOT / "Outputs" / run_name / "experiments")

            for exp in experiments_cfg:
                if not isinstance(exp, dict):
                    raise TypeError("Each entry in 'train.experiments' must be a dictionary.")

                exp_name = str(exp.get("name", "experiment")).strip()
                if not exp_name:
                    exp_name = "experiment"

                # Derive per‑experiment output directory:
                #   <base_output_dir>/<experiment_name>/
                exp_output_dir = (base_output_dir / exp_name).resolve()

                # Start from the base config and apply experiment overrides.
                exp_kwargs = vars(base_train_config).copy()
                exp_kwargs["output_dir"] = str(exp_output_dir)

                # Allow experiments to override common knobs (feature toggles, capacity, schedule).
                override_keys = {
                    "epochs",
                    "batch_size",
                    "learning_rate",
                    "crop_size",
                    "limit_pairs",
                    "model_base_channels",
                    "model_bilinear",
                    "model_learn_residual",
                    "use_view_transmittance",
                    "use_light_transmittance",
                    "use_linear_depth",
                    "use_normals",
                    "depth_normalization_max",
                    "use_auxiliary_in_loss",
                    "save_epoch_stride",
                    "save_every_n_steps",
                    "log_every_n_steps",
                    "save_only_last_epoch",
                }
                for key in override_keys:
                    if key in exp:
                        exp_kwargs[key] = exp[key]

                exp_config = TrainConfig(**exp_kwargs)
                print(f"===============================================================")
                print(f"Running experiment '{exp_name}'")
                print(f"  output_dir = {exp_config.output_dir}")
                print(
                    f"  features   = view={exp_config.use_view_transmittance}, "
                    f"light={exp_config.use_light_transmittance}, "
                    f"depth={exp_config.use_linear_depth}, "
                    f"normals={exp_config.use_normals}"
                )
                print(f"===============================================================")

                train_unet(exp_config)

            # After all experiments complete, aggregate their epoch curves into
            # a single comparison PNG under:
            #   Outputs/<run_name>/experiments/result.png
            _aggregate_experiment_loss_curves(base_output_dir)

        # After training (single-run or experiments), store a snapshot of the
        # configuration file under Outputs/<run_name>/ for reproducibility.
        _snapshot_config(config_path, PROJECT_ROOT, run_name)
    else:
        infer_section = config_map.get("infer", {})
        # Optional: run inference over all training experiments in a single call.
        # When infer.run_all_experiments is True, we loop over train.experiments
        # and, for each experiment, build an InferConfig that points at that
        # experiment's checkpoints under:
        #   Outputs/<run_name>/experiments/<experiment_name>/checkpoints
        if isinstance(infer_section, dict) and infer_section.get("run_all_experiments", False):
            run_name = _get_run_name(config_map)
            train_section = config_map.get("train", {})
            experiments_cfg = train_section.get("experiments")

            if not experiments_cfg:
                raise ValueError(
                    "infer.run_all_experiments is True, but no 'train.experiments' list was found in the config."
                )
            if not isinstance(experiments_cfg, list):
                raise TypeError("The 'train.experiments' section must be a list of experiment dictionaries.")

            base_infer_config = _build_infer_config(config_map, PROJECT_ROOT)

            experiments_root = PROJECT_ROOT / "Outputs" / run_name / "experiments"
            # Group all inference outputs for this configuration under a shared
            # experiments/infer folder, and encode experiment / checkpoint in
            # the filename rather than nested directories.
            infer_root = experiments_root / "infer"
            for exp in experiments_cfg:
                if not isinstance(exp, dict):
                    raise TypeError("Each entry in 'train.experiments' must be a dictionary.")

                exp_name = str(exp.get("name", "experiment")).strip()
                if not exp_name:
                    exp_name = "experiment"

                exp_ckpt_dir = (experiments_root / exp_name / "checkpoints").resolve()

                exp_kwargs = vars(base_infer_config).copy()
                exp_kwargs["experiment_name"] = exp_name
                exp_kwargs["checkpoint_dir"] = str(exp_ckpt_dir)
                # Group inference outputs under:
                #   Outputs/<run_name>/experiments/infer/
                # and encode experiment / checkpoint in the filename.
                exp_kwargs["output_dir"] = str(infer_root.resolve())
                # When evaluating all experiments we also iterate over all of
                # their checkpoints by default.
                exp_kwargs["run_all_checkpoints"] = True

                # Mirror the most important architectural / feature toggles
                # from the training experiment so that inference stays
                # consistent with the checkpoints.
                infer_override_keys = {
                    "model_base_channels",
                    "model_bilinear",
                    "model_learn_residual",
                    "use_view_transmittance",
                    "use_light_transmittance",
                    "use_linear_depth",
                    "use_normals",
                    "depth_normalization_max",
                }
                for key in infer_override_keys:
                    if key in exp:
                        exp_kwargs[key] = exp[key]

                exp_infer_config = InferConfig(**exp_kwargs)

                print("===============================================================")
                print(f"Running inference for experiment '{exp_name}'")
                print(f"  checkpoint_dir = {exp_infer_config.checkpoint_dir}")
                print(
                    f"  features       = view={exp_infer_config.use_view_transmittance}, "
                    f"light={exp_infer_config.use_light_transmittance}, "
                    f"depth={exp_infer_config.use_linear_depth}, "
                    f"normals={exp_infer_config.use_normals}"
                )
                print("===============================================================")

                infer_unet(exp_infer_config)
        else:
            infer_config = _build_infer_config(config_map, PROJECT_ROOT)
            infer_unet(infer_config)


if __name__ == "__main__":
    main()

