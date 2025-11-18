# ================================================================
# Entry point for volumetric cloud training.
# Loads configuration from config/config.yaml and launches training.
# ================================================================

import argparse
import sys
from dataclasses import MISSING, fields
from pathlib import Path

import yaml


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
        "save_every_n_steps": train_section.get("save_every_n_steps", field_defaults["save_every_n_steps"]),
        "log_every_n_steps": train_section.get("log_every_n_steps", field_defaults["log_every_n_steps"]),
        # Model configuration (controls parameter count / capacity)
        "model_base_channels": train_section.get("model_base_channels", field_defaults["model_base_channels"]),
        "model_bilinear": train_section.get("model_bilinear", field_defaults["model_bilinear"]),
        "model_learn_residual": train_section.get("model_learn_residual", field_defaults["model_learn_residual"]),
        # Auxiliary feature usage ------------------------------------------------
        "use_view_transmittance": train_section.get("use_view_transmittance", field_defaults["use_view_transmittance"]),
        "use_light_transmittance": train_section.get("use_light_transmittance", field_defaults["use_light_transmittance"]),
        "use_linear_depth": train_section.get("use_linear_depth", field_defaults["use_linear_depth"]),
        "depth_normalization_max": train_section.get("depth_normalization_max", field_defaults["depth_normalization_max"]),
    }

    return TrainConfig(**kwargs)

def _build_infer_config(config: dict, project_root: Path) -> InferConfig:
    infer_section = config.get("infer", {})
    if not isinstance(infer_section, dict):
        raise TypeError("The 'infer' section of the configuration must be a dictionary.")

    # Resolve paths relative to project root
    checkpoint_path = _resolve_project_path(infer_section.get("checkpoint", "Outputs/checkpoints/unet_epoch_1.pt"), project_root)
    input_path = _resolve_project_path(infer_section.get("input", "TrainingCaptures"), project_root)
    output_dir = _resolve_project_path(infer_section.get("output_dir", "Outputs/infer"), project_root)

    return InferConfig(
        checkpoint=checkpoint_path,
        input_path=input_path,
        output_dir=output_dir,
        input_glob=infer_section.get("input_glob", "*_low.pfm"),
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
        # Auxiliary feature usage (must match training)
        use_view_transmittance=bool(infer_section.get("use_view_transmittance", True)),
        use_light_transmittance=bool(infer_section.get("use_light_transmittance", True)),
        use_linear_depth=bool(infer_section.get("use_linear_depth", True)),
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
        train_config = _build_train_config(config_map, PROJECT_ROOT)
        train_unet(train_config)
    else:
        infer_config = _build_infer_config(config_map, PROJECT_ROOT)
        infer_unet(infer_config)


if __name__ == "__main__":
    main()

