"""
Script principal de entrenamiento para identificación de parámetros de tape.

Usage:
    python scripts/train.py --config configs/default.yaml --name exp01_baseline
    python scripts/train.py --config configs/triple_param.yaml --name exp02_triple --learning_rate 1e-4
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from tape_id.models.encoder import SpectralEncoder
from tape_id.models.controller import ParameterController
from tape_id.data.dataset import TapeSaturationDataset
from tape_id.training.trainer import TapeIdentificationTrainer
from tape_id.utils import model_summary, seed_worker


def load_config(config_path: str, cli_overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train tape parameter identification model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (outputs saved to outputs/<name>/)")

    # All config keys can be overridden from CLI
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--ext", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--audio_length", type=int, default=None)
    parser.add_argument("--train_examples_per_epoch", type=int, default=None)
    parser.add_argument("--val_examples_per_epoch", type=int, default=None)
    parser.add_argument("--buffer_size_gb", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--degradation_model", type=str, default=None)
    parser.add_argument("--regression", type=bool, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--min_param", type=float, default=None)
    parser.add_argument("--max_param", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)

    # Triple/multi-param overrides
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--min_rate", type=float, default=None)
    parser.add_argument("--max_rate", type=float, default=None)
    parser.add_argument("--signal_loss_weight", type=float, default=None)
    parser.add_argument("--param_loss_weight", type=float, default=None)

    return parser.parse_args()


def setup_experiment_dirs(name: str) -> tuple:
    """Create experiment directory structure. Returns (output_dir, log_dir)."""
    exp_dir = Path("outputs") / name
    output_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    eval_dir = exp_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir), str(log_dir)


def create_controller(config: dict) -> ParameterController:
    """Create the appropriate ParameterController based on config."""
    regression = config.get("regression", False)
    degradation_model = config.get("degradation_model", "ja")
    triple_param = (degradation_model == "ja_wf")
    wf_target_param = config.get("wf_target_param", "depth")

    if triple_param:
        return ParameterController(
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            regression=True,
            triple_param=True,
        )
    elif wf_target_param == "both":
        return ParameterController(
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_classes_depth=config["num_classes_depth"],
            num_classes_rate=config["num_classes_rate"],
            regression=regression,
        )
    else:
        kwargs = dict(
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            regression=regression,
        )
        if not regression:
            kwargs["num_classes"] = config["num_classes"]
        return ParameterController(**kwargs)


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config, {
        k: v for k, v in vars(args).items()
        if k not in ("config", "name") and v is not None
    })

    # Experiment name
    name = args.name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir, log_dir = setup_experiment_dirs(name)

    # Save config copy for reproducibility
    exp_dir = Path("outputs") / name
    config_copy_path = exp_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Experiment: {name}")
    print(f"Config saved to: {config_copy_path}")
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Determine signal loss settings before dataset creation (return_clean affects batch format)
    signal_loss_weight = config.get("signal_loss_weight", 0.0)
    param_loss_weight = config.get("param_loss_weight", 1.0)
    use_signal_loss = config.get("regression", False)

    # Create datasets
    print("\nLoading datasets...")
    dataset_kwargs = dict(
        audio_dir=config["audio_dir"],
        input_dirs=config.get("input_dirs"),
        ext=config["ext"],
        length=config["audio_length"],
        min_param=config["min_param"],
        max_param=config["max_param"],
        num_classes=config.get("num_classes", 3),
        degradation_model=config["degradation_model"],
        log_scale=config.get("log_scale", False),
        buffer_size_gb=config["buffer_size_gb"],
        buffer_reload_rate=config.get("buffer_reload_rate", 2000),
        sample_rate=config["sample_rate"],
        regression=config.get("regression", False),
        wow_rate=config.get("wow_rate", 0.4),
        flutter_rate=config.get("flutter_rate", 0.5),
        enable_ou=config.get("enable_ou", True),
        wf_interpolation=config.get("wf_interpolation", "linear"),
        wf_target_param=config.get("wf_target_param", "depth"),
        wf_fixed_depth=config.get("wf_fixed_depth", 0.5),
        num_classes_depth=config.get("num_classes_depth"),
        num_classes_rate=config.get("num_classes_rate"),
        min_depth=config.get("min_depth"),
        max_depth=config.get("max_depth"),
        min_rate=config.get("min_rate"),
        max_rate=config.get("max_rate"),
        return_clean=use_signal_loss,
        # Tape noise (MagTapeDB) — ignored for other models
        noise_dir=config.get("noise_dir"),
        noise_input_dirs=config.get("noise_input_dirs"),
        noise_ext=config.get("noise_ext", "wav"),
        noise_train_frac=config.get("noise_train_frac", 0.8),
        noise_buffer_size_gb=config.get("noise_buffer_size_gb", 0.5),
        noise_buffer_reload_rate=config.get("noise_buffer_reload_rate", 1000),
        noise_preload=config.get("noise_preload", False),
        min_data=config.get("min_data"),
        max_data=config.get("max_data"),
    )

    train_dataset = TapeSaturationDataset(
        subset="train",
        num_examples_per_epoch=config["train_examples_per_epoch"],
        **dataset_kwargs,
    )

    val_dataset = TapeSaturationDataset(
        subset="val",
        num_examples_per_epoch=config["val_examples_per_epoch"],
        **dataset_kwargs,
    )

    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    num_workers = config["num_workers"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=min(num_workers, 1),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=min(num_workers, 1) > 0,
    )

    # Create models
    print("\nCreating models...")
    encoder = SpectralEncoder(
        num_params=1,
        sample_rate=config["sample_rate"],
        encoder_model=config.get("encoder_model", "mobilenet_v2"),
        embed_dim=config["embed_dim"],
        width_mult=2,
        multi_resolution=config.get("multi_resolution", False),
    )
    controller = create_controller(config)
    model_summary(encoder, controller)

    # Signal loss: instantiate forward model and loss function
    forward_model = None
    signal_loss_fn = None

    if use_signal_loss:
        from tape_id.models.tape_processor import DifferentiableForwardModel
        from tape_id.training.losses import MultiResolutionSTFTLoss

        forward_model = DifferentiableForwardModel(
            degradation_model=config["degradation_model"],
            min_param=config["min_param"],
            max_param=config["max_param"],
            min_depth=config.get("min_depth", 0.1),
            max_depth=config.get("max_depth", 0.8),
            min_rate=config.get("min_rate", 0.1),
            max_rate=config.get("max_rate", 0.8),
            sample_rate=config["sample_rate"],
        )
        signal_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=config.get("signal_loss_fft_sizes", [1024, 2048, 8192]),
            hop_sizes=config.get("signal_loss_hop_sizes", [256, 512, 2048]),
            win_lengths=config.get("signal_loss_win_lengths", [1024, 2048, 8192]),
        )
        print(f"\nSignal loss enabled: weight={signal_loss_weight}, param_weight={param_loss_weight}")

    # Optimizer
    params = list(encoder.parameters()) + list(controller.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-5),
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("scheduler_factor", 0.5),
        patience=config.get("scheduler_patience", 5),
    )

    # Create trainer
    trainer = TapeIdentificationTrainer(
        encoder=encoder,
        controller=controller,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config["device"],
        output_dir=output_dir,
        log_dir=log_dir,
        patience=config.get("patience", 15),
        min_delta=config.get("min_delta", 0.001),
        grad_clip_norm=config.get("grad_clip_norm", 1.0),
        forward_model=forward_model,
        signal_loss_weight=signal_loss_weight,
        param_loss_weight=param_loss_weight,
        signal_loss_fn=signal_loss_fn,
        min_param=config["min_param"],
        max_param=config["max_param"],
        min_depth=config.get("min_depth", 0.0),
        max_depth=config.get("max_depth", 1.0),
        min_rate=config.get("min_rate", 0.0),
        max_rate=config.get("max_rate", 1.0),
    )

    # Auto-resume from last checkpoint
    checkpoint_dir = Path(output_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    resume_from = None
    if checkpoints:
        regular = [c for c in checkpoints if "emergency" not in c.name]
        if regular:
            resume_from = str(regular[-1])
            print(f"\nFound checkpoint: {resume_from}")
            response = input("Resume from this checkpoint? (y/n): ")
            if response.lower() != "y":
                resume_from = None

    # Train
    trainer.train(num_epochs=config["num_epochs"], resume_from=resume_from)


if __name__ == "__main__":
    main()
