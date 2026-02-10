"""
Script principal de entrenamiento para identificación de parámetros de tape.
"""

import sys
from pathlib import Path

# Agregar tape_id al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from tape_id.models.encoder import SpectralEncoder
from tape_id.models.controller import ParameterController
from tape_id.data.dataset import TapeSaturationDataset
from tape_id.training.trainer import TapeIdentificationTrainer
from tape_id.utils import model_summary, seed_worker


def main():
    # Configuración
    config = {
        # Datos
        "audio_dir": "/mnt/data/working_datasets/jamendo",
        "input_dirs": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
        "ext": "mp3",
        "batch_size": 32,
        "num_workers": 0,
        "sample_rate": 22050,
        "audio_length": 65536,
        "train_examples_per_epoch": 5000,
        "val_examples_per_epoch": 500,
        "buffer_size_gb": 3.5,
        "buffer_reload_rate": 2000,

        # Modelo
        "embed_dim": 1024,
        "hidden_dim": 256,
        "min_gain": 0.1,
        "max_gain": 0.8,
        "num_classes": 3,
        "degradation_model": "wow_flutter",
        "log_scale": False,

        # Wow/flutter
        "wf_target_param": "rate",    # clasificar por rate (depth fijo)
        "wf_fixed_depth": 0.5,        # depth fijo a 5 ms
        "flutter_rate": 0.0,
        "enable_ou": False,
        "wf_interpolation": "linear",

        # Training
        "num_epochs": 400,
        "learning_rate": 5e-5,
        "device": "cuda",

        # Output
        "output_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Crear datasets
    print("\nLoading datasets...")
    dataset_kwargs = dict(
        audio_dir=config["audio_dir"],
        input_dirs=config["input_dirs"],
        ext=config["ext"],
        length=config["audio_length"],
        min_param=config["min_gain"],
        max_param=config["max_gain"],
        num_classes=config["num_classes"],
        degradation_model=config["degradation_model"],
        log_scale=config["log_scale"],
        buffer_size_gb=config["buffer_size_gb"],
        buffer_reload_rate=config["buffer_reload_rate"],
        sample_rate=config["sample_rate"],
        wow_rate=config.get("wow_rate", 0.4),
        flutter_rate=config.get("flutter_rate", 0.5),
        enable_ou=config.get("enable_ou", True),
        wf_interpolation=config.get("wf_interpolation", "linear"),
        wf_target_param=config.get("wf_target_param", "depth"),
        wf_fixed_depth=config.get("wf_fixed_depth", 0.5),
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

    # Generator para reproducibilidad
    g = torch.Generator()
    g.manual_seed(0)

    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
    )

    # Crear modelos
    print("\nCreating models...")
    encoder = SpectralEncoder(
        num_params=1,
        sample_rate=config["sample_rate"],
        embed_dim=config["embed_dim"],
        width_mult=2,
    )

    controller = ParameterController(
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
    )

    # Mostrar resumen del modelo
    model_summary(encoder, controller)

    # Optimizer (solo encoder + controller)
    params = (
        list(encoder.parameters())
        + list(controller.parameters())
    )
    optimizer = torch.optim.Adam(
        params,
        lr=config["learning_rate"],
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Crear trainer
    trainer = TapeIdentificationTrainer(
        encoder=encoder,
        controller=controller,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config["device"],
        output_dir=config["output_dir"],
        log_dir=config["log_dir"],
    )

    # Buscar último checkpoint para reanudar
    from pathlib import Path
    checkpoints = sorted(Path(config["output_dir"]).glob("checkpoint_epoch*.pt"))
    resume_from = None
    if checkpoints:
        # Filtrar checkpoints de emergencia
        regular_checkpoints = [c for c in checkpoints if "emergency" not in c.name]
        if regular_checkpoints:
            resume_from = str(regular_checkpoints[-1])
            print(f"\nFound checkpoint: {resume_from}")
            response = input("Resume from this checkpoint? (y/n): ")
            if response.lower() != 'y':
                resume_from = None

    # Entrenar
    trainer.train(num_epochs=config["num_epochs"], resume_from=resume_from)


if __name__ == "__main__":
    main()
