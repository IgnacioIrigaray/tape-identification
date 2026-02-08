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
from tape_id.models.tape_processor import HardClippingProcessor
from tape_id.data.dataset import TapeSaturationDataset
from tape_id.training.losses import MultiResolutionSTFTLoss
from tape_id.training.trainer import TapeIdentificationTrainer
from tape_id.utils import model_summary, seed_worker


def main():
    # Configuración
    config = {
        # Datos
        "audio_dir": "/mnt/data/working_datasets/jamendo",
        "input_dirs": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
        "ext": "wav",
        "batch_size": 8,  # Reducido de 16 para menor uso de memoria
        "num_workers": 0,
        "audio_length": 65536,
        "train_examples_per_epoch": 5000,
        "val_examples_per_epoch": 500,
        "buffer_size_gb": 0.2,  # Reducido de 0.5 para menor uso de memoria
        "buffer_reload_rate": 2000,
        "train_frac": 0.8,

        # Modelo
        "embed_dim": 1024,
        "hidden_dim": 256,
        "params_reg_weight": 0, # peso de regularizacion basada en params
        "min_gain": 1.0,  # Hard clipping: gain=1 es bypass
        "max_gain": 4.0,  # gain=4 es clipping severo
        "num_classes": 3,
        "saturation_model": "hard_clipping",
        "log_scale": False,

        # Training
        "num_epochs": 100,
        "learning_rate": 3e-4,
        "device": "cuda",  # GPU no compatible con PyTorch actual (sm_86)

        # Output
        "output_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Crear datasets
    print("\nLoading datasets...")
    train_dataset = TapeSaturationDataset(
        audio_dir=config["audio_dir"],
        input_dirs=config["input_dirs"],
        ext=config["ext"],
        subset="train",
        train_frac=config["train_frac"],
        length=config["audio_length"],
        min_param=config["min_gain"],
        max_param=config["max_gain"],
        num_classes=config["num_classes"],
        saturation_model=config["saturation_model"],
        log_scale=config["log_scale"],
        num_examples_per_epoch=config["train_examples_per_epoch"],
        buffer_size_gb=config["buffer_size_gb"],
        buffer_reload_rate=config["buffer_reload_rate"],
    )

    val_dataset = TapeSaturationDataset(
        audio_dir=config["audio_dir"],
        input_dirs=config["input_dirs"],
        ext=config["ext"],
        subset="val",
        train_frac=config["train_frac"],
        length=config["audio_length"],
        min_param=config["min_gain"],
        max_param=config["max_gain"],
        num_classes=config["num_classes"],
        saturation_model=config["saturation_model"],
        log_scale=config["log_scale"],
        num_examples_per_epoch=config["val_examples_per_epoch"],
        buffer_size_gb=config["buffer_size_gb"],
        buffer_reload_rate=config["buffer_reload_rate"],
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
        sample_rate=24000,
        embed_dim=config["embed_dim"],
        width_mult=2,  # Igual que DeepAFx-ST
    )

    controller = ParameterController(
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
    )

    processor = HardClippingProcessor(
        min_gain=config["min_gain"],
        max_gain=config["max_gain"],
        num_classes=config["num_classes"],
    )

    # Mostrar resumen del modelo
    model_summary(encoder, controller, processor)

    # Loss y optimizer
    loss_fn = MultiResolutionSTFTLoss()

    params = (
        list(encoder.parameters())
        + list(controller.parameters())
        + list(processor.parameters())
    )
    optimizer = torch.optim.Adam(
        params,
        lr=config["learning_rate"],
        weight_decay=1e-5  # L2 regularization para reducir overfitting
    )

    # Learning rate scheduler - reduce LR cuando validation se estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Crear trainer
    trainer = TapeIdentificationTrainer(
        encoder=encoder,
        controller=controller,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        params_reg_weight=config["params_reg_weight"],
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
