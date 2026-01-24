"""
Utility functions - Copiado desde DeepAFx-ST.
"""

import torch
import torch.nn as nn


def count_parameters(model, trainable_only=True):
    """
    Cuenta parámetros de un modelo.

    Args:
        model: Modelo de PyTorch
        trainable_only: Si es True, solo cuenta parámetros entrenables

    Returns:
        Número de parámetros
    """
    if trainable_only:
        if len(list(model.parameters())) > 0:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            params = 0
    else:
        if len(list(model.parameters())) > 0:
            params = sum(p.numel() for p in model.parameters())
        else:
            params = 0

    return params


def model_summary(encoder, controller, processor):
    """
    Imprime resumen de parámetros del modelo.

    Args:
        encoder: Encoder model
        controller: Controller model
        processor: Processor model
    """
    encoder_params = count_parameters(encoder)
    controller_params = count_parameters(controller)
    processor_params = count_parameters(processor)
    total_params = encoder_params + controller_params + processor_params

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Encoder:     {encoder_params/1e6:>8.2f} M parameters")
    print(f"Controller:  {controller_params/1e6:>8.2f} M parameters")
    print(f"Processor:   {processor_params/1e6:>8.2f} M parameters")
    print("-" * 60)
    print(f"Total:       {total_params/1e6:>8.2f} M parameters")

    # Non-trainable params
    encoder_nt = count_parameters(encoder, trainable_only=False) - encoder_params
    controller_nt = count_parameters(controller, trainable_only=False) - controller_params
    processor_nt = count_parameters(processor, trainable_only=False) - processor_params
    total_nt = encoder_nt + controller_nt + processor_nt

    if total_nt > 0:
        print(f"Non-trainable: {total_nt/1e6:>6.2f} M parameters")

    print("=" * 60 + "\n")


def split_dataset(file_list, subset, train_frac):
    """Given a list of files, split into train/val/test sets.

    Args:
        file_list (list): List of audio files.
        subset (str): One of "train", "val", or "test".
        train_frac (float): Fraction of the dataset to use for training.

    Returns:
        file_list (list): List of audio files corresponding to subset.
    """
    assert train_frac > 0.1 and train_frac < 1.0

    total_num_examples = len(file_list)

    train_num_examples = int(total_num_examples * train_frac)
    val_num_examples = int(total_num_examples * (1 - train_frac) / 2)
    test_num_examples = total_num_examples - (train_num_examples + val_num_examples)

    if train_num_examples < 0:
        raise ValueError(
            f"No examples in training set. Try increasing train_frac: {train_frac}."
        )
    elif val_num_examples < 0:
        raise ValueError(
            f"No examples in validation set. Try decreasing train_frac: {train_frac}."
        )
    elif test_num_examples < 0:
        raise ValueError(
            f"No examples in test set. Try decreasing train_frac: {train_frac}."
        )

    if subset == "train":
        file_list = file_list[:train_num_examples]
    elif subset == "val":
        file_list = file_list[
            train_num_examples : train_num_examples + val_num_examples
        ]
    elif subset == "test":
        file_list = file_list[train_num_examples + val_num_examples :]

    return file_list


def conform_length(x: torch.Tensor, length: int):
    """Crop or pad input on last dim to match `length`."""
    if x.shape[-1] < length:
        padsize = length - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, padsize))
    elif x.shape[-1] > length:
        x = x[..., :length]

    return x


def linear_fade(
    x: torch.Tensor,
    fade_ms: float = 50.0,
    sample_rate: float = 22050,
):
    """Apply fade in and fade out to last dim."""
    fade_samples = int(fade_ms * 1e-3 * 22050)

    fade_in = torch.linspace(0.0, 1.0, steps=fade_samples)
    fade_out = torch.linspace(1.0, 0.0, steps=fade_samples)

    # fade in
    x[..., :fade_samples] *= fade_in

    # fade out
    x[..., -fade_samples:] *= fade_out

    return x


def seed_worker(worker_id):
    """Seed workers for reproducibility."""
    import random
    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
