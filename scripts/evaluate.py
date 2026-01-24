"""
Script de evaluación para modelo entrenado.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchaudio
import argparse
from typing import Tuple

from tape_id.models.encoder import SpectralEncoder
from tape_id.models.controller import ParameterController
from tape_id.models.tape_processor import TapeSaturationProcessor, apply_tape_saturation


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Carga modelo desde checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = SpectralEncoder(num_params=1, embed_dim=128, width_mult=2).to(device)
    controller = ParameterController(num_params=1, embed_dim=128).to(device)
    processor = TapeSaturationProcessor(min_depth=1.0, max_depth=10.0).to(device)

    encoder.load_state_dict(checkpoint["encoder_state"])
    controller.load_state_dict(checkpoint["controller_state"])
    processor.load_state_dict(checkpoint["processor_state"])

    encoder.eval()
    controller.eval()
    processor.eval()

    return encoder, controller, processor


@torch.no_grad()
def estimate_parameter(
    audio: torch.Tensor,
    audio_saturated: torch.Tensor,
    encoder,
    controller,
    device: str,
) -> float:
    """
    Estima el parámetro de saturación.

    Args:
        audio: Audio limpio [samples]
        audio_saturated: Audio saturado [samples]
        encoder: Encoder model
        controller: Controller model
        device: Device

    Returns:
        Parámetro estimado [1, 10]
    """
    # Preparar tensores
    x = audio.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
    y = audio_saturated.unsqueeze(0).unsqueeze(0).to(device)

    # Obtener embeddings
    e_x = encoder(x)
    e_y = encoder(y)

    # Predecir parámetro
    p_norm = controller(e_x, e_y)  # [0, 1]

    # Desnormalizar
    p_real = 1.0 + p_norm[0, 0].item() * 9.0

    return p_real


def main():
    parser = argparse.ArgumentParser(description="Evaluate tape identification model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--audio", type=str, help="Path to audio file (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Cargar modelo
    print(f"Loading model from {args.checkpoint}...")
    encoder, controller, processor = load_model(args.checkpoint, device)
    print("Model loaded successfully!")

    # Prueba con audio sintético si no se proporciona
    if args.audio is None:
        print("\nGenerating synthetic audio for testing...")
        sr = 24000
        duration = 3.0
        t = torch.linspace(0, duration, int(sr * duration))
        audio = torch.sin(2 * 3.14159 * 220 * t)
        audio += 0.5 * torch.sin(2 * 3.14159 * 440 * t)
        audio = audio / audio.abs().max()
    else:
        print(f"\nLoading audio from {args.audio}...")
        audio, sr = torchaudio.load(args.audio)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)
        audio = audio / audio.abs().max()

    # Tomar 3 segundos
    max_samples = 24000 * 3
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    print(f"Audio shape: {audio.shape}")

    # Probar con diferentes depths
    print("\n" + "=" * 60)
    print("PARAMETER ESTIMATION TEST")
    print("=" * 60)
    print(f"{'Real Depth':>12} | {'Estimated':>12} | {'Error':>10}")
    print("-" * 60)

    test_depths = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    errors = []

    for real_depth in test_depths:
        # Aplicar saturación
        audio_sat = apply_tape_saturation(audio, real_depth)

        # Estimar parámetro
        estimated = estimate_parameter(audio, audio_sat, encoder, controller, device)

        error = abs(real_depth - estimated)
        errors.append(error)

        print(f"{real_depth:>12.2f} | {estimated:>12.4f} | {error:>10.4f}")

    print("-" * 60)
    print(f"Average error: {sum(errors) / len(errors):.4f}")
    print(f"Max error: {max(errors):.4f}")


if __name__ == "__main__":
    main()
