"""
Script de evaluación para modelo entrenado.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchaudio
import argparse

from tape_id.models.encoder import SpectralEncoder
from tape_id.models.controller import ParameterController
from tape_id.models.tape_processor import HardClippingProcessor, apply_hard_clipping


def load_model(checkpoint_path: str, device: str = "cuda", num_classes: int = 3):
    """Carga modelo desde checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = SpectralEncoder(num_params=1, sample_rate=24000, embed_dim=1024, width_mult=2).to(device)
    controller = ParameterController(num_classes=num_classes, embed_dim=1024, hidden_dim=256).to(device)
    processor = HardClippingProcessor(min_gain=1.0, max_gain=4.0, num_classes=num_classes).to(device)

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
    processor,
    device: str,
) -> float:
    """
    Estima el parámetro de saturación.

    Args:
        audio: Audio limpio [samples]
        audio_saturated: Audio saturado [samples]
        encoder: Encoder model
        controller: Controller model
        processor: Processor model
        device: Device

    Returns:
        Parámetro estimado
    """
    x = audio.unsqueeze(0).unsqueeze(0).to(device)
    y = audio_saturated.unsqueeze(0).unsqueeze(0).to(device)

    e_x = encoder(x)
    e_y = encoder(y)
    logits = controller(e_x, e_y)
    param = processor.get_gain_from_logits(logits, use_argmax=True)

    return param[0].item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate tape identification model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--audio", type=str, help="Path to audio file (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of discrete classes")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    encoder, controller, processor = load_model(args.checkpoint, device, args.num_classes)
    print(f"Model loaded successfully! (num_classes={args.num_classes})")
    print(f"gain_values: {processor.gain_values.tolist()}")

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

    max_samples = 24000 * 3
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    print(f"Audio shape: {audio.shape}")

    # Test con los valores de gain del procesador
    test_params = processor.gain_values.tolist()

    print("\n" + "=" * 60)
    print("PARAMETER ESTIMATION TEST (Hard Clipping)")
    print("=" * 60)
    print(f"{'Real gain':>12} | {'Estimated':>12} | {'Error':>10}")
    print("-" * 60)

    errors = []

    for real_param in test_params:
        audio_sat = apply_hard_clipping(audio, real_param)
        estimated = estimate_parameter(audio, audio_sat, encoder, controller, processor, device)

        error = abs(real_param - estimated)
        errors.append(error)

        print(f"{real_param:>12.3f} | {estimated:>12.3f} | {error:>10.4f}")

    print("-" * 60)
    print(f"Average error: {sum(errors) / len(errors):.4f}")
    print(f"Max error: {max(errors):.4f}")


if __name__ == "__main__":
    main()
