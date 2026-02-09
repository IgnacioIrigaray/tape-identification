"""
Tests básicos para verificar que los modelos funcionen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tape_id.models import (
    SpectralEncoder,
    ParameterController,
    TapeSaturationProcessor,
    apply_tape_saturation,
    HardClippingProcessor,
    apply_hard_clipping,
)


def test_encoder():
    """Test encoder forward pass."""
    print("Testing SpectralEncoder...")
    encoder = SpectralEncoder(num_params=1, sample_rate=22050, embed_dim=128, width_mult=2)
    x = torch.randn(2, 1, 22050)  # [batch, channels, samples]
    e = encoder(x)
    assert e.shape == (2, 128), f"Expected (2, 128), got {e.shape}"
    print("✓ Encoder OK")


def test_controller():
    """Test controller forward pass (single embedding)."""
    print("Testing ParameterController...")
    num_classes = 3
    controller = ParameterController(num_classes=num_classes, embed_dim=128)
    e_y = torch.randn(2, 128)
    logits = controller(e_y)
    assert logits.shape == (2, num_classes), f"Expected (2, {num_classes}), got {logits.shape}"
    print("✓ Controller OK")


def test_processor():
    """Test processor forward pass."""
    print("Testing TapeSaturationProcessor...")
    num_classes = 3
    processor = TapeSaturationProcessor(min_gain=1.0, max_gain=10.0, num_classes=num_classes)
    x = torch.randn(2, 1, 22050)
    logits = torch.randn(2, num_classes)  # [batch, num_classes]
    y = processor(x, logits)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    print("✓ Processor OK")


def test_apply_saturation():
    """Test saturation function."""
    print("Testing apply_tape_saturation...")
    x = torch.randn(22050)
    y = apply_tape_saturation(x, gain=5.0)
    assert y.shape == x.shape
    assert y.abs().max() <= 1.0, "Tanh should bound output to [-1, 1]"
    print("✓ Saturation function OK")


def test_hard_clipping_processor():
    """Test hard clipping processor."""
    print("Testing HardClippingProcessor...")
    num_classes = 3
    processor = HardClippingProcessor(min_gain=1.0, max_gain=4.0, num_classes=num_classes)
    x = torch.randn(2, 1, 22050)
    logits = torch.randn(2, num_classes)
    y = processor(x, logits)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    assert y.abs().max() <= 1.0, "Hard clipping should bound output to [-1, 1]"
    print("✓ HardClippingProcessor OK")


def test_hard_clipping_bypass():
    """Test that gain=1 is bypass."""
    print("Testing hard clipping bypass (gain=1)...")
    x = torch.randn(22050)
    y = apply_hard_clipping(x.clamp(-0.5, 0.5), 1.0)  # Use small input to avoid clipping
    x_clamped = x.clamp(-0.5, 0.5)
    mse = ((x_clamped - y) ** 2).mean()
    assert mse < 1e-6, f"gain=1 should be bypass, but MSE={mse}"
    print("✓ Hard clipping bypass OK")


def test_full_pipeline():
    """Test full forward pass (single-input classification)."""
    print("\nTesting full pipeline...")
    num_classes = 3
    encoder = SpectralEncoder(num_params=1, sample_rate=22050, embed_dim=128, width_mult=2)
    controller = ParameterController(num_classes=num_classes, embed_dim=128)

    # Audio saturado
    y = torch.randn(2, 1, 22050)

    # Forward: solo audio saturado
    e_y = encoder(y)
    logits = controller(e_y)

    assert logits.shape == (2, num_classes), f"Expected (2, {num_classes}), got {logits.shape}"

    # Verificar que se puede calcular CrossEntropyLoss
    class_idx = torch.tensor([0, 2])
    loss = torch.nn.functional.cross_entropy(logits, class_idx)
    assert loss.item() > 0, "Loss should be positive"

    print("✓ Full pipeline OK")


if __name__ == "__main__":
    print("Running tests...\n")
    test_encoder()
    test_controller()
    test_processor()
    test_apply_saturation()
    test_hard_clipping_processor()
    test_hard_clipping_bypass()
    test_full_pipeline()
    print("\n✅ All tests passed!")
