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
)


def test_encoder():
    """Test encoder forward pass."""
    print("Testing SpectralEncoder...")
    encoder = SpectralEncoder(num_params=1, embed_dim=128, width_mult=2)
    x = torch.randn(2, 1, 24000)  # [batch, channels, samples]
    e = encoder(x)
    assert e.shape == (2, 128), f"Expected (2, 128), got {e.shape}"
    print("✓ Encoder OK")


def test_controller():
    """Test controller forward pass."""
    print("Testing ParameterController...")
    controller = ParameterController(num_params=1, embed_dim=128)
    e_x = torch.randn(2, 128)
    e_y = torch.randn(2, 128)
    p = controller(e_x, e_y)
    assert p.shape == (2, 1), f"Expected (2, 1), got {p.shape}"
    assert (p >= 0).all() and (p <= 1).all(), "Parameters should be in [0, 1]"
    print("✓ Controller OK")


def test_processor():
    """Test processor forward pass."""
    print("Testing TapeSaturationProcessor...")
    processor = TapeSaturationProcessor(min_depth=1.0, max_depth=10.0)
    x = torch.randn(2, 1, 24000)
    p = torch.rand(2, 1)  # [0, 1]
    y = processor(x, p)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    print("✓ Processor OK")


def test_apply_saturation():
    """Test saturation function."""
    print("Testing apply_tape_saturation...")
    x = torch.randn(24000)
    y = apply_tape_saturation(x, depth=5.0)
    assert y.shape == x.shape
    assert y.abs().max() <= 1.0, "Tanh should bound output to [-1, 1]"
    print("✓ Saturation function OK")


def test_full_pipeline():
    """Test full forward pass."""
    print("\nTesting full pipeline...")
    encoder = SpectralEncoder(num_params=1, embed_dim=128, width_mult=2)
    controller = ParameterController(num_params=1, embed_dim=128)
    processor = TapeSaturationProcessor(min_depth=1.0, max_depth=10.0)

    # Input audio
    x = torch.randn(2, 1, 24000)
    y_target = torch.randn(2, 1, 24000)

    # Forward
    e_x = encoder(x)
    e_y = encoder(y_target)
    p = controller(e_x, e_y)
    y_pred = processor(x, p)

    assert y_pred.shape == x.shape
    print("✓ Full pipeline OK")


if __name__ == "__main__":
    print("Running tests...\n")
    test_encoder()
    test_controller()
    test_processor()
    test_apply_saturation()
    test_full_pipeline()
    print("\n✅ All tests passed!")
