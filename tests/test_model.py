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


def test_wow_flutter_basic():
    """Test wow_flutter function produces output of correct shape."""
    print("Testing wow_flutter basic...")
    from tape_id.data.dataset import wow_flutter
    x = torch.randn(1, 22050)
    x = x / (x.abs().max() + 1e-8)
    y = wow_flutter(x, depth=0.5, sample_rate=22050)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    print("✓ Wow/flutter basic OK")


def test_wow_flutter_bypass():
    """Test that depth~0 without OU is approximately bypass."""
    print("Testing wow_flutter bypass (depth~0)...")
    from tape_id.data.dataset import wow_flutter
    x = torch.randn(1, 22050)
    x = x / (x.abs().max() + 1e-8)
    y = wow_flutter(x, depth=0.001, sample_rate=22050, enable_ou=False)
    mse = ((x - y) ** 2).mean()
    assert mse < 0.01, f"depth~0 should be near-bypass, but MSE={mse}"
    print(f"✓ Wow/flutter bypass OK (MSE={mse:.6f})")


def test_wow_flutter_lagrange3():
    """Test Lagrange3 interpolation mode."""
    print("Testing wow_flutter lagrange3...")
    from tape_id.data.dataset import wow_flutter
    x = torch.randn(1, 22050)
    x = x / (x.abs().max() + 1e-8)
    y = wow_flutter(x, depth=0.5, sample_rate=22050, interpolation="lagrange3", enable_ou=False)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print("✓ Wow/flutter lagrange3 OK")


def test_controller_regression():
    """Test controller regression mode: output [batch, 1] in [0, 1]."""
    print("Testing ParameterController regression...")
    controller = ParameterController(num_classes=3, embed_dim=128, regression=True)
    assert controller.regression is True
    e_y = torch.randn(4, 128)
    out = controller(e_y)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), f"Sigmoid output should be in [0,1], got min={out.min():.4f} max={out.max():.4f}"
    print("✓ Controller regression OK")


def test_controller_regression_multi():
    """Test controller regression multi-param: dict output with [batch, 1] values."""
    print("Testing ParameterController regression multi-param...")
    controller = ParameterController(
        embed_dim=128, hidden_dim=64,
        num_classes_depth=3, num_classes_rate=3,
        regression=True,
    )
    assert controller.regression is True
    assert controller.multi_param is True
    e_y = torch.randn(4, 128)
    out = controller(e_y)
    assert isinstance(out, dict), f"Expected dict, got {type(out)}"
    assert out["depth"].shape == (4, 1), f"Expected (4, 1), got {out['depth'].shape}"
    assert out["rate"].shape == (4, 1), f"Expected (4, 1), got {out['rate'].shape}"
    assert (out["depth"] >= 0).all() and (out["depth"] <= 1).all()
    assert (out["rate"] >= 0).all() and (out["rate"] <= 1).all()
    print("✓ Controller regression multi-param OK")


def test_full_pipeline_regression():
    """Test full pipeline with regression: encoder -> controller -> MSE loss."""
    print("Testing full pipeline regression...")
    encoder = SpectralEncoder(num_params=1, sample_rate=22050, embed_dim=128, width_mult=2)
    controller = ParameterController(num_classes=3, embed_dim=128, regression=True)

    y = torch.randn(2, 1, 22050)
    e_y = encoder(y)
    pred = controller(e_y)

    assert pred.shape == (2, 1)

    target = torch.rand(2, 1)
    loss = torch.nn.functional.mse_loss(pred, target)
    assert loss.item() >= 0, "MSE loss should be non-negative"
    loss.backward()
    print("✓ Full pipeline regression OK")


if __name__ == "__main__":
    print("Running tests...\n")
    test_encoder()
    test_controller()
    test_processor()
    test_apply_saturation()
    test_hard_clipping_processor()
    test_hard_clipping_bypass()
    test_full_pipeline()
    test_wow_flutter_basic()
    test_wow_flutter_bypass()
    test_wow_flutter_lagrange3()
    test_controller_regression()
    test_controller_regression_multi()
    test_full_pipeline_regression()
    print("\n✅ All tests passed!")
