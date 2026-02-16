"""
Tests for model components: encoder, controller, processors, and full pipelines.
"""

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
    encoder = SpectralEncoder(num_params=1, sample_rate=22050, embed_dim=128, width_mult=2)
    x = torch.randn(2, 1, 22050)
    e = encoder(x)
    assert e.shape == (2, 128)


def test_controller():
    num_classes = 3
    controller = ParameterController(num_classes=num_classes, embed_dim=128)
    e_y = torch.randn(2, 128)
    logits = controller(e_y)
    assert logits.shape == (2, num_classes)


def test_processor():
    num_classes = 3
    processor = TapeSaturationProcessor(min_gain=1.0, max_gain=10.0, num_classes=num_classes)
    x = torch.randn(2, 1, 22050)
    logits = torch.randn(2, num_classes)
    y = processor(x, logits)
    assert y.shape == x.shape


def test_apply_saturation():
    x = torch.randn(22050)
    y = apply_tape_saturation(x, gain=5.0)
    assert y.shape == x.shape
    assert y.abs().max() <= 1.0


def test_hard_clipping_processor():
    num_classes = 3
    processor = HardClippingProcessor(min_gain=1.0, max_gain=4.0, num_classes=num_classes)
    x = torch.randn(2, 1, 22050)
    logits = torch.randn(2, num_classes)
    y = processor(x, logits)
    assert y.shape == x.shape
    assert y.abs().max() <= 1.0


def test_hard_clipping_bypass():
    x = torch.randn(22050).clamp(-0.5, 0.5)
    y = apply_hard_clipping(x, 1.0)
    mse = ((x - y) ** 2).mean()
    assert mse < 1e-6


def test_full_pipeline():
    num_classes = 3
    encoder = SpectralEncoder(num_params=1, sample_rate=22050, embed_dim=128, width_mult=2)
    controller = ParameterController(num_classes=num_classes, embed_dim=128)

    y = torch.randn(2, 1, 22050)
    e_y = encoder(y)
    logits = controller(e_y)

    assert logits.shape == (2, num_classes)

    class_idx = torch.tensor([0, 2])
    loss = torch.nn.functional.cross_entropy(logits, class_idx)
    assert loss.item() > 0


def test_wow_flutter_basic():
    from tape_id.data.dataset import wow_flutter
    x = torch.randn(1, 22050)
    x = x / (x.abs().max() + 1e-8)
    y = wow_flutter(x, depth=0.5, sample_rate=22050)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_wow_flutter_bypass():
    from tape_id.data.dataset import wow_flutter
    x = torch.randn(1, 22050)
    x = x / (x.abs().max() + 1e-8)
    y = wow_flutter(x, depth=0.001, sample_rate=22050, enable_ou=False)
    mse = ((x - y) ** 2).mean()
    assert mse < 0.01


def test_wow_flutter_lagrange3():
    from tape_id.data.dataset import wow_flutter
    x = torch.randn(1, 22050)
    x = x / (x.abs().max() + 1e-8)
    y = wow_flutter(x, depth=0.5, sample_rate=22050, interpolation="lagrange3", enable_ou=False)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_controller_regression():
    controller = ParameterController(num_classes=3, embed_dim=128, regression=True)
    assert controller.regression is True
    e_y = torch.randn(4, 128)
    out = controller(e_y)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_controller_regression_multi():
    controller = ParameterController(
        embed_dim=128, hidden_dim=64,
        num_classes_depth=3, num_classes_rate=3,
        regression=True,
    )
    assert controller.regression is True
    assert controller.multi_param is True
    e_y = torch.randn(4, 128)
    out = controller(e_y)
    assert isinstance(out, dict)
    assert out["depth"].shape == (4, 1)
    assert out["rate"].shape == (4, 1)
    assert (out["depth"] >= 0).all() and (out["depth"] <= 1).all()
    assert (out["rate"] >= 0).all() and (out["rate"] <= 1).all()


def test_full_pipeline_regression():
    encoder = SpectralEncoder(num_params=1, sample_rate=22050, embed_dim=128, width_mult=2)
    controller = ParameterController(num_classes=3, embed_dim=128, regression=True)

    y = torch.randn(2, 1, 22050)
    e_y = encoder(y)
    pred = controller(e_y)

    assert pred.shape == (2, 1)

    target = torch.rand(2, 1)
    loss = torch.nn.functional.mse_loss(pred, target)
    assert loss.item() >= 0
    loss.backward()


def test_controller_triple_param():
    controller = ParameterController(
        embed_dim=128, hidden_dim=64,
        triple_param=True,
    )
    assert controller.triple_param is True
    e_y = torch.randn(4, 128)
    out = controller(e_y)
    assert isinstance(out, dict)
    for key in ["ja", "depth", "rate"]:
        assert key in out
        assert out[key].shape == (4, 1)
        assert (out[key] >= 0).all() and (out[key] <= 1).all()


def test_full_pipeline_triple():
    encoder = SpectralEncoder(num_params=1, sample_rate=22050, embed_dim=128, width_mult=2)
    controller = ParameterController(embed_dim=128, hidden_dim=64, triple_param=True)

    y = torch.randn(2, 1, 22050)
    e_y = encoder(y)
    pred = controller(e_y)

    assert isinstance(pred, dict)
    target_ja = torch.rand(2, 1)
    target_d = torch.rand(2, 1)
    target_r = torch.rand(2, 1)

    loss = (torch.nn.functional.mse_loss(pred["ja"], target_ja)
            + torch.nn.functional.mse_loss(pred["depth"], target_d)
            + torch.nn.functional.mse_loss(pred["rate"], target_r))
    assert loss.item() >= 0
    loss.backward()
