"""
Tape Saturation Processors - Differentiable audio processors with discrete classification.

Includes: TapeSaturationProcessor (tanh), JilesAthertonProcessor, HardClippingProcessor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def straight_through_select(logits: torch.Tensor, values: torch.Tensor,
                            num_classes: int, use_argmax: bool = False) -> torch.Tensor:
    """Select a value from discrete options using straight-through estimator.

    During training (use_argmax=False): forward uses argmax, backward uses softmax gradients.
    During inference (use_argmax=True): pure argmax selection.

    Args:
        logits: Classification logits [batch, num_classes]
        values: Value for each class [num_classes]
        num_classes: Number of discrete classes
        use_argmax: If True, use pure argmax (no gradient flow)

    Returns:
        Selected values [batch]
    """
    if use_argmax:
        return values[torch.argmax(logits, dim=-1)]

    probs = F.softmax(logits, dim=-1)
    hard = F.one_hot(probs.argmax(-1), num_classes).float()
    probs_st = probs + (hard - probs).detach()
    return (probs_st * values).sum(dim=-1)


def gain_from_logits(logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Get discrete gain value from logits (for logging/inference).

    Args:
        logits: Classification logits [batch, num_classes]
        values: Gain value for each class [num_classes]

    Returns:
        Selected gain values [batch]
    """
    return values[torch.argmax(logits, dim=-1)]


class TapeSaturationProcessor(nn.Module):
    """Differentiable tanh saturation with N discrete gain classes.

    y = tanh(x * gain)

    Args:
        min_gain: Minimum gain value
        max_gain: Maximum gain value
        num_classes: Number of discrete classes
    """

    def __init__(self, min_gain=1.0, max_gain=10.0, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('gain_values', torch.linspace(min_gain, max_gain, num_classes))

    def forward(self, x, logits, use_argmax=False):
        gain = straight_through_select(logits, self.gain_values, self.num_classes, use_argmax)
        gain = gain.unsqueeze(-1).unsqueeze(-1)
        return torch.tanh(x * gain)

    def get_gain_from_logits(self, logits):
        return gain_from_logits(logits, self.gain_values)


def apply_tape_saturation(x: torch.Tensor, gain: float) -> torch.Tensor:
    """Apply tanh saturation: y = tanh(x * gain)."""
    return torch.tanh(x * gain)


def langevin(x: torch.Tensor) -> torch.Tensor:
    """Langevin function: L(x) = coth(x) - 1/x.

    Describes anhysteretic magnetization in Jiles-Atherton model.
    For small x: L(x) ~ x/3 (Taylor approximation).
    """
    small = torch.abs(x) < 0.01
    result = torch.zeros_like(x)
    result[small] = x[small] / 3.0
    large = ~small
    x_large = x[large]
    result[large] = 1.0 / torch.tanh(x_large) - 1.0 / x_large
    return result


def apply_ja_saturation(x: torch.Tensor, gain: float) -> torch.Tensor:
    """Apply Jiles-Atherton anhysteretic saturation: y = 3a * L(x/a) where a = 1/gain."""
    a = 1.0 / gain
    return 3.0 * a * langevin(x / a)


class JilesAthertonProcessor(nn.Module):
    """Jiles-Atherton anhysteretic saturation with N discrete gain classes.

    Uses the anhysteretic magnetization curve for more realistic tape saturation.
    y = 3a * L(x/a) where L is the Langevin function, a = 1/gain.

    Args:
        min_gain: Minimum gain (less saturation)
        max_gain: Maximum gain (more saturation)
        num_classes: Number of discrete classes
    """

    def __init__(self, min_gain=2.0, max_gain=5.0, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('gain_values', torch.linspace(min_gain, max_gain, num_classes))

    def forward(self, x, logits, use_argmax=False):
        gain = straight_through_select(logits, self.gain_values, self.num_classes, use_argmax)
        gain = gain.unsqueeze(-1).unsqueeze(-1)
        a = 1.0 / gain
        return 3.0 * a * langevin(x / a)

    def get_gain_from_logits(self, logits):
        return gain_from_logits(logits, self.gain_values)


def apply_hard_clipping(x: torch.Tensor, gain: float) -> torch.Tensor:
    """Apply hard clipping: y = clamp(x * gain, -1, 1)."""
    return torch.clamp(x * gain, -1.0, 1.0)


class HardClippingProcessor(nn.Module):
    """Hard clipping with N discrete gain classes.

    y = clamp(x * gain, -1, 1)
    gain = 1.0 is bypass, gain > 1.0 produces clipping.

    Args:
        min_gain: Minimum gain (1.0 = bypass)
        max_gain: Maximum gain
        num_classes: Number of discrete classes
    """

    def __init__(self, min_gain=1.0, max_gain=4.0, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('gain_values', torch.linspace(min_gain, max_gain, num_classes))

    def forward(self, x, logits, use_argmax=False):
        gain = straight_through_select(logits, self.gain_values, self.num_classes, use_argmax)
        gain = gain.unsqueeze(-1).unsqueeze(-1)
        return torch.clamp(x * gain, -1.0, 1.0)

    def get_gain_from_logits(self, logits):
        return gain_from_logits(logits, self.gain_values)
