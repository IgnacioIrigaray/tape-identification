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


class DifferentiableForwardModel(nn.Module):
    """Apply a degradation with continuous predicted parameters in [0, 1].

    Converts normalised predictions to physical parameter space and applies
    a differentiable approximation of the degradation for signal-loss computation.

    Supported degradation_model values: "tanh", "ja", "hard_clipping", "ja_wf".

    For "ja_wf" (triple-param), applies JA anhysteretic saturation followed by a
    sinusoidal wow/flutter LFO with torch.lerp interpolation.

    Args:
        degradation_model: Name of the degradation model.
        min_param: Minimum physical parameter (JA drive / tanh gain / clip gain).
        max_param: Maximum physical parameter.
        min_depth: Minimum wow/flutter depth (ja_wf only).
        max_depth: Maximum wow/flutter depth (ja_wf only).
        min_rate: Minimum wow rate (ja_wf only).
        max_rate: Maximum wow rate (ja_wf only).
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(
        self,
        degradation_model: str,
        min_param: float,
        max_param: float,
        min_depth: float = 0.1,
        max_depth: float = 0.8,
        min_rate: float = 0.1,
        max_rate: float = 0.8,
        sample_rate: int = 22050,
    ):
        super().__init__()
        self.degradation_model = degradation_model
        self.min_param = min_param
        self.max_param = max_param
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.sample_rate = sample_rate

    def forward(self, x_clean: torch.Tensor, pred) -> torch.Tensor:
        """Apply degradation differentiably.

        Args:
            x_clean: Clean audio [batch, 1, samples].
            pred: Normalised predictions in [0, 1].
                  Tensor [B, 1] for single-param modes.
                  Dict {"ja": [B,1], "depth": [B,1], "rate": [B,1]} for "ja_wf".

        Returns:
            Degraded audio [batch, 1, samples].
        """
        if self.degradation_model == "ja_wf":
            return self._apply_ja_wf(x_clean, pred)
        elif self.degradation_model == "wf_ja":
            return self._apply_wf_ja(x_clean, pred)
        elif self.degradation_model == "ja":
            return self._apply_ja(x_clean, pred)
        elif self.degradation_model == "hard_clipping":
            return self._apply_hard_clipping(x_clean, pred)
        else:  # "tanh"
            return self._apply_tanh(x_clean, pred)

    def _apply_tanh(self, x, pred):
        gain = self.min_param + pred * (self.max_param - self.min_param)  # [B, 1]
        gain = gain.unsqueeze(-1)                                          # [B, 1, 1]
        return torch.tanh(x * gain)

    def _apply_ja(self, x, pred):
        gain = self.min_param + pred * (self.max_param - self.min_param)  # [B, 1]
        gain = gain.unsqueeze(-1)                                          # [B, 1, 1]
        a = 1.0 / gain
        return 3.0 * a * langevin(x / a)

    def _apply_hard_clipping(self, x, pred):
        gain = self.min_param + pred * (self.max_param - self.min_param)  # [B, 1]
        gain = gain.unsqueeze(-1)                                          # [B, 1, 1]
        return torch.clamp(x * gain, -1.0, 1.0)

    def _apply_ja_wf(self, x, pred):
        # JA saturation (differentiable anhysteretic approximation)
        gain = self.min_param + pred["ja"] * (self.max_param - self.min_param)  # [B, 1]
        gain = gain.unsqueeze(-1)                                                # [B, 1, 1]
        a = 1.0 / gain
        y = 3.0 * a * langevin(x / a)

        # Sinusoidal wow/flutter with torch.lerp
        y = self._apply_wow_flutter_batch(y, pred["depth"], pred["rate"])
        return y

    def _apply_wf_ja(self, x, pred):
        # Sinusoidal wow/flutter first, then JA saturation
        y = self._apply_wow_flutter_batch(x, pred["depth"], pred["rate"])
        gain = self.min_param + pred["ja"] * (self.max_param - self.min_param)  # [B, 1]
        gain = gain.unsqueeze(-1)                                                # [B, 1, 1]
        a = 1.0 / gain
        return 3.0 * a * langevin(y / a)

    def _apply_wow_flutter_batch(self, y, depth_norm, rate_norm):
        """Apply sinusoidal wow/flutter per item in batch (differentiable w.r.t. depth, rate)."""
        B, _, S = y.shape
        out = torch.zeros_like(y)
        for i in range(B):
            depth = self.min_depth + depth_norm[i, 0] * (self.max_depth - self.min_depth)
            rate = self.min_rate + rate_norm[i, 0] * (self.max_rate - self.min_rate)
            out[i] = self._sinusoidal_wow_flutter(y[i], depth, rate, S)
        return out

    def _sinusoidal_wow_flutter(self, x, depth, rate, num_samples):
        """Differentiable sinusoidal LFO with linear interpolation.

        Args:
            x: Audio [1, samples].
            depth: Wow depth (physical), scalar tensor — differentiable.
            rate: Wow rate (physical), scalar tensor — differentiable.
            num_samples: Number of samples.

        Returns:
            Time-warped audio [1, samples].
        """
        import math
        # Exponential mapping rate → frequency (mirrors dataset mapping)
        wow_freq = torch.pow(torch.tensor(4.5, device=x.device, dtype=x.dtype), rate) - 1.0
        amplitude = depth * self.sample_rate * 0.005  # max ~5 ms delay

        t = torch.arange(num_samples, dtype=x.dtype, device=x.device)
        phase = 2.0 * math.pi * wow_freq * t / self.sample_rate
        lfo = amplitude * torch.cos(phase) + amplitude  # always >= 0

        # Fractional delay indices
        indices = t - lfo
        indices = torch.clamp(indices, 0.0, float(num_samples - 1))
        idx_floor = indices.long()
        idx_ceil = torch.clamp(idx_floor + 1, max=num_samples - 1)
        frac = indices - idx_floor.float()

        x_flat = x.squeeze(0)
        y_flat = torch.lerp(x_flat[idx_floor], x_flat[idx_ceil], frac)
        return y_flat.unsqueeze(0)
