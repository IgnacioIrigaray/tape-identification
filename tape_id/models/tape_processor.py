"""
Tape Saturation Processor - Modelo diferenciable de saturación tipo tape.

Este módulo implementa un procesador de saturación simple usando tanh.
Soporta clasificación discreta con N clases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TapeSaturationProcessor(nn.Module):
    """
    Procesador de saturación tipo tape diferenciable con N clases discretas.

    Args:
        min_gain: Ganancia mínima (default: 1.0)
        max_gain: Ganancia máxima (default: 10.0)
        num_classes: Número de clases discretas (default: 10)
    """

    def __init__(
        self,
        min_gain: float = 1.0,
        max_gain: float = 10.0,
        num_classes: int = 10,
    ):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.num_classes = num_classes

        # Valores de gain para cada clase (buffer para que se mueva con el modelo)
        gain_values = torch.linspace(min_gain, max_gain, num_classes)
        self.register_buffer('gain_values', gain_values)

    def forward(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """
        Aplica saturación con clasificación discreta.

        Args:
            x: Audio tensor [batch, channels, samples]
            logits: Logits de clasificación [batch, num_classes]
            use_argmax: Si True, usa argmax puro (inferencia).
                        Si False, usa straight-through estimator (training).

        Returns:
            Audio saturado [batch, channels, samples]
        """
        if use_argmax:
            # Inferencia: valor discreto exacto
            class_idx = torch.argmax(logits, dim=-1)  # [batch]
            gain = self.gain_values[class_idx]  # [batch]
        else:
            # Training: Straight-through estimator
            # Forward: usa valor discreto (argmax)
            # Backward: gradientes fluyen a través de softmax
            probs = F.softmax(logits, dim=-1)  # [batch, num_classes]

            # One-hot del argmax
            hard = F.one_hot(probs.argmax(-1), self.num_classes).float()

            # Straight-through: forward usa hard, backward usa probs
            probs_st = probs + (hard - probs).detach()

            gain = (probs_st * self.gain_values).sum(dim=-1)  # [batch]

        # Expandir para broadcasting: [batch] -> [batch, 1, 1]
        gain = gain.unsqueeze(-1).unsqueeze(-1)

        # Aplicar saturación
        saturated = torch.tanh(x * gain)

        return saturated

    def get_gain_from_logits(
        self,
        logits: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """
        Obtiene el valor de gain desde logits (para logging).

        Args:
            use_argmax: Si True, usa argmax. Si False, usa straight-through
                        (que en forward da el mismo resultado que argmax).

        Returns:
            gain: [batch] valores de gain
        """
        # Con straight-through, training y inference dan el mismo valor discreto
        class_idx = torch.argmax(logits, dim=-1)
        return self.gain_values[class_idx]


def apply_tape_saturation(x: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Función helper para aplicar saturación con gain específico.

    Args:
        x: Audio tensor
        gain: Ganancia de saturación [1, 10]

    Returns:
        Audio saturado
    """
    return torch.tanh(x * gain)


def langevin(x: torch.Tensor) -> torch.Tensor:
    """
    Función de Langevin: L(x) = coth(x) - 1/x

    Es la función que describe la magnetización anhisterética en el modelo
    Jiles-Atherton. Para x pequeño: L(x) ≈ x/3
    """
    small = torch.abs(x) < 0.01
    result = torch.zeros_like(x)

    # Aproximación Taylor para valores pequeños
    result[small] = x[small] / 3.0

    # Fórmula exacta para valores grandes
    large = ~small
    x_large = x[large]
    result[large] = 1.0 / torch.tanh(x_large) - 1.0 / x_large

    return result


def apply_ja_saturation(x: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Aplica saturación usando modelo Jiles-Atherton normalizado.

    Internamente usa: y = 3a * L(x/a) donde a = 1/gain

    Args:
        x: Audio tensor
        gain: Ganancia (gain alto = más saturación, gain bajo = bypass)

    Returns:
        Audio saturado
    """
    a = 1.0 / gain
    return 3.0 * a * langevin(x / a)


class JilesAthertonProcessor(nn.Module):
    """
    Procesador de saturación basado en el modelo Jiles-Atherton.

    Usa la curva anhisterética de magnetización para simular saturación
    de cinta magnética de forma más realista que tanh.

    Fórmula interna: y = 3a * L(x/a) donde L(x) = coth(x) - 1/x
    Pero externamente usa 'gain' donde: a = 1/gain

    Así: gain alto = más saturación (como tanh tradicional)

    Args:
        min_gain: Ganancia mínima (menos saturación)
        max_gain: Ganancia máxima (más saturación)
        num_classes: Número de clases discretas
    """

    def __init__(
        self,
        min_gain: float = 2.0,
        max_gain: float = 5.0,
        num_classes: int = 3,
    ):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.num_classes = num_classes

        # Valores de gain para cada clase (escala lineal)
        gain_values = torch.linspace(min_gain, max_gain, num_classes)
        self.register_buffer('gain_values', gain_values)

    def _langevin(self, x: torch.Tensor) -> torch.Tensor:
        """Función de Langevin vectorizada."""
        small = torch.abs(x) < 0.01
        result = torch.zeros_like(x)
        result[small] = x[small] / 3.0
        large = ~small
        x_large = x[large]
        result[large] = 1.0 / torch.tanh(x_large) - 1.0 / x_large
        return result

    def _apply_saturation(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """Aplica saturación JA con gain (internamente a = 1/gain)."""
        # gain tiene shape [batch, 1, 1], x tiene shape [batch, channels, samples]
        a = 1.0 / gain
        return 3.0 * a * self._langevin(x / a)

    def forward(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """
        Aplica saturación Jiles-Atherton con clasificación discreta.

        Args:
            x: Audio tensor [batch, channels, samples]
            logits: Logits de clasificación [batch, num_classes]
            use_argmax: Si True, usa argmax (inferencia). Si False, straight-through.

        Returns:
            Audio saturado [batch, channels, samples]
        """
        if use_argmax:
            class_idx = torch.argmax(logits, dim=-1)
            gain = self.gain_values[class_idx]
        else:
            # Straight-through estimator
            probs = F.softmax(logits, dim=-1)
            hard = F.one_hot(probs.argmax(-1), self.num_classes).float()
            probs_st = probs + (hard - probs).detach()
            gain = (probs_st * self.gain_values).sum(dim=-1)

        # Expandir: [batch] -> [batch, 1, 1]
        gain = gain.unsqueeze(-1).unsqueeze(-1)

        return self._apply_saturation(x, gain)

    def get_gain_from_logits(
        self,
        logits: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """Obtiene el valor de gain desde logits (para logging)."""
        class_idx = torch.argmax(logits, dim=-1)
        return self.gain_values[class_idx]


def apply_hard_clipping(x: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Aplica hard clipping con ganancia.

    Fórmula: y = clamp(x * gain, -1, 1)

    Args:
        x: Audio tensor
        gain: Ganancia antes del clipping (gain > 1 produce clipping)

    Returns:
        Audio con hard clipping aplicado
    """
    return torch.clamp(x * gain, -1.0, 1.0)


class HardClippingProcessor(nn.Module):
    """
    Procesador de hard clipping con clasificación discreta.

    El modelo más simple de saturación/distorsión:
    y = clamp(x * gain, -1, 1)

    Características:
    - gain = 1.0: sin distorsión (bypass)
    - gain > 1.0: los picos que excedan ±1/gain son recortados
    - Diferencias muy marcadas entre valores de gain

    Args:
        min_gain: Ganancia mínima (default: 1.0 = bypass)
        max_gain: Ganancia máxima (default: 4.0 = clipping severo)
        num_classes: Número de clases discretas
    """

    def __init__(
        self,
        min_gain: float = 1.0,
        max_gain: float = 4.0,
        num_classes: int = 3,
    ):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.num_classes = num_classes

        # Valores de gain para cada clase
        gain_values = torch.linspace(min_gain, max_gain, num_classes)
        self.register_buffer('gain_values', gain_values)

    def forward(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """
        Aplica hard clipping con clasificación discreta.

        Args:
            x: Audio tensor [batch, channels, samples]
            logits: Logits de clasificación [batch, num_classes]
            use_argmax: Si True, usa argmax (inferencia). Si False, straight-through.

        Returns:
            Audio con hard clipping [batch, channels, samples]
        """
        if use_argmax:
            class_idx = torch.argmax(logits, dim=-1)
            gain = self.gain_values[class_idx]
        else:
            # Straight-through estimator
            probs = F.softmax(logits, dim=-1)
            hard = F.one_hot(probs.argmax(-1), self.num_classes).float()
            probs_st = probs + (hard - probs).detach()
            gain = (probs_st * self.gain_values).sum(dim=-1)

        # Expandir: [batch] -> [batch, 1, 1]
        gain = gain.unsqueeze(-1).unsqueeze(-1)

        # Hard clipping
        return torch.clamp(x * gain, -1.0, 1.0)

    def get_gain_from_logits(
        self,
        logits: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """Obtiene el valor de gain desde logits (para logging)."""
        class_idx = torch.argmax(logits, dim=-1)
        return self.gain_values[class_idx]
