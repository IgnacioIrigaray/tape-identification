"""
Tape Saturation Processor - Modelo diferenciable de saturación tipo tape.

Este módulo implementa un procesador de saturación simple usando tanh.
"""

import torch
import torch.nn as nn


class TapeSaturationProcessor(nn.Module):
    """
    Procesador de saturación tipo tape diferenciable.

    Args:
        min_depth: Ganancia mínima (default: 1.0)
        max_depth: Ganancia máxima (default: 10.0)
    """

    def __init__(self, min_depth: float = 1.0, max_depth: float = 10.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_params = 1  # Solo un parámetro: depth

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Aplica saturación con parámetro normalizado.

        Args:
            x: Audio tensor [batch, channels, samples]
            p: Parámetro normalizado [batch, 1] en rango [0, 1]

        Returns:
            Audio saturado [batch, channels, samples]
        """
        # Desnormalizar parámetro: [0, 1] -> [min_depth, max_depth]
        depth = self.min_depth + p * (self.max_depth - self.min_depth)
        depth = depth.unsqueeze(-1)  # [batch, 1, 1]

        # Aplicar saturación
        saturated = torch.tanh(x * depth)

        return saturated


def apply_tape_saturation(x: torch.Tensor, depth: float) -> torch.Tensor:
    """
    Función helper para aplicar saturación con depth específico.

    Args:
        x: Audio tensor
        depth: Ganancia de saturación [1, 10]

    Returns:
        Audio saturado
    """
    return torch.tanh(x * depth)
