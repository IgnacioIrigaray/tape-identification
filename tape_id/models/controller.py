"""
Parameter Controller - Predice parámetros de tape desde embeddings de audio.

Simplificado desde DeepAFx-ST para identificación de parámetros únicamente.
"""

import torch
import torch.nn as nn


class ParameterController(nn.Module):
    """
    Controlador que predice clases de parámetros desde un embedding de audio.

    Soporta modo single-param (backward compat) y multi-param (dual head).

    Single-param: retorna tensor [batch, num_classes]
    Multi-param:  retorna dict {"depth": [batch, N_d], "rate": [batch, N_r]}

    Args:
        num_classes: Número de clases (modo single-param)
        embed_dim: Dimensión de los embeddings del encoder
        hidden_dim: Dimensión oculta del MLP
        num_classes_depth: Clases para depth (modo multi-param)
        num_classes_rate: Clases para rate (modo multi-param)
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes_depth: int = None,
        num_classes_rate: int = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.multi_param = (num_classes_depth is not None and num_classes_rate is not None)

        if self.multi_param:
            self.num_classes_depth = num_classes_depth
            self.num_classes_rate = num_classes_rate
            self.trunk = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.3),
            )
            self.head_depth = nn.Linear(hidden_dim, num_classes_depth)
            self.head_rate = nn.Linear(hidden_dim, num_classes_rate)
        else:
            self.num_classes = num_classes
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, e_y: torch.Tensor):
        if self.multi_param:
            h = self.trunk(e_y)
            return {
                "depth": self.head_depth(h),
                "rate": self.head_rate(h),
            }
        return self.mlp(e_y)
