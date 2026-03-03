"""
Parameter Controller - Predice parámetros de tape desde embeddings de audio.

Simplificado desde DeepAFx-ST para identificación de parámetros únicamente.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ParameterController(nn.Module):
    """
    Controlador que predice parámetros desde un embedding de audio.

    Soporta clasificación y regresión, single-param, multi-param y N-param.

    Clasificación single-param: retorna tensor [batch, num_classes]
    Clasificación multi-param:  retorna dict {"depth": [batch, N_d], "rate": [batch, N_r]}
    Regresión single-param:     retorna tensor [batch, 1]
    Regresión multi-param:      retorna dict {"depth": [batch, 1], "rate": [batch, 1]}
    Regresión N-param:          retorna dict {name: [B,1] for name in param_names}

    Args:
        num_classes: Número de clases (modo clasificación single-param)
        embed_dim: Dimensión de los embeddings del encoder
        hidden_dim: Dimensión oculta del MLP
        num_classes_depth: Clases para depth (modo clasificación multi-param)
        num_classes_rate: Clases para rate (modo clasificación multi-param)
        regression: Si True, predice valores continuos
        param_names: Lista de nombres de parámetros para N-param regression
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes_depth: int = None,
        num_classes_rate: int = None,
        regression: bool = False,
        param_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.regression = regression
        self.param_names = param_names
        self.multi_param = (num_classes_depth is not None and num_classes_rate is not None)

        if self.param_names is not None:
            # N-param regression: shared trunk + N heads
            self.trunk = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1),
            )
            self.heads = nn.ModuleDict({
                name: nn.Linear(hidden_dim, 1) for name in param_names
            })
        elif self.multi_param:
            self.num_classes_depth = num_classes_depth
            self.num_classes_rate = num_classes_rate
            self.trunk = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1),
            )
            if regression:
                self.head_depth = nn.Linear(hidden_dim, 1)
                self.head_rate = nn.Linear(hidden_dim, 1)
            else:
                self.head_depth = nn.Linear(hidden_dim, num_classes_depth)
                self.head_rate = nn.Linear(hidden_dim, num_classes_rate)
        else:
            self.num_classes = num_classes
            if regression:
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1),
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes),
                )

    def forward(self, e_y: torch.Tensor):
        if self.param_names is not None:
            h = self.trunk(e_y)
            return {name: head(h) for name, head in self.heads.items()}
        if self.multi_param:
            h = self.trunk(e_y)
            return {
                "depth": self.head_depth(h),
                "rate":  self.head_rate(h),
            }
        return self.mlp(e_y)
