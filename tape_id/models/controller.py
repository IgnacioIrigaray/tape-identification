"""
Parameter Controller - Predice parámetros de tape desde embeddings de audio.

Simplificado desde DeepAFx-ST para identificación de parámetros únicamente.
"""

import torch
import torch.nn as nn


class ParameterController(nn.Module):
    """
    Controlador que predice parámetros desde un embedding de audio.

    Soporta clasificación y regresión, single-param, multi-param y triple-param.

    Clasificación single-param: retorna tensor [batch, num_classes]
    Clasificación multi-param:  retorna dict {"depth": [batch, N_d], "rate": [batch, N_r]}
    Regresión single-param:     retorna tensor [batch, 1] en [0, 1]
    Regresión multi-param:      retorna dict {"depth": [batch, 1], "rate": [batch, 1]}
    Regresión triple-param:     retorna dict {"ja": [B,1], "depth": [B,1], "rate": [B,1]}

    Args:
        num_classes: Número de clases (modo clasificación single-param)
        embed_dim: Dimensión de los embeddings del encoder
        hidden_dim: Dimensión oculta del MLP
        num_classes_depth: Clases para depth (modo clasificación multi-param)
        num_classes_rate: Clases para rate (modo clasificación multi-param)
        regression: Si True, predice valores continuos en [0, 1]
        triple_param: Si True, modo 3 parámetros (JA + depth + rate)
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes_depth: int = None,
        num_classes_rate: int = None,
        regression: bool = False,
        triple_param: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.regression = regression
        self.triple_param = triple_param
        self.multi_param = (num_classes_depth is not None and num_classes_rate is not None)

        if self.triple_param:
            # Trunk compartido + 3 heads independientes (solo regresión)
            self.trunk = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.3),
            )
            self.head_ja = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.head_depth = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.head_rate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        elif self.multi_param:
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
            if regression:
                self.head_depth = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
                self.head_rate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            else:
                self.head_depth = nn.Linear(hidden_dim, num_classes_depth)
                self.head_rate = nn.Linear(hidden_dim, num_classes_rate)
        else:
            self.num_classes = num_classes
            if regression:
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
            else:
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
        if self.triple_param:
            h = self.trunk(e_y)
            return {
                "ja": self.head_ja(h),
                "depth": self.head_depth(h),
                "rate": self.head_rate(h),
            }
        if self.multi_param:
            h = self.trunk(e_y)
            return {
                "depth": self.head_depth(h),
                "rate": self.head_rate(h),
            }
        return self.mlp(e_y)
