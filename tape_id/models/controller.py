"""
Parameter Controller - Predice parámetros de tape desde embeddings de audio.

Simplificado desde DeepAFx-ST para identificación de parámetros únicamente.
"""

import torch
import torch.nn as nn
from typing import Optional


class ParameterController(nn.Module):
    """
    Controlador que predice clases de parámetros desde embeddings de audio.

    Toma embeddings de input (x) y target (y) y predice logits para N clases
    que representan valores discretos del parámetro.

    Args:
        num_classes: Número de clases discretas a predecir (default: 10)
        embed_dim: Dimensión de los embeddings del encoder
        hidden_dim: Dimensión oculta del MLP predictor
        agg_method: Método de agregación de embeddings ["mlp", "linear"]
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        agg_method: str = "mlp",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.agg_method = agg_method

        # Agregación de embeddings x e y
        if agg_method == "linear":
            self.agg = nn.Linear(embed_dim * 2, embed_dim)
            mlp_in_dim = embed_dim
        elif agg_method == "mlp":
            self.agg = None  # Concatenación simple
            mlp_in_dim = embed_dim * 2
        else:
            raise ValueError(f"Invalid agg_method: {agg_method}")

        # MLP predictor de clases (logits, sin activación final)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
            # Sin Sigmoid - retorna logits para softmax en processor
        )

    def forward(
        self,
        e_x: torch.Tensor,
        e_y: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predice logits de clases desde embeddings.

        Args:
            e_x: Embedding del audio input [batch, embed_dim]
            e_y: Embedding del audio target [batch, embed_dim]
            z: Bottleneck latent (no usado, para compatibilidad)

        Returns:
            Logits para clasificación [batch, num_classes]
        """
        # Agregar embeddings
        if self.agg_method == "linear":
            e_xy = torch.cat((e_x, e_y), dim=-1)
            e_xy = self.agg(e_xy)
        else:  # mlp
            e_xy = torch.cat((e_x, e_y), dim=-1)

        # Predecir parámetros
        p = self.mlp(e_xy)

        return p
