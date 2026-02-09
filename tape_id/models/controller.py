"""
Parameter Controller - Predice parámetros de tape desde embeddings de audio.

Simplificado desde DeepAFx-ST para identificación de parámetros únicamente.
"""

import torch
import torch.nn as nn


class ParameterController(nn.Module):
    """
    Controlador que predice clases de parámetros desde un embedding de audio.

    Toma el embedding del audio saturado y predice logits para N clases
    que representan valores discretos del parámetro.

    Args:
        num_classes: Número de clases discretas a predecir (default: 10)
        embed_dim: Dimensión de los embeddings del encoder
        hidden_dim: Dimensión oculta del MLP predictor
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # MLP predictor de clases (logits, sin activación final)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, e_y: torch.Tensor) -> torch.Tensor:
        """
        Predice logits de clases desde embedding del audio saturado.

        Args:
            e_y: Embedding del audio saturado [batch, embed_dim]

        Returns:
            Logits para clasificación [batch, num_classes]
        """
        return self.mlp(e_y)
