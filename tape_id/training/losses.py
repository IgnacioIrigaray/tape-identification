"""
Loss functions para entrenamiento de identificación de parámetros.

Incluye MR-STFT loss para comparación de audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss.

    Calcula loss espectral en múltiples resoluciones.
    Adaptado de https://github.com/csteinmetz1/auraloss
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        w_sc=0.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calcula STFT loss entre x e y.

        Args:
            x: Audio predicho [batch, samples]
            y: Audio target [batch, samples]

        Returns:
            Loss escalar
        """
        total_loss = 0.0

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Ventana de Hann
            window = torch.hann_window(win_length).to(x.device)

            # STFT
            x_stft = torch.stft(
                x,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
            y_stft = torch.stft(
                y,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            # Magnitud
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)

            # Spectral convergence
            if self.w_sc > 0:
                sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(
                    y_mag, p="fro"
                )
                total_loss += self.w_sc * sc_loss

            # Log magnitude
            if self.w_log_mag > 0:
                log_mag_loss = F.l1_loss(torch.log(x_mag + 1e-8), torch.log(y_mag + 1e-8))
                total_loss += self.w_log_mag * log_mag_loss

            # Linear magnitude
            if self.w_lin_mag > 0:
                lin_mag_loss = F.l1_loss(x_mag, y_mag)
                total_loss += self.w_lin_mag * lin_mag_loss

        return total_loss / len(self.fft_sizes)
