"""
Audio Encoder - Based on DeepAFx-ST.

Extracts embeddings from audio using MobileNetV2 or EfficientNet-B2 on spectrograms.
"""

import torch

from .mobilenetv2 import MobileNetV2
from .efficient_net import EfficientNet

# Normalization constants for log-magnitude STFT (dB), computed from 200
# peak-normalized Jamendo clips (n_fft=4096, hop=2048, floor=-80 dB).
_SPEC_MEAN = -10.0
_SPEC_STD = 19.4


class SpectralEncoder(torch.nn.Module):
    def __init__(
        self,
        num_params,
        sample_rate,
        encoder_model="mobilenet_v2",
        embed_dim=1024,
        width_mult=1,
        n_fft=4096,
        hop_length=2048,
    ):
        """Encoder operating on spectrograms.

        Args:
            num_params (int): Number of processor parameters to generate.
            sample_rate (float): Audio sample rate.
            encoder_model (str): "mobilenet_v2" or "efficient_net".
            embed_dim (int): Dimensionality of the encoder representations.
            width_mult (int): MobileNetV2 width multiplier.
            n_fft (int): FFT size for STFT.
            hop_length (int): Hop size for STFT.
        """
        super().__init__()
        self.num_params = num_params
        self.sample_rate = sample_rate
        self.encoder_model = encoder_model
        self.embed_dim = embed_dim
        self.width_mult = width_mult
        self.n_fft = n_fft
        self.hop_length = hop_length

        if encoder_model == "mobilenet_v2":
            self.encoder = MobileNetV2(embed_dim=embed_dim, width_mult=width_mult)
        elif encoder_model == "efficient_net":
            self.encoder = EfficientNet.from_name(
                "efficientnet-b2",
                in_channels=1,
                image_size=(128, 65),
                include_top=False,
            )
            self.embedding_projection = torch.nn.Conv2d(
                in_channels=1408,
                out_channels=embed_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            )
        else:
            raise ValueError(f"Invalid encoder_model: {encoder_model}")

        self.window = torch.nn.Parameter(torch.hann_window(n_fft))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform [batch, channels, samples]

        Returns:
            e_norm (Tensor): L2-normalized embedding [batch, embed_dim]
        """
        bs, chs, samp = x.size()

        X = torch.stft(
            x.view(bs, -1),
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        X_db = 20.0 * torch.log10(X.abs() + 1e-8)
        X_db = torch.clamp(X_db, min=-80.0)

        X_db_norm = (X_db - _SPEC_MEAN) / _SPEC_STD
        # [batch, 1, freq, time] -> [batch, 1, time, freq]
        X_db_norm = X_db_norm.unsqueeze(1).permute(0, 1, 3, 2)

        if self.encoder_model == "mobilenet_v2":
            # Repeat to 3 channels for MobileNetV2 (designed for RGB)
            X_db_norm = X_db_norm.repeat(1, 3, 1, 1)

            e = self.encoder(X_db_norm)
            e = torch.nn.functional.adaptive_avg_pool2d(e, 1).reshape(e.shape[0], -1)

        elif self.encoder_model == "efficient_net":
            e = self.encoder(X_db_norm)
            e = self.embedding_projection(e)
            e = torch.squeeze(e, dim=3)
            e = torch.squeeze(e, dim=2)

        # L2 normalize
        norm = torch.norm(e, p=2, dim=-1, keepdim=True)
        return e / norm
