"""
Models for tape saturation parameter identification.
"""

from .encoder import SpectralEncoder
from .controller import ParameterController
from .tape_processor import (
    TapeSaturationProcessor,
    apply_tape_saturation,
    JilesAthertonProcessor,
    apply_ja_saturation,
    HardClippingProcessor,
    apply_hard_clipping,
)
from .mobilenetv2 import MobileNetV2

__all__ = [
    "SpectralEncoder",
    "ParameterController",
    "TapeSaturationProcessor",
    "apply_tape_saturation",
    "JilesAthertonProcessor",
    "apply_ja_saturation",
    "HardClippingProcessor",
    "apply_hard_clipping",
    "MobileNetV2",
]
