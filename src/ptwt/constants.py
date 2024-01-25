"""Constants and types used throughout the PyTorch Wavelet Toolbox."""

from typing import Literal

__all__ = [
    "BoundaryMode",
]

BoundaryMode = Literal["constant", "zero", "reflect", "periodic", "symmetric"]
"""
This is a type literal for the way of padding.

- Refection padding mirrors samples along the border.
- Zero padding pads zeros.
- Constant padding replicates border values.
- Periodic padding cyclically repeats samples.
- Symmetric padding mirrors samples along the border
"""
