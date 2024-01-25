"""Constants and types used throughout the PyTorch Wavelet Toolbox."""

from typing import Literal, Union

__all__ = [
    "BoundaryMode",
    "ExtendedBoundaryMode",
    "Conv2DMode",
    "OrthogonalizeMethod",
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

ExtendedBoundaryMode = Union[Literal["boundary"], BoundaryMode]

Conv2DMode = Literal["full", "valid", "same", "sameshift"]

OrthogonalizeMethod = Literal["qr", "gramschmidt"]
