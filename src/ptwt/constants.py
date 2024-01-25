"""Constants and types used throughout the PyTorch Wavelet Toolbox."""

from typing import Literal, Union

__all__ = [
    "BoundaryMode",
    "ExtendedBoundaryMode",
    "PaddingMode",
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

PaddingMode = Literal["full", "valid", "same", "sameshift"]
"""
The padding mode is used when construction convolution matrices.
"""

OrthogonalizeMethod = Literal["qr", "gramschmidt"]
"""
The method for orthogonalizing a matrix.

1. 'qr' relies on pytorch's dense qr implementation, it is fast but memory hungry.
2. 'gramschmidt' option is sparse, memory efficient, and slow.

Choose 'gramschmidt' if 'qr' runs out of memory.
"""
