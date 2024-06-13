"""Constants and types used throughout the PyTorch Wavelet Toolbox."""

from typing import Literal, Union

import torch.Tensor
from typing_extensions import TypeAlias, Unpack

__all__ = [
    "BoundaryMode",
    "ExtendedBoundaryMode",
    "PaddingMode",
    "OrthogonalizeMethod",
    "WaveletDetailTuple2d",
    "WaveletCoeffDetailTuple2d",
    "WaveletCoeffDetailDict",
    "WaveletDetailDict",
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


WaveletDetailTuple2d: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""Detail coefficients of a 2d wavelet transform for a given level.

This is a type alias for a tuple ``(H, V, D)`` of detail coefficient tensors
where ``H`` denotes horizontal, ``V`` vertical and ``D`` diagonal coefficients.

Alias of ``tuple[torch.Tensor, torch.Tensor, torch.Tensor]``
"""


WaveletDetailDict: TypeAlias = dict[str, torch.Tensor]
"""Type alias for a dict containing detail coefficient for a given level.

Thus type alias represents the detail coefficient tensors of a given level for
a wavelet transform in :math:`N` dimensions as the values of a dictionary.
Its keys are a string of length :math:`N` describing the detail coefficient
by the applied filter for each axis where 'a' denotes the low pass
or approximation filter and 'd' the high-pass or detail filter.
For a 3d transform, the dictionary thus uses the keys::

("aad", "ada", "add", "daa", "dad", "dda", "ddd")

Alias of ``dict[str, torch.Tensor]``
"""


WaveletCoeffDetailTuple2d: TypeAlias = tuple[
    torch.Tensor, Unpack[tuple[WaveletDetailTuple2d, ...]]
]
"""Type alias for 2d wavelet transform results.

This type alias represents the result of a 2d wavelet transform
with :math:`L` levels as a tuple ``(A, T1, T2, ...)`` of length :math:`L + 1`
where ``A`` denotes a tensor of approximation coefficients and
``Tl`` is a tuple of detail coefficients for level ``l``,
see :data:`ptwt.constants.WaveletDetailTuple2d`.

Note that this type always contains an approximation coefficient tensor but does not
necesseraily contain any detail coefficients.

Alias of ``tuple[torch.Tensor, *tuple[WaveletDetailTuple2d, ...]]``
"""

WaveletCoeffDetailDict: TypeAlias = tuple[
    torch.Tensor, Unpack[tuple[WaveletDetailDict, ...]]
]
"""Type alias for wavelet transform results in any dimension.

This type alias represents the result of a Nd wavelet transform
with :math:`L` levels as a tuple ``(A, D1, D2, ...)`` of length :math:`L + 1`
where ``A`` denotes a tensor of approximation coefficients and
``Dl`` is a dictionary of detail coefficients for level ``l``,
see :data:`ptwt.constants.WaveletDetailDict`.

Note that this type always contains an approximation coefficient tensor but does not
necesseraily contain any detail coefficients.

Alias of ``tuple[torch.Tensor, *tuple[WaveletDetailDict, ...]]``
"""
