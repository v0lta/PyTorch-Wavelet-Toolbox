"""Constants and types used throughout the PyTorch Wavelet Toolbox."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple, Protocol, TypeAlias

import torch

__all__ = [
    "BoundaryMode",
    "ExtendedBoundaryMode",
    "OrthogonalizeMethod",
    "PacketNodeOrder",
    "PaddingMode",
    "Wavelet",
    "WaveletCoeff1d",
    "WaveletCoeff2d",
    "WaveletCoeff2dSeparable",
    "WaveletCoeffNd",
    "WaveletDetailDict",
    "WaveletDetailTuple2d",
    "WaveletTensorTuple",
]

SUPPORTED_DTYPES = {torch.float32, torch.float64}


class Wavelet(Protocol):
    """Wavelet object interface, based on the pywt wavelet object."""

    name: str
    dec_lo: Sequence[float]
    dec_hi: Sequence[float]
    rec_lo: Sequence[float]
    rec_hi: Sequence[float]
    dec_len: int
    rec_len: int
    filter_bank: tuple[
        Sequence[float], Sequence[float], Sequence[float], Sequence[float]
    ]

    def __len__(self) -> int:
        """Return the number of filter coefficients."""
        return len(self.dec_lo)


class WaveletTensorTuple(NamedTuple):
    """Named tuple containing the wavelet filter bank to use in JIT code."""

    dec_lo: torch.Tensor
    dec_hi: torch.Tensor
    rec_lo: torch.Tensor
    rec_hi: torch.Tensor

    @property
    def dec_len(self) -> int:
        """Length of decomposition filters."""
        return len(self.dec_lo)

    @property
    def rec_len(self) -> int:
        """Length of reconstruction filters."""
        return len(self.rec_lo)

    @property
    def filter_bank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter bank of the wavelet."""
        return self

    @classmethod
    def from_wavelet(cls, wavelet: Wavelet, dtype: torch.dtype) -> WaveletTensorTuple:
        """Construct Wavelet named tuple from wavelet protocol member."""
        return cls(
            torch.tensor(wavelet.dec_lo, dtype=dtype),
            torch.tensor(wavelet.dec_hi, dtype=dtype),
            torch.tensor(wavelet.rec_lo, dtype=dtype),
            torch.tensor(wavelet.rec_hi, dtype=dtype),
        )


BoundaryMode = Literal["constant", "zero", "reflect", "periodic", "symmetric"]
r"""This is a type literal for the way of padding used at boundaries.

- ``reflect``: Reflection padding reflects samples at the border:

  .. math::

      \dots, x_3,  x_2 \mid x_1, x_2 \dots, x_{n-1}, x_n \mid x_{n-1}, x_{n-2}, \dots

- ``zero``: Zero padding extends the signal with zeros:

  .. math::

      \dots, 0, 0 \mid x_1, x_2 \dots, x_{n-1}, x_n \mid 0, 0, \dots

- ``constant``: Constant padding replicates border values:

  .. math::

      \dots, x_1, x_1 \mid x_1, x_2, \dots, x_{n-1}, x_n \mid x_n, x_n, \dots

- ``periodic``: Periodic padding cyclically repeats samples:

  .. math::

      \dots, x_{n-1}, x_n \mid x_1, x_2, \dots, x_{n-1}, x_n \mid x_1, x_2, \dots

- ``symmetric``: Symmetric padding mirrors samples along the border:

  .. math::

      \dots, x_2, x_1 \mid x_1, x_2, \dots, x_{n-1}, x_n \mid x_n, x_{n-1}, \dots
"""

ExtendedBoundaryMode: TypeAlias = Literal["boundary"] | BoundaryMode
"""This is a type literal for the way of handling signal boundaries.

This is either a form of padding (see :data:`ptwt.constants.BoundaryMode` for padding
options) or ``boundary`` to use boundary wavelets.
"""


# TODO: Add documentation on the different values of PaddingMode

PaddingMode = Literal["full", "valid", "same", "sameshift"]
"""The padding mode is used when construction convolution matrices."""

OrthogonalizeMethod = Literal["qr", "gramschmidt"]
"""The method for orthogonalizing a matrix.

1. ``qr`` relies on PyTorch's dense QR implementation, it is fast but memory hungry.
2. ``gramschmidt`` option is sparse, memory efficient, and slow.

Choose ``gramschmidt`` if ``qr`` runs out of memory.
"""

PacketNodeOrder = Literal["freq", "natural"]
"""This is a type literal for the order of wavelet packet tree nodes.

- frequency order (``freq``)
- natural order (``natural``)
"""


# Note: This data structure was chosen to follow pywt's conventions
WaveletCoeff1d: TypeAlias = Sequence[torch.Tensor]
r"""Type alias for 1d wavelet transform results.

This type alias represents the result of a 1d wavelet transform with :math:`n` levels as
a sequence:

.. math::

    [cA_n, cD_n, cD_{n-1}, \dots, cD_1]

of :math:`n + 1` tensors. The first entry of the sequence (:math:`cA_n`) is the
approximation coefficient tensor. The following entries (:math:`cD_n, \dots, cD_1`) are
the detail coefficient tensors of the respective level.

Note that this type always contains an approximation coefficient tensor but does not
necessarily contain any detail coefficients.

Alias of ``Sequence[torch.Tensor]``
"""


class WaveletDetailTuple2d(NamedTuple):
    """Detail coefficients of a 2d wavelet transform for a given level.

    This is a type alias for a named tuple ``(H, V, D)`` of detail coefficient tensors
    where ``H`` denotes horizontal, ``V`` vertical and ``D`` diagonal coefficients.

    We follow the pywt convention for the orientation of axes , i.e. axis 0 is
    horizontal and axis 1 vertical. For more information, see the `pywt docs
    <https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html#d-coordinate-conventions>`_.
    """

    horizontal: torch.Tensor
    vertical: torch.Tensor
    diagonal: torch.Tensor


WaveletDetailDict: TypeAlias = dict[str, torch.Tensor]
"""Type alias for a dict containing detail coefficient for a given level.

This type alias represents the detail coefficient tensors of a given level for a wavelet
transform in :math:`N` dimensions as the values of a dictionary. Its keys are a string
of length :math:`N` describing the detail coefficient by the applied filter for each
axis. The string consists only of chars 'a' and 'd' where 'a' denotes the low pass or
approximation filter and 'd' the high-pass or detail filter. For a 3d transform, the
dictionary thus uses the keys:

::

    ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

Alias of ``dict[str, torch.Tensor]``
"""


# Note: This data structure was chosen to follow pywt's conventions
WaveletCoeff2d: TypeAlias = tuple[torch.Tensor, *tuple[WaveletDetailTuple2d, ...]]
r"""Type alias for 2d wavelet transform results.

This type alias represents the result of a 2d wavelet transform with :math:`n` levels as
a tuple:

.. math::

    (cA_n, T_n, \dots, T_1)

of length :math:`n + 1`. :math:`cA_n` denotes a tensor of approximation coefficients for
the `n`-th level of decomposition. :math`T_l` is a tuple of detail coefficients for
level :math`l`, see :class:`ptwt.constants.WaveletDetailTuple2d`.

Note that this type always contains an approximation coefficient tensor but does not
necessarily contain any detail coefficients.

Alias of ``tuple[torch.Tensor, *tuple[WaveletDetailTuple2d, ...]]``
"""

# Note: This data structure was chosen to follow pywt's conventions
WaveletCoeffNd: TypeAlias = tuple[torch.Tensor, *tuple[WaveletDetailDict, ...]]
r"""Type alias for wavelet transform results in any dimension.

This type alias represents the result of a N-dimensional wavelet transform with
:math:`n` levels as a tuple:

.. math::

    (cA_n, D_n, \dots, D_1)

of length :math:`n + 1`. :math:`cA_n` denotes a tensor of approximation coefficients for
the `n`-th level of decomposition. :math:`D_l` is a dictionary of detail coefficients
for level :math:`l`, see :data:`ptwt.constants.WaveletDetailDict`.

Note that this type always contains an approximation coefficient tensor but does not
necessarily contain any detail coefficients.

Alias of ``tuple[torch.Tensor, *tuple[WaveletDetailDict, ...]]``
"""

WaveletCoeff2dSeparable: TypeAlias = WaveletCoeffNd
"""Type alias for separable 2d wavelet transform results.

This is an alias of :data:`ptwt.constants.WaveletCoeffNd`. It is used to emphasize the
use of :data:`ptwt.constants.WaveletDetailDict` for detail coefficients in a 2d setting
-- in contrast to :data:`ptwt.constants.WaveletCoeff2d`.

Alias of :data:`ptwt.constants.WaveletCoeffNd`, i.e. of ``tuple[torch.Tensor,
*tuple[WaveletDetailDict, ...]]``.
"""
