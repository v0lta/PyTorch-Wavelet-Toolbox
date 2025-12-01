# Originally created by moritz (wolter@cs.uni-bonn.de)
# at https://github.com/v0lta/Wavelet-network-compression/blob/master/wavelet_learning/wavelet_linear.py

from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, TypeAlias

import numpy as np
import pywt
import torch
from torch.nn.parameter import Parameter
import torch.nn
from ptwt import wavedec, waverec, waverec2, waverec3
from ptwt.constants import WaveletCoeff1d, WaveletCoeff2d, WaveletCoeffNd
from ptwt.wavelets_learnable import WaveletFilter

X = TypeVar("X")
Y = TypeVar("Y")


class WaveletLayer(torch.nn.Module):
    """
    Create a learnable Wavelet layer as described here:
    https://arxiv.org/pdf/2004.09569.pdf
    The weights are parametrized by S*W*G*P*W*B
    With S,G,B diagonal matrices, P a random permutation and W a
    learnable-wavelet transform.
    """

    def __init__(
        self,
        depth: int,
        init_wavelet: WaveletFilter,
        scales: int,
        p_drop: float = 0.5,
        mode: str = "zero",
    ) -> None:
        super().__init__()

        coefficient_lengths = _get_coefficient_lengths(
            depth=depth, scales=scales, wavelet=init_wavelet, mode=mode
        )
        wave_depth = np.sum(coefficient_lengths)

        mul_b = MMDropoutDiagonal(p_drop, depth)
        wavelet_decomposition = WaveletDecomposition1D(scales, coefficient_lengths, init_wavelet)
        mul_p = Permutator(wave_depth)
        mul_g = MMDropoutDiagonal(p_drop, wave_depth)
        wavelet_reconstruction = WaveletReconstruction1D(scales, coefficient_lengths, init_wavelet)
        mul_s = MMDropoutDiagonal(p_drop, depth)

        self.sequence = torch.nn.Sequential(
            mul_b,
            wavelet_decomposition,
            mul_p,
            mul_g,
            wavelet_reconstruction,
            mul_s,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequence(x)


def _get_coefficient_lengths(
    depth: int, scales: int, wavelet: WaveletFilter, mode: str
) -> list[int]:
    coefficient_len_lst = [depth]
    for _ in range(scales):
        coefficient_len_lst.append(
            pywt.dwt_coeff_len(
                coefficient_len_lst[-1],
                wavelet.filter_bank.dec_lo.shape[-1],
                mode,
            )
        )
    coefficient_len_lst = coefficient_len_lst[1:]
    coefficient_len_lst.append(coefficient_len_lst[-1])
    return coefficient_len_lst


class Permutator(torch.nn.Module):
    def __init__(self, wave_depth: int) -> None:
        super().__init__()
        perm = np.random.permutation(np.eye(wave_depth, dtype=np.float32))
        self.perm = Parameter(torch.from_numpy(perm), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x, self.perm)


class MMDropoutDiagonal(torch.nn.Module):
    """A module that diagonalizes a vector parameter, applies dropout, then multiples it by the input."""

    def __init__(self, dropout: float, depth: int) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.ones = Parameter(torch.from_numpy(np.ones(depth, np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x, self.dropout(torch.diag(self.ones)))


class WaveletDecomposition1D(torch.nn.Module):
    def __init__(
        self, scales: int, coefficient_lengths: list[int], wavelet: WaveletFilter
    ) -> None:
        super().__init__()
        self.scales = scales
        self.wavelet = wavelet
        self.coefficient_len_lst = coefficient_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a 1d-analysis transform.
        Args:
            x (torch.tensor): 2d input tensor
        Returns:
            [torch.tensor]: 2d output tensor.
        """
        # c_lst = self.wavelet.analysis(x.unsqueeze(0).unsqueeze(0))
        c_lst = wavedec(x.unsqueeze(1), self.wavelet, level=self.scales)
        shape_lst = [c_el.shape[-1] for c_el in c_lst]
        c_tensor = torch.cat([c for c in c_lst], -1)
        assert (
            shape_lst == self.coefficient_len_lst[::-1]
        ), "Wavelet shape assumptions false. This is a bug."
        return c_tensor.squeeze(1)


Reconstruction: TypeAlias = Callable[[X, WaveletFilter, Y], torch.Tensor]


class WaveletReconstruction(torch.nn.Module, Generic[X, Y], ABC):
    """Am abstract wavelet reconstruction module."""

    def __init__(
        self,
        scales: int,
        coefficient_lengths: list[int],
        wavelet: WaveletFilter,
        func: Reconstruction[X, Y],
        axis: Y,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.wavelet = wavelet
        self.coefficient_lengths = coefficient_lengths
        self.axis = axis
        self.func = func

    @abstractmethod
    def get_coefficients(self, x: torch.Tensor) -> X:
        """Get coefficients."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction from a tensor input."""
        coefficients = self.get_coefficients(x)
        y = self.func(coefficients, self.wavelet, self.axis)
        return y


class WaveletReconstruction1d(WaveletReconstruction[WaveletCoeff1d, int]):
    """A module for 1D wavelet construction."""

    def __init__(
        self,
        scales: int,
        coefficient_lengths: list[int],
        wavelet: WaveletFilter,
        axis: int | None = None,
    ) -> None:
        super().__init__(
            scales=scales,
            wavelet=wavelet,
            coefficient_lengths=coefficient_lengths,
            axis=-1 if axis is None else None,
            func=waverec,
        )

    def get_coefficients(self, x: torch.Tensor) -> WaveletCoeff1d:
        """Get coefficients for 1D reconstruction."""
        coefficients = []
        start = 0
        # turn tensor into list
        for s in range(self.scales + 1):
            stop = start + self.coefficient_lengths[::-1][s]
            coefficients.append(x[..., start:stop])
            start = self.coefficient_lengths[s]
        return coefficients


class WaveletReconstruction2d(WaveletReconstruction[WaveletCoeff2d, tuple[int, int]]):
    """A module for 2D wavelet construction."""

    def __init__(
        self,
        scales: int,
        coefficient_lengths: list[int],
        wavelet: WaveletFilter,
        axis: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            scales=scales,
            wavelet=wavelet,
            coefficient_lengths=coefficient_lengths,
            axis=(-2, -1) if axis is None else None,
            func=waverec2,
        )

    def get_coefficients(self, x: torch.Tensor) -> WaveletCoeff2d:
        """Get coefficients for 2D reconstruction."""
        raise NotImplementedError


class WaveletReconstruction3d(
    WaveletReconstruction[WaveletCoeffNd, tuple[int, int, int]]
):
    """A module for 3D wavelet construction."""

    def __init__(
        self,
        scales: int,
        coefficient_lengths: list[int],
        wavelet: WaveletFilter,
        axis: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            scales=scales,
            wavelet=wavelet,
            coefficient_lengths=coefficient_lengths,
            axis=(-3, -2, -1) if axis is None else None,
            func=waverec3,
        )

    def get_coefficients(self, x: torch.Tensor) -> WaveletCoeffNd:
        """Get coefficients for 3D reconstruction."""
        raise NotImplementedError
