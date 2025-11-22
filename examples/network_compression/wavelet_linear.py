# Originally created by moritz (wolter@cs.uni-bonn.de)
# at https://github.com/v0lta/Wavelet-network-compression/blob/master/wavelet_learning/wavelet_linear.py

import numpy as np
import pywt
import torch
from torch.nn.parameter import Parameter
import torch.nn
from ptwt import wavedec, waverec
from ptwt.wavelets_learnable import WaveletFilter


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
        wavelet: WaveletFilter,
        scales: int,
        dropout: float = 0.5,
        mode: str = "zero"
    ) -> None:
        super().__init__()

        coefficient_lengths = _get_coefficient_lengths(depth=depth, scales=scales, wavelet=wavelet, mode=mode)
        wave_depth = np.sum(coefficient_lengths)

        mul_b = MMDropoutDiagonal(dropout, depth)
        wavelet_analyzer = WaveletAnalyzer(scales, coefficient_lengths, wavelet)
        mul_p = Permutator(wave_depth)
        mul_g = MMDropoutDiagonal(dropout, wave_depth)
        wavelet_reconstructor = WaveletReconstructor(scales, coefficient_lengths)
        mul_s = MMDropoutDiagonal(dropout, depth)

        self.sequence = torch.nn.Sequential(
            mul_b,
            wavelet_analyzer,
            mul_p,
            mul_g,
            wavelet_reconstructor,
            mul_s,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequence(x)


def _get_coefficient_lengths(depth: int, scales: int, wavelet: WaveletFilter, mode: str) -> list[int]:
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
        self.vec = Parameter(torch.from_numpy(np.ones(depth, np.float32)))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x, self.dropout(torch.diag(self.vec)))


class WaveletAnalyzer(torch.nn.Module):
    def __init__(self, scales: int, coefficient_lengths: list[int], wavelet: WaveletFilter) -> None:
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
        assert shape_lst == self.coefficient_len_lst[::-1], (
            "Wavelet shape assumptions false. This is a bug."
        )
        return c_tensor


class WaveletReconstructor(torch.nn.Module):
    def __init__(self, scales: int, coefficient_lengths: list[int]) -> None:
        super().__init__()
        self.scales = scales
        self.coefficient_lengths = coefficient_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction from a tensor input.
        Args:
            x (torch.Tensor): Analysis coefficient tensor.
        Returns:
            torch.Tensor: Input reconstruction.
        """
        coeff_lst = []
        start = 0
        # turn tensor into list
        for s in range(self.scales + 1):
            stop = start + self.coefficient_lengths[::-1][s]
            coeff_lst.append(x[..., start:stop])
            start = self.coefficient_lengths[s]
        y = waverec(coeff_lst, self.wavelet)
        return y
