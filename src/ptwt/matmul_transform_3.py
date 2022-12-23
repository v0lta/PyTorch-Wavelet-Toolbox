"""Implement 3D seperable boundary transforms."""
import sys
from collections import namedtuple
from functools import partial
from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch

from ._util import Wavelet, _as_wavelet, _is_boundary_mode_supported

# from .conv_transform import get_filter_tensors
from .matmul_transform import construct_boundary_a  # construct_boundary_s,
from .sparse_math import _batch_dim_mm

pad_tuple = namedtuple("pad", ("depth", "height", "width"))


def _matrix_pad_3(
    depth: int, height: int, width: int
) -> Tuple[int, int, int, Tuple[bool, bool, bool]]:
    pad_depth, pad_height, pad_width = (False, False, False)
    if height % 2 != 0:
        height += 1
        pad_height = True
    if width % 2 != 0:
        width += 1
        pad_width = True
    if depth % 2 != 0:
        depth += 1
        pad_depth = True
    return depth, height, width, pad_tuple(pad_depth, pad_height, pad_width)


class MatrixWavedec3(object):
    """Compute 3d seperable transforms."""

    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        boundary: Optional[str] = "qr",
    ):
        """Create a *seperable* three dimensional fast boundary wavelet transform.

        Args:
            wavelet (Union[Wavelet, str]): The wavelet to use.
            level (Optional[int]): The desired decompositon level.
                Defaults to None.
            boundary (Optional[str]): The matrix orthogonalization method.
                Defaults to "qr".

        Raises:
            NotImplementedError: If the chosen orthogonalization method
                is not implemented.
            ValueError: If the analysis and synthesis filters do not have
                the same length.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.level = level
        self.boundary = boundary
        self.input_signal_shape: Optional[Tuple[int, int, int]] = None
        self.fwt_matrix_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        if not _is_boundary_mode_supported(self.boundary):
            raise NotImplementedError
        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    def _construct_analysis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError
        self.fwt_matrix_list = []
        self.size_list = []
        self.pad_list = []
        self.padded = False

        filt_len = self.wavelet.dec_len
        current_depth, current_height, current_width = self.input_signal_shape
        for curr_level in range(1, self.level + 1):
            if (
                current_height < filt_len
                or current_width < filt_len
                or current_depth < filt_len
            ):
                # we have reached the max decomposition depth.
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f"depth, height, and width ({current_depth}, {current_height},"
                    f"{current_width}) is smaller "
                    f"than the filter length {filt_len}. Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            # the conv matrices require even length inputs.
            current_depth, current_height, current_width, pad_tuple = _matrix_pad_3(
                depth=current_depth, height=current_height, width=current_width
            )
            if any(pad_tuple):
                self.padded = True
            self.pad_list.append(pad_tuple)
            self.size_list.append((current_depth, current_height, current_width))

            matrix_construction_fun = partial(
                construct_boundary_a,
                wavelet=self.wavelet,
                boundary=self.boundary,
                device=device,
                dtype=dtype,
            )
            analysis_matrics = [
                matrix_construction_fun(length=dimension_length)
                for dimension_length in (current_depth, current_height, current_width)
            ]

            self.fwt_matrix_list.append(analysis_matrics)

            current_depth, current_height, current_width = (
                current_depth // 2,
                current_height // 2,
                current_width // 2,
            )
        self.size_list.append((current_depth, current_height, current_width))

    def __call__(
        self, input_signal: torch.Tensor
    ) -> List[Union[torch.Tensor, TypedDict[str, torch.Tensor]]]:
        """Compute a seperable 3d-boundary wavelet transform.

        Args:
            input_signal (torch.Tensor): An input signal of shape
                [batch_size, depth, height, width].

        Raises:
            ValueError: If the input dimensions dont work.

        Returns:
            List[Union[torch.Tensor, TypedDict[str, torch.Tensor]]]:
                A list with the approximation coefficients,
                and a coefficeint dict for each scale.
        """
        if input_signal.dim() == 3:
            # add batch dim to unbatched input
            input_signal = input_signal.unsqueeze(0)
        elif input_signal.dim() != 4:
            raise ValueError(
                f"Invalid input tensor shape {input_signal.size()}. "
                "The input signal is expected to be of the form "
                "[batch_size, depth, height, width]."
            )

        batch_size, depth, height, width = input_signal.shape

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != height
            or self.input_signal_shape[1] != width
            or self.input_signal_shape[2] != depth
        ):
            self.input_signal_shape = depth, height, width
            re_build = True

        if self.level is None:
            self.level = int(np.min([np.log2(depth), np.log2(height), np.log2(width)]))
            re_build = True
        elif self.level <= 0:
            raise ValueError("level must be a positive integer.")

        if not self.fwt_matrix_list or re_build:
            self._construct_analysis_matrices(
                device=input_signal.device, dtype=input_signal.dtype
            )

        split_list = []
        ll = input_signal
        for scale, fwt_mats in enumerate(self.fwt_matrix_list):
            # fwt_depth_matrix, fwt_row_matrix, fwt_col_matrix = fwt_mats
            pad_tuple = self.pad_list[scale]
            current_depth, current_height, current_width = self.size_list[scale]
            if pad_tuple.width:
                ll = torch.nn.functional.pad(ll, [0, 1])
            elif pad_tuple.height:
                ll = torch.nn.functional.pad(ll, [0, 0, 0, 1])
            elif pad_tuple.depth:
                ll = torch.nn.functional.pad(ll, [0, 0, 0, 0, 0, 1])

            for dim, mat in enumerate(fwt_mats[::-1]):
                ll = _batch_dim_mm(mat, ll, dim=(-1) * (dim + 1))

            coeff_a, coeff_d = torch.split(ll, current_depth // 2, dim=-3)
            coeff_aa, coeff_ad = torch.split(coeff_a, current_height // 2, dim=-2)
            coeff_da, coeff_dd = torch.split(coeff_d, current_height // 2, dim=-2)
            ll, coeff_aad = torch.split(coeff_aa, current_width // 2, dim=-1)
            coeff_ada, coeff_add = torch.split(coeff_ad, current_width // 2, dim=-1)
            coeff_daa, coeff_dad = torch.split(coeff_da, current_width // 2, dim=-1)
            coeff_dda, coeff_ddd = torch.split(coeff_dd, current_width // 2, dim=-1)
            split_list.append(
                {
                    "aad": coeff_aad,
                    "ada": coeff_ada,
                    "add": coeff_add,
                    "daa": coeff_daa,
                    "dad": coeff_dad,
                    "dda": coeff_dda,
                    "ddd": coeff_ddd,
                }
            )
        split_list.append(ll)
        return split_list[::-1]
