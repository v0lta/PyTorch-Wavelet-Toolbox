"""Implement 3D seperable boundary transforms."""
import numpy as np

from typing import List, Optional, Tuple, Union
from functools import partial


from ._util import Wavelet, _as_wavelet, _is_boundary_mode_supported
from .conv_transform import get_filter_tensors
from .matmul_transform import (
    cat_sparse_identity_matrix,
    construct_boundary_a,
    construct_boundary_s,
    orthogonalize,
)
from .sparse_math import batch_mm, construct_strided_conv2d_matrix
import torch
import sys


def _matrix_pad_3(depth: int, height: int, width: int
        ) -> Tuple[int, int, int, Tuple[bool, bool, bool]]:
    pad_tuple = (False, False, False)
    if height % 2 != 0:
        height += 1
        pad_tuple = (pad_tuple[0], True, pad_tuple[2])
    if width % 2 != 0:
        width += 1
        pad_tuple = (True, pad_tuple[1], pad_tuple[2])
    if depth % 2 != 0:
        depth += 1
        pad_tuple = (pad_tuple[0], pad_tuple[1], True)
    return depth, height, width, pad_tuple


class MatrixWavedec3(object):
    """Compute 3d seperable transforms."""

    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        boundary: str = "qr",
    ):
        self.wavelet = wavelet
        self.level = level
        self.boundary = boundary

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
            if (current_height < filt_len
                or current_width < filt_len
                or current_depth < filt_len):
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
                current_depth, current_height, current_width
            )
            if any(pad_tuple):
                self.padded = True
            self.pad_list.append(pad_tuple)
            self.size_list.append((current_depth, current_height, current_width))

            matrix_construction_fun = partial(construct_boundary_a, wavelet=self.wavelet,
                boundary=self.boundary, device=device, dtype=dtype)
            analysis_matrics = [matrix_construction_fun(dimension_length) for dimension_length in
                (current_depth, current_height, current_width)]
     
            self.fwt_matrix_list.append(analysis_matrics)

            current_depth, current_height, current_width = \
                current_depth // 2, current_height // 2, current_width // 2
        self.size_list.append((current_depth, current_height, current_width))


    def __call__(
        self, input_signal: torch.Tensor) -> List:
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
            fwt_depth_matrix, fwt_row_matrix, fwt_col_matrix = fwt_mats
            pad = self.pad_list[scale]
            current_depth, current_height, current_width = self.size_list[scale]
            if pad[0] or pad[1] or pad[2]:
                if pad[0] and not pad[1] and not pad[2]:
                    ll = torch.nn.functional.pad(ll, [0, 1])
                elif pad[1] and not pad[0] and not pad[2]:
                    ll = torch.nn.functional.pad(ll, [0, 0, 0, 1])
                elif pad[0] and pad[1] and not pad[2]:
                    ll = torch.nn.functional.pad(ll, [0, 1, 0, 1])
            ll = batch_mm(fwt_col_matrix, ll.transpose(-2, -1)).transpose(-2, -1)
            ll = batch_mm(fwt_row_matrix, ll)
            a_coeffs, d_coeffs = torch.split(ll, current_height // 2, dim=-2)
            ll, lh = torch.split(a_coeffs, current_width // 2, dim=-1)
            hl, hh = torch.split(d_coeffs, current_width // 2, dim=-1)
            split_list.append((lh, hl, hh))
        split_list.append(ll)
        return split_list[::-1]