"""Implement 3D separable boundary transforms."""

from __future__ import annotations

import sys
from typing import Optional, Union

import numpy as np
import torch

from ._util import (
    Wavelet,
    _check_same_device_dtype,
    _deprecated_alias,
    _matrix_pad,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import (
    BoundaryMode,
    OrthogonalizeMethod,
    WaveletCoeffNd,
    WaveletDetailDict,
)
from .conv_transform_3 import _fwt_pad3
from .matmul_transform import BaseMatrixWaveDec, BaseMatrixWaveRec
from .sparse_math import _batch_dim_mm


class MatrixWavedec3(BaseMatrixWaveDec):
    """Compute 3d separable transforms."""

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        axes: tuple[int, int, int] = (-3, -2, -1),
        orthogonalization: OrthogonalizeMethod = "qr",
        odd_coeff_padding_mode: BoundaryMode = "zero",
    ):
        """Create a *separable* three-dimensional fast boundary wavelet transform.

        Input signals should have the shape [batch_size, depth, height, width],
        this object transforms the last three dimensions.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            level (int, optional): The desired decomposition level.
                Defaults to None.
            orthogonalization: The method used to orthogonalize
                boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
                Defaults to 'qr'.
            odd_coeff_padding_mode: The constructed FWT matrices require inputs
                with even lengths. Thus, any odd-length approximation coefficients
                are padded to an even length using this mode,
                see :data:`ptwt.constants.BoundaryMode`.
                Defaults to 'zero'.

        .. versionchanged:: 1.10
            The argument `boundary` has been renamed to `orthogonalization`.
        """
        super().__init__(
            ndim=3,
            wavelet=wavelet,
            level=level,
            axes=axes,
            separable=True,
            orthogonalization=orthogonalization,
            odd_coeff_padding_mode=odd_coeff_padding_mode,
        )

    def _construct_analysis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError
        self.fwt_matrix_list = []
        self.size_list = []
        self._pad_list = []
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
                    f"then the filter length {filt_len}. Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            # the conv matrices require even length inputs.
            (current_depth, current_height, current_width), pad_tuple = _matrix_pad(
                current_depth, current_height, current_width
            )
            if any(pad_tuple):
                self.padded = True
            self._pad_list.append(pad_tuple)
            self.size_list.append((current_depth, current_height, current_width))

            analysis_matrices = self.construct_separable_analysis_matrices(
                (current_depth, current_height, current_width),
                device=device,
                dtype=dtype,
            )

            self.fwt_matrix_list.append(analysis_matrices)

            current_depth, current_height, current_width = (
                current_depth // 2,
                current_height // 2,
                current_width // 2,
            )
        self.size_list.append((current_depth, current_height, current_width))

    def __call__(self, input_signal: torch.Tensor) -> WaveletCoeffNd:
        """Compute a separable 3d-boundary wavelet transform.

        Args:
            input_signal (torch.Tensor): An input signal. For example
                of shape ``[batch_size, depth, height, width]``.

        Returns:
            The resulting coefficients for each level are stored in a tuple,
            see :data:`ptwt.constants.WaveletCoeffNd`.

        Raises:
            ValueError: If the input dimensions don't work.
        """
        input_signal, ds = _preprocess_tensor(
            input_signal, ndim=3, axes=self.axes, add_channel_dim=False
        )
        input_signal_shape = input_signal.shape[1:]

        re_build = False
        if self.input_signal_shape != input_signal_shape:
            self.input_signal_shape = input_signal_shape
            re_build = True

        if self.level is None:
            wlen = len(self.wavelet)
            max_level_per_axis = map(
                lambda size: np.log2(size / (wlen - 1)), input_signal_shape
            )
            self.level = int(min(max_level_per_axis))
            re_build = True
        elif self.level <= 0:
            raise ValueError("level must be a positive integer.")

        if not self.fwt_matrix_list or re_build:
            self._construct_analysis_matrices(
                device=input_signal.device, dtype=input_signal.dtype
            )

        def _add_padding(signal: torch.Tensor, pad: tuple[bool, ...]) -> torch.Tensor:
            if any(pad):
                axis_padding = [(0, 1) if pad_axis else (0, 0) for pad_axis in pad]
                assert len(axis_padding) == 3
                padding = axis_padding[0] + axis_padding[1] + axis_padding[2]
                signal = _fwt_pad3(
                    signal,
                    wavelet=self.wavelet,
                    mode=self.odd_coeff_padding_mode,
                    padding=padding,
                )
            return signal

        split_list: list[WaveletDetailDict] = []
        lll = input_signal
        for scale, fwt_mats in enumerate(self.fwt_matrix_list):
            lll = _add_padding(lll, self._pad_list[scale])

            for dim, mat in enumerate(fwt_mats[::-1]):
                lll = _batch_dim_mm(mat, lll, dim=(-1) * (dim + 1))

            def _split_rec(
                tensor: torch.Tensor,
                key: str,
                depth: int,
                to_dict: WaveletDetailDict,
            ) -> None:
                if key:
                    to_dict[key] = tensor
                if len(key) < depth:
                    dim = len(key) + 1
                    ca, cd = torch.split(tensor, tensor.shape[-dim] // 2, dim=-dim)
                    _split_rec(ca, "a" + key, depth, to_dict)
                    _split_rec(cd, "d" + key, depth, to_dict)

            coeff_dict: WaveletDetailDict = {}
            _split_rec(lll, "", 3, coeff_dict)
            lll = coeff_dict["aaa"]
            result_keys = list(
                filter(lambda x: len(x) == 3 and not x == "aaa", coeff_dict.keys())
            )
            coeff_dict = {
                key: tensor for key, tensor in coeff_dict.items() if key in result_keys
            }
            split_list.append(coeff_dict)

        split_list.reverse()
        coeffs: WaveletCoeffNd = lll, *split_list

        return _postprocess_coeffs(coeffs, ndim=3, ds=ds, axes=self.axes)


class MatrixWaverec3(BaseMatrixWaveRec):
    """Reconstruct a signal from 3d-separable-fwt coefficients."""

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        axes: tuple[int, int, int] = (-3, -2, -1),
        orthogonalization: OrthogonalizeMethod = "qr",
    ):
        """Compute a three-dimensional separable boundary wavelet synthesis transform.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            axes (tuple[int, int, int]): Transform these axes instead of the
                last three. Defaults to (-3, -2, -1).
            orthogonalization: The method used to orthogonalize
                boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
                Defaults to 'qr'.

        .. versionchanged:: 1.10
            The argument `boundary` has been renamed to `orthogonalization`.
        """
        super().__init__(
            ndim=3,
            wavelet=wavelet,
            axes=axes,
            separable=True,
            orthogonalization=orthogonalization,
        )

    def _construct_synthesis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        self.ifwt_matrix_list = []
        self.padded = False
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError

        current_depth, current_height, current_width = self.input_signal_shape
        filt_len = self.wavelet.rec_len

        for curr_level in range(1, self.level + 1):
            if (
                current_depth < filt_len
                or current_height < filt_len
                or current_width < filt_len
            ):
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f" depth, height and width ({current_depth}, {current_height}, "
                    f"{current_width}) is smaller than the filter length {filt_len}."
                    f" Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            # the conv matrices require even length inputs.
            (current_depth, current_height, current_width), pad_tuple = _matrix_pad(
                current_depth, current_height, current_width
            )
            if any(pad_tuple):
                self.padded = True

            synthesis_matrices = self.construct_separable_synthesis_matrices(
                (current_depth, current_height, current_width),
                device=device,
                dtype=dtype,
            )
            self.ifwt_matrix_list.append(synthesis_matrices)
            current_depth, current_height, current_width = (
                current_depth // 2,
                current_height // 2,
                current_width // 2,
            )

    def _cat_coeff_recursive(self, input_dict: WaveletDetailDict) -> torch.Tensor:
        done_dict = {}
        a_initial_keys = list(filter(lambda x: x[0] == "a", input_dict.keys()))
        for a_key in a_initial_keys:
            d_key = "d" + a_key[1:]
            cat_d = input_dict[d_key]
            d_shape = cat_d.shape
            # undo any analysis padding.
            cat_a = input_dict[a_key][:, : d_shape[1], : d_shape[2], : d_shape[3]]
            cat_tensor = torch.cat([cat_a, cat_d], dim=-len(a_key))
            if a_key[1:]:
                done_dict[a_key[1:]] = cat_tensor
            else:
                return cat_tensor
        return self._cat_coeff_recursive(done_dict)

    def __call__(self, coefficients: WaveletCoeffNd) -> torch.Tensor:
        """Reconstruct a batched 3d-signal from its coefficients.

        Args:
            coefficients (WaveletCoeffNd):
                The output from the `MatrixWavedec3` object,
                see :data:`ptwt.constants.WaveletCoeffNd`.

        Returns:
            torch.Tensor: A reconstruction of the original signal.

        Raises:
            ValueError: If the data structure is inconsistent.
        """
        coefficients, ds = _preprocess_coeffs(coefficients, ndim=3, axes=self.axes)
        torch_device, torch_dtype = _check_same_device_dtype(coefficients)

        level = len(coefficients) - 1

        if not isinstance(coefficients[-1], dict):
            raise ValueError("Waverec3 expects dicts of tensors.")

        input_signal_shape = torch.Size(
            [c * 2 for c in coefficients[-1]["ddd"].shape[-3:]]
        )

        re_build = False
        if self.level != level or self.input_signal_shape != input_signal_shape:
            self.level = level
            self.input_signal_shape = input_signal_shape
            re_build = True

        if not self.ifwt_matrix_list or re_build:
            self._construct_synthesis_matrices(
                device=torch_device,
                dtype=torch_dtype,
            )

        lll = coefficients[0]
        for c_pos, coeff_dict in enumerate(coefficients[1:]):
            if not isinstance(coeff_dict, dict) or len(coeff_dict) != 7:
                raise ValueError(
                    f"Unexpected detail coefficient type: {type(coeff_dict)}. Detail "
                    "coefficients must be a dict containing 7 tensors as returned by "
                    "MatrixWavedec3."
                )
            test_shape = None
            for coeff in coeff_dict.values():
                if test_shape is None:
                    test_shape = coeff.shape
                elif test_shape != coeff.shape:
                    raise ValueError(
                        "All coefficients on each level must have the same shape"
                    )

            coeff_dict["a" * len(list(coeff_dict.keys())[-1])] = lll
            lll = self._cat_coeff_recursive(coeff_dict)

            for dim, mat in enumerate(self.ifwt_matrix_list[level - 1 - c_pos][::-1]):
                lll = _batch_dim_mm(mat, lll, dim=(-1) * (dim + 1))

        return _postprocess_tensor(lll, ndim=3, ds=ds, axes=self.axes)
