"""Implement 3D separable boundary transforms."""

from __future__ import annotations

import sys
from functools import partial
from typing import NamedTuple, Optional, Union

import numpy as np
import torch

from ._util import (
    _as_wavelet,
    _check_axes_argument,
    _check_if_tensor,
    _deprecated_alias,
    _fold_axes,
    _is_dtype_supported,
    _is_orthogonalize_method_supported,
    _map_result,
    _swap_axes,
    _undo_swap_axes,
    _unfold_axes,
)
from .constants import BoundaryMode, OrthogonalizeMethod, Wavelet, WaveletCoeffNd
from .conv_transform_3 import _fwt_pad3, _waverec3d_fold_channels_3d_list
from .matmul_transform import (
    BaseMatrixWaveDec,
    construct_boundary_a,
    construct_boundary_s,
)
from .sparse_math import _batch_dim_mm


class _PadTuple(NamedTuple):
    """Replaces _PadTuple = namedtuple("_PadTuple", ("depth", "height", "width"))."""

    depth: bool
    height: bool
    width: bool


def _matrix_pad_3(
    depth: int, height: int, width: int
) -> tuple[int, int, int, _PadTuple]:
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
    return depth, height, width, _PadTuple(pad_depth, pad_height, pad_width)


class MatrixWavedec3(BaseMatrixWaveDec):
    """Compute 3d separable transforms.

    Note:
        On each level of the transform both axis of
        the convolved signal are required to be of even length.
        This transform uses zero padding to transform coefficients
        with an odd length.
        To avoid padding consider transforming signals
        with dimensions divisable by :math:`2^L`
        for a :math:`L`-level transform.
    """

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        *,
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

        Raises:
            NotImplementedError: If the chosen orthogonalization method
                is not implemented.
            ValueError: If the analysis and synthesis filters do not have
                the same length.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.level = level
        self.orthogonalization = orthogonalization
        self.odd_coeff_padding_mode = odd_coeff_padding_mode
        if len(axes) != 3:
            raise ValueError("3D transforms work with three axes.")
        else:
            _check_axes_argument(list(axes))
            self.axes = axes
        self.input_signal_shape: Optional[tuple[int, int, int]] = None
        self.fwt_matrix_list: list[list[torch.Tensor]] = []

        if not _is_orthogonalize_method_supported(self.orthogonalization):
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
                    f"then the filter length {filt_len}. Therefore, the transformation "
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
                orthogonalization=self.orthogonalization,
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
        if self.axes != (-3, -2, -1):
            input_signal = _swap_axes(input_signal, list(self.axes))

        ds = None
        if input_signal.dim() < 3:
            raise ValueError("At least three dimensions are required for 3d wavedec.")
        elif len(input_signal.shape) == 3:
            input_signal = input_signal.unsqueeze(1)
        else:
            input_signal, ds = _fold_axes(input_signal, 3)

        _, depth, height, width = input_signal.shape

        if not _is_dtype_supported(input_signal.dtype):
            raise ValueError(f"Input dtype {input_signal.dtype} not supported")

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != depth
            or self.input_signal_shape[1] != height
            or self.input_signal_shape[2] != width
        ):
            self.input_signal_shape = depth, height, width
            re_build = True

        if self.level is None:
            wlen = len(self.wavelet)
            self.level = int(
                np.min(
                    [
                        np.log2(depth / (wlen - 1)),
                        np.log2(height / (wlen - 1)),
                        np.log2(width / (wlen - 1)),
                    ]
                )
            )
            re_build = True
        elif self.level <= 0:
            raise ValueError("level must be a positive integer.")

        if not self.fwt_matrix_list or re_build:
            self._construct_analysis_matrices(
                device=input_signal.device, dtype=input_signal.dtype
            )

        split_list: list[dict[str, torch.Tensor]] = []
        lll = input_signal
        for scale, fwt_mats in enumerate(self.fwt_matrix_list):
            pad_tuple = self.pad_list[scale]
            padding_width = (0, 1) if pad_tuple.width else (0, 0)
            padding_height = (0, 1) if pad_tuple.height else (0, 0)
            padding_depth = (0, 1) if pad_tuple.depth else (0, 0)
            lll = _fwt_pad3(
                lll,
                wavelet=self.wavelet,
                mode=self.odd_coeff_padding_mode,
                padding=padding_width + padding_height + padding_depth,
            )

            for dim, mat in enumerate(fwt_mats[::-1]):
                lll = _batch_dim_mm(mat, lll, dim=(-1) * (dim + 1))

            def _split_rec(
                tensor: torch.Tensor,
                key: str,
                depth: int,
                dict: dict[str, torch.Tensor],
            ) -> None:
                if key:
                    dict[key] = tensor
                if len(key) < depth:
                    dim = len(key) + 1
                    ca, cd = torch.split(tensor, tensor.shape[-dim] // 2, dim=-dim)
                    _split_rec(ca, "a" + key, depth, dict)
                    _split_rec(cd, "d" + key, depth, dict)

            coeff_dict: dict[str, torch.Tensor] = {}
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
        result: WaveletCoeffNd = lll, *split_list

        if ds:
            _unfold_axes_fn = partial(_unfold_axes, ds=ds, keep_no=3)
            result = _map_result(result, _unfold_axes_fn)

        if self.axes != (-3, -2, -1):
            undo_swap_fn = partial(_undo_swap_axes, axes=self.axes)
            result = _map_result(result, undo_swap_fn)

        return result


class MatrixWaverec3(object):
    """Reconstruct a signal from 3d-separable-fwt coefficients."""

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        *,
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

        Raises:
            NotImplementedError: If the selected `orthogonalization` mode
                is not supported.
            ValueError: If the wavelet filters have different lengths.
        """
        self.wavelet = _as_wavelet(wavelet)
        if len(axes) != 3:
            raise ValueError("3D transforms work with three axes")
        else:
            _check_axes_argument(list(axes))
            self.axes = axes
        self.orthogonalization = orthogonalization
        self.ifwt_matrix_list: list[list[torch.Tensor]] = []
        self.input_signal_shape: Optional[tuple[int, int, int]] = None
        self.level: Optional[int] = None

        if not _is_orthogonalize_method_supported(self.orthogonalization):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

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
            current_depth, current_height, current_width, pad_tuple = _matrix_pad_3(
                depth=current_depth, height=current_height, width=current_width
            )
            if any(pad_tuple):
                self.padded = True

            matrix_construction_fun = partial(
                construct_boundary_s,
                wavelet=self.wavelet,
                orthogonalization=self.orthogonalization,
                device=device,
                dtype=dtype,
            )
            synthesis_matrices = [
                matrix_construction_fun(length=dimension_length)
                for dimension_length in (current_depth, current_height, current_width)
            ]

            self.ifwt_matrix_list.append(synthesis_matrices)
            current_depth, current_height, current_width = (
                current_depth // 2,
                current_height // 2,
                current_width // 2,
            )

    def _cat_coeff_recursive(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
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
        if self.axes != (-3, -2, -1):
            swap_axes_fn = partial(_swap_axes, axes=list(self.axes))
            coefficients = _map_result(coefficients, swap_axes_fn)

        ds = None
        # the Union[tensor, dict] idea is coming from pywt. We don't change it here.
        res_lll = _check_if_tensor(coefficients[0])
        if res_lll.dim() < 3:
            raise ValueError(
                "Three dimensional transforms require at least three dimensions."
            )
        elif res_lll.dim() >= 5:
            coefficients, ds = _waverec3d_fold_channels_3d_list(coefficients)
            res_lll = _check_if_tensor(coefficients[0])

        level = len(coefficients) - 1
        if type(coefficients[-1]) is dict:
            depth, height, width = tuple(
                c * 2 for c in coefficients[-1]["ddd"].shape[-3:]
            )
        else:
            raise ValueError("Waverec3 expects dicts of tensors.")

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != depth
            or self.input_signal_shape[1] != height
            or self.input_signal_shape[2] != width
        ):
            self.input_signal_shape = depth, height, width
            re_build = True

        if self.level != level:
            self.level = level
            re_build = True

        lll = coefficients[0]
        if not isinstance(lll, torch.Tensor):
            raise ValueError(
                "First element of coeffs must be the approximation coefficient tensor."
            )

        torch_device = lll.device
        torch_dtype = lll.dtype

        if not _is_dtype_supported(torch_dtype):
            if not _is_dtype_supported(torch_dtype):
                raise ValueError(f"Input dtype {torch_dtype} not supported")

        if not self.ifwt_matrix_list or re_build:
            self._construct_synthesis_matrices(
                device=torch_device,
                dtype=torch_dtype,
            )

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
                if torch_device != coeff.device:
                    raise ValueError("coefficients must be on the same device")
                elif torch_dtype != coeff.dtype:
                    raise ValueError("coefficients must have the same dtype")
                elif test_shape != coeff.shape:
                    raise ValueError(
                        "All coefficients on each level must have the same shape"
                    )

            coeff_dict["a" * len(list(coeff_dict.keys())[-1])] = lll
            lll = self._cat_coeff_recursive(coeff_dict)

            for dim, mat in enumerate(self.ifwt_matrix_list[level - 1 - c_pos][::-1]):
                lll = _batch_dim_mm(mat, lll, dim=(-1) * (dim + 1))

        if ds:
            lll = _unfold_axes(lll, ds, 3)

        if self.axes != (-3, -2, -1):
            lll = _undo_swap_axes(lll, list(self.axes))

        return lll
