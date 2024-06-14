"""Compute separable convolution-based transforms.

This module takes multi-dimensional convolutions apart.
It uses single-dimensional convolutions to transform
axes individually.
Under the hood, code in this module transforms all dimensions
using torch.nn.functional.conv1d and it's transpose.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Union

import numpy as np
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_axes_argument,
    _check_if_tensor,
    _fold_axes,
    _is_dtype_supported,
    _map_result,
    _swap_axes,
    _undo_swap_axes,
    _unfold_axes,
)
from .constants import BoundaryMode, WaveletCoeff2dSeparable, WaveletCoeffNd
from .conv_transform import wavedec, waverec
from .conv_transform_2 import _preprocess_tensor_dec2d


def _separable_conv_dwtn_(
    rec_dict: dict[str, torch.Tensor],
    input_arg: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    key: str = "",
) -> None:
    """Compute a single-level separable fast wavelet transform.

    All but the first axes are transformed.

    Args:
        input_arg (torch.Tensor): Tensor of shape ``[batch, data_1, ... data_n]``.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode : The padding mode. The following methods are supported::

                "reflect", "zero", "constant", "periodic".

            Defaults to "reflect".
        key (str): The filter application path. Defaults to "".
        dict (dict[str, torch.Tensor]): The result will be stored here
            in place. Defaults to {}.
    """
    axis_total = len(input_arg.shape) - 1
    if len(key) == axis_total:
        rec_dict[key] = input_arg
    if len(key) < axis_total:
        current_axis = len(key) + 1
        res_a, res_d = wavedec(
            input_arg, wavelet, level=1, mode=mode, axis=-current_axis
        )
        _separable_conv_dwtn_(rec_dict, res_a, wavelet, mode=mode, key="a" + key)
        _separable_conv_dwtn_(rec_dict, res_d, wavelet, mode=mode, key="d" + key)


def _separable_conv_idwtn(
    in_dict: dict[str, torch.Tensor], wavelet: Union[Wavelet, str]
) -> torch.Tensor:
    """Separable single level inverse fast wavelet transform.

    Args:
        in_dict (dict[str, torch.Tensor]): The dictionary produced
            by _separable_conv_dwtn_ .
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet, as used by ``_separable_conv_dwtn_``.

    Returns:
        A reconstruction of the original signal.
    """
    done_dict = {}
    a_initial_keys = list(filter(lambda x: x[0] == "a", in_dict.keys()))
    for a_key in a_initial_keys:
        current_axis = len(a_key)
        d_key = "d" + a_key[1:]
        coeff_d = in_dict[d_key]
        d_shape = coeff_d.shape
        # undo any analysis padding.
        coeff_a = in_dict[a_key][tuple(slice(0, ds) for ds in d_shape)]
        trans_a, trans_d = (
            coeff.transpose(-1, -current_axis) for coeff in (coeff_a, coeff_d)
        )
        flat_a, flat_d = (
            coeff.reshape(-1, coeff.shape[-1]) for coeff in (trans_a, trans_d)
        )
        rec_ad = waverec([flat_a, flat_d], wavelet)
        rec_ad = rec_ad.reshape(list(trans_a.shape[:-1]) + [rec_ad.shape[-1]])
        rec_ad = rec_ad.transpose(-current_axis, -1)
        if a_key[1:]:
            done_dict[a_key[1:]] = rec_ad
        else:
            return rec_ad
    return _separable_conv_idwtn(done_dict, wavelet)


def _separable_conv_wavedecn(
    input: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
) -> WaveletCoeffNd:
    """Compute a multilevel separable padded wavelet analysis transform.

    Args:
        input (torch.Tensor): A tensor i.e. of shape ``[batch,axis_1, ... axis_n]``.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode : The desired padding mode.
        level (int): The desired decomposition level.

    Returns:
        A tuple with the approximation coefficients and
        for each scale a dictionary containing the detail coefficients.
        The dictionaries use a string of length n as keys with
        'a' denoting the low pass or approximation filter and
        'd' the high-pass or detail filter.
    """
    result: list[dict[str, torch.Tensor]] = []
    approx = input

    if level is None:
        wlen = len(_as_wavelet(wavelet))
        level = int(
            min([np.log2(axis_len / (wlen - 1)) for axis_len in input.shape[1:]])
        )

    for _ in range(level):
        level_dict: dict[str, torch.Tensor] = {}
        _separable_conv_dwtn_(level_dict, approx, wavelet, mode=mode, key="")
        approx_key = "a" * (len(input.shape) - 1)
        approx = level_dict.pop(approx_key)
        result.append(level_dict)
    result.reverse()
    return approx, *result


def _separable_conv_waverecn(
    coeffs: WaveletCoeffNd,
    wavelet: Union[Wavelet, str],
) -> torch.Tensor:
    """Separable n-dimensional wavelet synthesis transform.

    Args:
        coeffs (WaveletCoeffNd):
            The output as produced by `_separable_conv_wavedecn`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet, as used by ``_separable_conv_wavedecn``.

    Returns:
        The reconstruction of the original signal.

    Raises:
        ValueError: If the coeffs is not structured as expected.
    """
    if not isinstance(coeffs[0], torch.Tensor):
        raise ValueError("approximation tensor must be first in coefficient list.")
    if not all(map(lambda x: isinstance(x, dict), coeffs[1:])):
        raise ValueError("All entries after approximation tensor must be dicts.")

    approx: torch.Tensor = coeffs[0]
    for level_dict in coeffs[1:]:
        keys = list(level_dict.keys())
        level_dict["a" * max(map(len, keys))] = approx
        approx = _separable_conv_idwtn(level_dict, wavelet)
    return approx


def fswavedec2(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axes: tuple[int, int] = (-2, -1),
) -> WaveletCoeff2dSeparable:
    """Compute a fully separable 2D-padded analysis wavelet transform.

    Args:
        data (torch.Tensor): An data signal of shape ``[batch, height, width]``
            or ``[batch, channels, height, width]``.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet. Refer to the output of
            ``pywt.wavelist(kind="discrete")`` for a list of possible choices.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.
        level (int): The number of desired scales.
            Defaults to None.
        axes ([int, int]): The axes we want to transform,
            defaults to (-2, -1).

    Returns:
        A tuple with the ll coefficients and for each scale a dictionary
        containing the detail coefficients,
        see :data:`ptwt.constants.WaveletCoeff2dSeparable`.
        The dictionaries use the filter order strings::

        ("ad", "da", "dd")

        as keys. 'a' denotes the low pass or approximation filter and
        'd' the high-pass or detail filter.

    Raises:
        ValueError: If the data is not a batched 2D signal.

    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10)
        >>> coeff = ptwt.fswavedec2(data, "haar", level=2)
    """
    if not _is_dtype_supported(data.dtype):
        raise ValueError(f"Input dtype {data.dtype} not supported")

    if tuple(axes) != (-2, -1):
        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            data = _swap_axes(data, list(axes))

    wavelet = _as_wavelet(wavelet)
    data, ds = _preprocess_tensor_dec2d(data)
    data = data.squeeze(1)
    res = _separable_conv_wavedecn(data, wavelet, mode=mode, level=level)

    if ds:
        _unfold_axes2 = partial(_unfold_axes, ds=ds, keep_no=2)
        res = _map_result(res, _unfold_axes2)

    if axes != (-2, -1):
        undo_swap_fn = partial(_undo_swap_axes, axes=axes)
        res = _map_result(res, undo_swap_fn)

    return res


def fswavedec3(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
) -> WaveletCoeffNd:
    """Compute a fully separable 3D-padded analysis wavelet transform.

    Args:
        data (torch.Tensor): An input signal of shape ``[batch, depth, height, width]``.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet. Refer to the output of
            ``pywt.wavelist(kind="discrete")`` for possible choices.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.
        level (int): The number of desired scales.
            Defaults to None.
        axes (tuple[int, int, int]): Compute the transform over these axes
            instead of the last three. Defaults to (-3, -2, -1).

    Returns:
        A tuple with the lll coefficients and for each scale a dictionary
        containing the detail coefficients,
        see :data:`ptwt.constants.WaveletCoeffNd`.
        The dictionaries use the filter order strings::

        ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

        as keys. 'a' denotes the low pass or approximation filter and
        'd' the high-pass or detail filter.

    Raises:
        ValueError: If the input is not a batched 3D signal.


    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10, 10)
        >>> coeff = ptwt.fswavedec3(data, "haar", level=2)
    """
    if not _is_dtype_supported(data.dtype):
        raise ValueError(f"Input dtype {data.dtype} not supported")

    if tuple(axes) != (-3, -2, -1):
        if len(axes) != 3:
            raise ValueError("2D transforms work with two axes.")
        else:
            data = _swap_axes(data, list(axes))

    wavelet = _as_wavelet(wavelet)
    ds = None
    if len(data.shape) >= 5:
        data, ds = _fold_axes(data, 3)
    elif len(data.shape) < 4:
        raise ValueError("At lest four input dimensions are required.")
    data = data.squeeze(1)
    res = _separable_conv_wavedecn(data, wavelet, mode=mode, level=level)

    if ds:
        _unfold_axes3 = partial(_unfold_axes, ds=ds, keep_no=3)
        res = _map_result(res, _unfold_axes3)

    if axes != (-3, -2, -1):
        undo_swap_fn = partial(_undo_swap_axes, axes=axes)
        res = _map_result(res, undo_swap_fn)

    return res


def fswaverec2(
    coeffs: WaveletCoeff2dSeparable,
    wavelet: Union[Wavelet, str],
    axes: tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    """Compute a fully separable 2D-padded synthesis wavelet transform.

    The function uses separate single-dimensional convolutions under
    the hood.

    Args:
        coeffs (WaveletCoeff2dSeparable):
            The wavelet coefficients as computed by `fswavedec2`,
            see :data:`ptwt.constants.WaveletCoeff2dSeparable`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        axes (tuple[int, int]): Compute the transform over these
            axes instead of the last two. Defaults to (-2, -1).

    Returns:
        A reconstruction of the signal encoded in the wavelet coefficients.

    Raises:
        ValueError: If the axes argument is not a tuple of two integers.

    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10)
        >>> coeff = ptwt.fswavedec2(data, "haar", level=2)
        >>> rec = ptwt.fswaverec2(coeff, "haar")
    """
    if tuple(axes) != (-2, -1):
        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            _check_axes_argument(list(axes))
            swap_fn = partial(_swap_axes, axes=list(axes))
            coeffs = _map_result(coeffs, swap_fn)

    ds = None
    wavelet = _as_wavelet(wavelet)

    res_ll = _check_if_tensor(coeffs[0])
    torch_dtype = res_ll.dtype

    if res_ll.dim() >= 4:
        # avoid the channel sum, fold the channels into batches.
        ds = _check_if_tensor(coeffs[0]).shape
        coeffs = _map_result(coeffs, lambda t: _fold_axes(t, 2)[0])
        res_ll = _check_if_tensor(coeffs[0])

    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    res_ll = _separable_conv_waverecn(coeffs, wavelet)

    if ds:
        res_ll = _unfold_axes(res_ll, list(ds), 2)

    if axes != (-2, -1):
        res_ll = _undo_swap_axes(res_ll, list(axes))

    return res_ll


def fswaverec3(
    coeffs: WaveletCoeffNd,
    wavelet: Union[Wavelet, str],
    axes: tuple[int, int, int] = (-3, -2, -1),
) -> torch.Tensor:
    """Compute a fully separable 3D-padded synthesis wavelet transform.

    Args:
        coeffs (WaveletCoeffNd):
            The wavelet coefficients as computed by `fswavedec3`,
            see :data:`ptwt.constants.WaveletCoeffNd`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        axes (tuple[int, int, int]): Compute the transform over these axes
            instead of the last three. Defaults to (-3, -2, -1).

    Returns:
        A reconstruction of the signal encoded in the wavelet coefficients.

    Raises:
        ValueError: If the axes argument is not a tuple with
            three ints.

    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10, 10)
        >>> coeff = ptwt.fswavedec3(data, "haar", level=2)
        >>> rec = ptwt.fswaverec3(coeff, "haar")
    """
    if tuple(axes) != (-3, -2, -1):
        if len(axes) != 3:
            raise ValueError("2D transforms work with two axes.")
        else:
            _check_axes_argument(list(axes))
            swap_fn = partial(_swap_axes, axes=list(axes))
            coeffs = _map_result(coeffs, swap_fn)

    ds = None
    wavelet = _as_wavelet(wavelet)
    res_ll = _check_if_tensor(coeffs[0])
    torch_dtype = res_ll.dtype

    if res_ll.dim() >= 5:
        # avoid the channel sum, fold the channels into batches.
        ds = _check_if_tensor(coeffs[0]).shape
        coeffs = _map_result(coeffs, lambda t: _fold_axes(t, 3)[0])
        res_ll = _check_if_tensor(coeffs[0])

    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    res_ll = _separable_conv_waverecn(coeffs, wavelet)

    if ds:
        res_ll = _unfold_axes(res_ll, list(ds), 3)

    if axes != (-3, -2, -1):
        res_ll = _undo_swap_axes(res_ll, list(axes))

    return res_ll
