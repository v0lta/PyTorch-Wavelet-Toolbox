"""This module implements stationary wavelet transforms."""

from collections.abc import Sequence
from typing import Optional, Union

import pywt
import torch
import torch.nn.functional as F  # noqa:N812

from ._util import (
    _check_same_device_dtype,
    _get_filter_tensors,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import Wavelet, WaveletCoeff1d


def _circular_pad(x: torch.Tensor, padding_dimensions: Sequence[int]) -> torch.Tensor:
    """Pad a tensor in circular mode, more than once if needed."""
    trailing_dimension = x.shape[-1]

    # if every padding dimension is smaller than or equal the trailing dimension,
    # we do not need to manually wrap
    if not any(
        padding_dimension > trailing_dimension
        for padding_dimension in padding_dimensions
    ):
        return F.pad(x, padding_dimensions, mode="circular")

    # repeat to pad at maximum trailing dimensions until all padding dimensions are zero
    while any(padding_dimension > 0 for padding_dimension in padding_dimensions):
        # reduce every padding dimension to at maximum trailing dimension width
        reduced_padding_dimensions = [
            min(trailing_dimension, padding_dimension)
            for padding_dimension in padding_dimensions
        ]
        # pad using reduced dimensions,
        # which will never throw the circular wrap error
        x = F.pad(x, reduced_padding_dimensions, mode="circular")
        # remove the pad width that was just padded, and repeat
        # if any pad width is greater than zero
        padding_dimensions = [
            max(padding_dimension - trailing_dimension, 0)
            for padding_dimension in padding_dimensions
        ]

    return x


def swt(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    level: Optional[int] = None,
    axis: int = -1,
) -> list[torch.Tensor]:
    """Compute a multilevel 1d stationary wavelet transform.

    This fuctions is equivalent to pywt's :func:`pywt.swt`
    with `trim_approx=True` and `norm=False`.

    Args:
        data (torch.Tensor): The input time series to transform.
            By default the last axis is transformed.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        level (int, optional): The maximum decomposition level.
            If None, the level is computed based on the signal shape.
            Defaults to None.
        axis (int): Compute the transform over this axis of the `data` tensor.
            Defaults to -1.

    Returns:
        Same as :func:`wavedec`. Equivalent to :func:`pywt.swt` with trim_approx=True.
    """
    data, ds = _preprocess_tensor(data, ndim=1, axes=axis)

    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    filt_len = dec_lo.shape[-1]
    filt = torch.stack([dec_lo, dec_hi], 0)

    if level is None:
        level = pywt.swt_max_level(data.shape[-1])

    result_list = []
    res_lo = data
    for current_level in range(level):
        dilation = 2**current_level
        padl, padr = dilation * (filt_len // 2 - 1), dilation * (filt_len // 2)
        res_lo = _circular_pad(res_lo, [padl, padr])
        res = torch.nn.functional.conv1d(res_lo, filt, stride=1, dilation=dilation)
        res_lo, res_hi = torch.split(res, 1, 1)
        # Trim_approx == False
        # result_list.append((res_lo.squeeze(1), res_hi.squeeze(1)))
        result_list.append(res_hi.squeeze(1))
    result_list.append(res_lo.squeeze(1))
    result_list.reverse()

    return _postprocess_coeffs(result_list, ndim=1, ds=ds, axes=axis)


def iswt(
    coeffs: WaveletCoeff1d,
    wavelet: Union[pywt.Wavelet, str],
    axis: int = -1,
) -> torch.Tensor:
    """Invert a 1d stationary wavelet transform.

    Args:
        coeffs: The wavelet coefficient sequence produced by the forward transform
            :func:`swt`. See :data:`ptwt.constants.WaveletCoeff1d`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet, as used in the forward transform.
        axis (int): Compute the transform over this axis of the `data` tensor.
            Defaults to -1.

    Returns:
        A reconstruction of the original :func:`swt` input.
    """
    if not isinstance(coeffs, list):
        coeffs = list(coeffs)
    coeffs, ds = _preprocess_coeffs(coeffs, ndim=1, axes=axis)
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, dtype=torch_dtype, device=torch_device
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = torch.stack([rec_lo, rec_hi], 0)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        dilation = 2 ** (len(coeffs[1:]) - c_pos - 1)
        res_lo = torch.stack([res_lo, res_hi], 1)
        padl, padr = dilation * (filt_len // 2), dilation * (filt_len // 2 - 1)
        # res_lo = torch.nn.functional.pad(res_lo, (padl, padr), mode="circular")
        res_lo_pad = _circular_pad(res_lo, (padl, padr))
        res_lo = torch.mean(
            torch.nn.functional.conv_transpose1d(
                res_lo_pad, rec_filt, dilation=dilation, groups=2, padding=(padl + padr)
            ),
            1,
        )

    res_lo = _postprocess_tensor(res_lo, ndim=1, ds=ds, axes=axis)

    return res_lo
