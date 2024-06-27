"""This module implements stationary wavelet transforms."""

from collections.abc import Sequence
from typing import Optional, Union

import pywt
import torch
import torch.nn.functional as F  # noqa:N812

from ._util import _as_wavelet, _get_filter_tensors, _unfold_axes
from .constants import Wavelet, WaveletCoeff1d
from .conv_transform import (
    _postprocess_result_list_dec1d,
    _preprocess_result_list_rec1d,
    _preprocess_tensor_dec1d,
)


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

    This fuctions is equivalent to pywt's swt with `trim_approx=True` and `norm=False`.

    Args:
        data (torch.Tensor): The input data of shape ``[batch_size, time]``.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        level (int, optional): The number of levels to compute.
        axis (int): The axis to transform along. Defaults to the last axis.

    Returns:
        Same as wavedec. Equivalent to pywt.swt with trim_approx=True.

    Raises:
        ValueError: Is the axis argument is not an integer.
    """
    if axis != -1:
        if isinstance(axis, int):
            data = data.swapaxes(axis, -1)
        else:
            raise ValueError("swt transforms a single axis only.")

    data, ds = _preprocess_tensor_dec1d(data)

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

    result_list = _postprocess_result_list_dec1d(result_list, ds, axis)

    return result_list[::-1]


def iswt(
    coeffs: WaveletCoeff1d,
    wavelet: Union[pywt.Wavelet, str],
    axis: Optional[int] = -1,
) -> torch.Tensor:
    """Invert a 1d stationary wavelet transform.

    Args:
        coeffs: The wavelet coefficient sequence produced by the forward transform
            :func:`swt`. See :data:`ptwt.constants.WaveletCoeff1d`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet, as used in the forward transform.
        axis (int, optional): The axis the forward trasform was computed over.
            Defaults to -1.

    Returns:
        A reconstruction of the original swt input.

    Raises:
        ValueError: If the axis argument is not an integer.
    """
    if axis != -1:
        swap = []
        if isinstance(axis, int):
            for coeff in coeffs:
                swap.append(coeff.swapaxes(axis, -1))
            coeffs = swap
        else:
            raise ValueError("iswt transforms a single axis only.")

    coeffs, ds = _preprocess_result_list_rec1d(coeffs)

    wavelet = _as_wavelet(wavelet)
    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, dtype=coeffs[0].dtype, device=coeffs[0].device
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

    if len(ds) == 1:
        res_lo = res_lo.squeeze(0)
    elif len(ds) > 2:
        res_lo = _unfold_axes(res_lo, ds, 1)

    if axis != -1:
        res_lo = res_lo.swapaxes(axis, -1)

    return res_lo
