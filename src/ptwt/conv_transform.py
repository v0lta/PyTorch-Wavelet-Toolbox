"""Fast wavelet transformations based on torch.nn.functional.conv1d and its transpose.

This module treats boundaries with edge-padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Optional, Union

import pywt
import torch

from ._util import (
    Wavelet,
    _check_same_device_dtype,
    _construct_nd_filt,
    _fwt_padn,
    _get_filter_tensors,
    _get_pad,
    _get_pad_removal_slice,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import BoundaryMode, WaveletCoeff2d


def _flatten_2d_coeff_lst(
    coeff_lst_2d: WaveletCoeff2d,
    flatten_tensors: bool = True,
) -> list[torch.Tensor]:
    """Flattens a sequence of tensor tuples into a single list.

    Args:
        coeff_lst_2d (WaveletCoeff2d): A pywt-style
            coefficient tuple of torch tensors.
        flatten_tensors (bool): If true, 2d tensors are flattened. Defaults to True.

    Returns:
        A single 1-d list with all original elements.
    """

    def _process_tensor(coeff: torch.Tensor) -> torch.Tensor:
        return coeff.flatten() if flatten_tensors else coeff

    flat_coeff_lst = [_process_tensor(coeff_lst_2d[0])]
    for coeff_tuple in coeff_lst_2d[1:]:
        flat_coeff_lst.extend(map(_process_tensor, coeff_tuple))
    return flat_coeff_lst


def _adjust_padding_at_reconstruction(
    res_ll_size: int, coeff_size: int, pad_end: int, pad_start: int
) -> tuple[int, int]:
    pred_size = res_ll_size - (pad_start + pad_end)
    next_size = coeff_size
    if next_size == pred_size:
        pass
    elif next_size == pred_size - 1:
        pad_end += 1
    else:
        raise AssertionError(
            "padding error, please check if dec and rec wavelets are identical."
        )
    return pad_end, pad_start


def wavedec(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axis: int = -1,
) -> list[torch.Tensor]:
    r"""Compute the analysis (forward) 1d fast wavelet transform.

    The transformation relies on convolution operations with filter
    pairs.

    .. math::
        x_s * h_k = c_{k,s+1}

    Where :math:`x_s` denotes the input at scale :math:`s`, with
    :math:`x_0` equal to the original input. :math:`h_k` denotes
    the convolution filter, with :math:`k \in {A, D}`, where :math:`A` for
    approximation and :math:`D` for detail. The processes uses approximation
    coefficients as inputs for higher scales.
    Set the `level` argument to choose the largest scale.

    Args:
        data (torch.Tensor): The input time series,
                             By default the last axis is transformed.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Please consider the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.
        level (int): The scale level to be computed.
                               Defaults to None.
        axis (int): Compute the transform over this axis instead of the
            last one. Defaults to -1.

    Returns:
        A list::

            [cA_s, cD_s, cD_s-1, …, cD2, cD1]

        containing the wavelet coefficients. A denotes
        approximation and D detail coefficients.

    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> # compute the forward fwt coefficients
        >>> ptwt.wavedec(data_torch, pywt.Wavelet('haar'),
        >>>              mode='zero', level=2)
    """
    data, ds = _preprocess_tensor(data, ndim=1, axes=axis)

    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    filt_len = dec_lo.shape[-1]

    dec_filt = _construct_nd_filt(lo=dec_lo, hi=dec_hi, ndim=1)

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_list = []
    res_lo = data
    for _ in range(level):
        res_lo = _fwt_padn(res_lo, wavelet, ndim=1, mode=mode)
        res = torch.nn.functional.conv1d(res_lo, dec_filt, stride=2)
        res_lo, res_hi = torch.split(res, 1, 1)
        result_list.append(res_hi.squeeze(1))
    result_list.append(res_lo.squeeze(1))
    result_list.reverse()

    return _postprocess_coeffs(result_list, ndim=1, ds=ds, axes=axis)


def waverec(
    coeffs: Sequence[torch.Tensor], wavelet: Union[Wavelet, str], axis: int = -1
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    Args:
        coeffs (Sequence): The wavelet coefficient sequence produced by wavedec.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        axis (int): Transform this axis instead of the last one. Defaults to -1.

    Returns:
        The reconstructed signal tensor.

    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> # invert the fast wavelet transform.
        >>> ptwt.waverec(ptwt.wavedec(data_torch, pywt.Wavelet('haar'),
        >>>                           mode='zero', level=2),
        >>>              pywt.Wavelet('haar'))

    """
    # fold channels and swap axis, if necessary.
    if not isinstance(coeffs, list):
        coeffs = list(coeffs)
    coeffs, ds = _preprocess_coeffs(coeffs, ndim=1, axes=axis)
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_nd_filt(lo=rec_lo, hi=rec_hi, ndim=1)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:], start=1):
        res_lo = torch.stack([res_lo, res_hi], 1)
        res_lo = torch.nn.functional.conv_transpose1d(
            res_lo, rec_filt, stride=2
        ).squeeze(1)

        # remove the padding
        if c_pos < len(coeffs) - 1:
            next_detail_shape = coeffs[c_pos + 1].shape
        else:
            next_detail_shape = None

        _slice = partial(
            _get_pad_removal_slice,
            filt_len=filt_len,
            data_shape=res_lo.shape,
            next_detail_shape=next_detail_shape,
        )

        res_lo = res_lo[..., _slice(-1)]

    # undo folding and swapping
    res_lo = _postprocess_tensor(res_lo, ndim=1, ds=ds, axes=axis)

    return res_lo
