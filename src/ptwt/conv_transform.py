"""Fast wavelet transformations based on torch.nn.functional.conv1d and its transpose.

This module treats boundaries with edge-padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import pywt
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_same_device_dtype,
    _get_len,
    _get_pad,
    _pad_symmetric,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
    _translate_boundary_strings,
)
from .constants import BoundaryMode, WaveletCoeff2d


def _create_tensor(
    filter: Sequence[float], flip: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if flip:
        if isinstance(filter, torch.Tensor):
            return filter.flip(-1).unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
    else:
        if isinstance(filter, torch.Tensor):
            return filter.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)


def _get_filter_tensors(
    wavelet: Union[Wavelet, str],
    flip: bool,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        flip (bool): Flip filters left-right, if true.
        device (torch.device or str): PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
               computation. Default: torch.float32.

    Returns:
        A tuple (dec_lo, dec_hi, rec_lo, rec_hi) containing
        the four filter tensors
    """
    wavelet = _as_wavelet(wavelet)
    device = torch.device(device)

    if isinstance(wavelet, tuple):
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet
    else:
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo, flip, device, dtype)
    dec_hi_tensor = _create_tensor(dec_hi, flip, device, dtype)
    rec_lo_tensor = _create_tensor(rec_lo, flip, device, dtype)
    rec_hi_tensor = _create_tensor(rec_hi, flip, device, dtype)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _fwt_pad(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: Optional[BoundaryMode] = None,
    padding: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Pad the input signal to make the fwt matrix work.

    The padding assumes a future step will transform the last axis.

    Args:
        data (torch.Tensor): Input data ``[batch_size, 1, time]``
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode: The desired padding mode for extending the signal along the edges.
            Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.
        padding (tuple[int, int], optional): A tuple (padl, padr) with the
            number of padded values on the left and right side of the last
            axes of `data`. If None, the padding values are computed based
            on the signal shape and the wavelet length. Defaults to None.

    Returns:
        A PyTorch tensor with the padded input data
    """
    # convert pywt to pytorch convention.
    if mode is None:
        mode = "reflect"
    pytorch_mode = _translate_boundary_strings(mode)

    if padding is None:
        padr, padl = _get_pad(data.shape[-1], _get_len(wavelet))
    else:
        padl, padr = padding
    if pytorch_mode == "symmetric":
        data_pad = _pad_symmetric(data, [(padl, padr)])
    else:
        data_pad = torch.nn.functional.pad(data, [padl, padr], mode=pytorch_mode)
    return data_pad


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
    filt = torch.stack([dec_lo, dec_hi], 0)

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_list = []
    res_lo = data
    for _ in range(level):
        res_lo = _fwt_pad(res_lo, wavelet, mode=mode)
        res = torch.nn.functional.conv1d(res_lo, filt, stride=2)
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
    filt = torch.stack([rec_lo, rec_hi], 0)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        res_lo = torch.stack([res_lo, res_hi], 1)
        res_lo = torch.nn.functional.conv_transpose1d(res_lo, filt, stride=2).squeeze(1)

        # remove the padding
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        if c_pos < len(coeffs) - 2:
            padr, padl = _adjust_padding_at_reconstruction(
                res_lo.shape[-1], coeffs[c_pos + 2].shape[-1], padr, padl
            )
        if padl > 0:
            res_lo = res_lo[..., padl:]
        if padr > 0:
            res_lo = res_lo[..., :-padr]

    # undo folding and swapping
    res_lo = _postprocess_tensor(res_lo, ndim=1, ds=ds, axes=axis)

    return res_lo
