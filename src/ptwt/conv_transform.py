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
    _fold_axes,
    _get_len,
    _is_dtype_supported,
    _pad_symmetric,
    _unfold_axes,
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


def _get_pad(data_len: int, filt_len: int) -> tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The size of the used filter.

    Returns:
        A tuple (padr, padl). The first entry specifies how many numbers
        to attach on the right. The second entry covers the left side.
    """
    # pad to ensure we see all filter positions and
    # for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    padr += data_len % 2

    return padr, padl


def _translate_boundary_strings(pywt_mode: BoundaryMode) -> str:
    """Translate pywt mode strings to PyTorch mode strings.

    We support constant, zero, reflect, and periodic.
    Unfortunately, "constant" has different meanings in the
    Pytorch and PyWavelet communities.

    Raises:
        ValueError: If the padding mode is not supported.
    """
    if pywt_mode == "constant":
        return "replicate"
    elif pywt_mode == "zero":
        return "constant"
    elif pywt_mode == "reflect":
        return pywt_mode
    elif pywt_mode == "periodic":
        return "circular"
    elif pywt_mode == "symmetric":
        # pytorch does not support symmetric mode,
        # we have our own implementation.
        return pywt_mode
    raise ValueError(f"Padding mode not supported: {pywt_mode}")


def _fwt_pad(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: Optional[BoundaryMode] = None,
) -> torch.Tensor:
    """Pad the input signal to make the fwt matrix work.

    The padding assumes a future step will transform the last axis.

    Args:
        data (torch.Tensor): Input data ``[batch_size, 1, time]``
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.

    Returns:
        A PyTorch tensor with the padded input data
    """
    wavelet = _as_wavelet(wavelet)

    # convert pywt to pytorch convention.
    if mode is None:
        mode = "reflect"
    pytorch_mode = _translate_boundary_strings(mode)

    padr, padl = _get_pad(data.shape[-1], _get_len(wavelet))
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


def _preprocess_tensor_dec1d(
    data: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Preprocess input tensor dimensions.

    Args:
        data (torch.Tensor): An input tensor of any shape.

    Returns:
        A tuple (data, ds) where data is a data tensor of shape
        [new_batch, 1, to_process] and ds contains the original shape.
    """
    ds = list(data.shape)
    if len(ds) == 1:
        # assume time series
        data = data.unsqueeze(0).unsqueeze(0)
    elif len(ds) == 2:
        # assume batched time series
        data = data.unsqueeze(1)
    else:
        data, ds = _fold_axes(data, 1)
        data = data.unsqueeze(1)
    return data, ds


def _postprocess_result_list_dec1d(
    result_list: list[torch.Tensor], ds: list[int], axis: int
) -> list[torch.Tensor]:
    if len(ds) == 1:
        result_list = [r_el.squeeze(0) for r_el in result_list]
    elif len(ds) > 2:
        # Unfold axes for the wavelets
        result_list = [_unfold_axes(fres, ds, 1) for fres in result_list]
    else:
        result_list = result_list

    if axis != -1:
        result_list = [coeff.swapaxes(axis, -1) for coeff in result_list]

    return result_list


def _preprocess_result_list_rec1d(
    result_lst: Sequence[torch.Tensor],
) -> tuple[Sequence[torch.Tensor], list[int]]:
    # Fold axes for the wavelets
    ds = list(result_lst[0].shape)
    fold_coeffs: Sequence[torch.Tensor]
    if len(ds) == 1:
        fold_coeffs = [uf_coeff.unsqueeze(0) for uf_coeff in result_lst]
    elif len(ds) > 2:
        fold_coeffs = [_fold_axes(uf_coeff, 1)[0] for uf_coeff in result_lst]
    else:
        fold_coeffs = result_lst
    return fold_coeffs, ds


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

    Raises:
        ValueError: If the dtype of the input data tensor is unsupported or
            if more than one axis is provided.

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
    if axis != -1:
        if isinstance(axis, int):
            data = data.swapaxes(axis, -1)
        else:
            raise ValueError("wavedec transforms a single axis only.")

    if not _is_dtype_supported(data.dtype):
        raise ValueError(f"Input dtype {data.dtype} not supported")

    data, ds = _preprocess_tensor_dec1d(data)

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

    result_list = _postprocess_result_list_dec1d(result_list, ds, axis)

    return result_list


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

    Raises:
        ValueError: If the dtype of the coeffs tensor is unsupported or if the
            coefficients have incompatible shapes, dtypes or devices or if
            more than one axis is provided.

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
    torch_device = coeffs[0].device
    torch_dtype = coeffs[0].dtype
    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    for coeff in coeffs[1:]:
        if torch_device != coeff.device:
            raise ValueError("coefficients must be on the same device")
        elif torch_dtype != coeff.dtype:
            raise ValueError("coefficients must have the same dtype")

    if axis != -1:
        swap = []
        if isinstance(axis, int):
            for coeff in coeffs:
                swap.append(coeff.swapaxes(axis, -1))
            coeffs = swap
        else:
            raise ValueError("waverec transforms a single axis only.")

    # fold channels, if necessary.
    ds = list(coeffs[0].shape)
    coeffs, ds = _preprocess_result_list_rec1d(coeffs)

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

    if len(ds) == 1:
        res_lo = res_lo.squeeze(0)
    elif len(ds) > 2:
        res_lo = _unfold_axes(res_lo, ds, 1)

    if axis != -1:
        res_lo = res_lo.swapaxes(axis, -1)

    return res_lo
