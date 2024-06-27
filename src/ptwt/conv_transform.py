"""Convolutional fast wavelet transformations.

The transformations in this module are based on ``torch.nn.functional.conv1d``
and its transpose. This module treats boundaries with edge-padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import pywt
import torch

from ._util import (
    _adjust_padding_at_reconstruction,
    _as_wavelet,
    _fold_axes,
    _get_filter_tensors,
    _get_len,
    _get_pad,
    _is_dtype_supported,
    _pad_symmetric,
    _translate_boundary_strings,
    _unfold_axes,
)
from .constants import BoundaryMode, Wavelet, WaveletCoeff1d


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

    The transformation relies on convolution operations with the filter pair
    :math:`(\mathbf{h}_A, \mathbf{h}_D)` of the wavelet
    where :math:`A` denotes approximation and :math:`D` detail.
    The coefficients on level :math:`s` are calculated iteratively as

    .. math::
        \mathbf{c}_{k,s} = \mathbf{c}_{A,s - 1} * \mathbf{h}_k
        \quad \text{for $k\in\{A, D\}$}

    with :math:`\mathbf{c}_{A, 0} = \mathbf{x}_0` the original input signal.
    The process uses approximation coefficients as inputs for higher scales.
    Set the `level` argument to choose the largest scale.

    Args:
        data (torch.Tensor): The input time series to transform.
            By default the last axis is transformed.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Please consider the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode: The desired padding mode for extending the signal along the edges.
            See :data:`ptwt.constants.BoundaryMode`. Defaults to "reflect".
        level (int, optional): The maximum decomposition level.
            If None, the level is computed based on the signal shape.
            Defaults to None.
        axis (int): Compute the transform over this axis of the `data` tensor.
            Defaults to -1.


    Returns:
        A list::

            [cA_n, cD_n, cD_n-1, …, cD2, cD1]

        containing the wavelet coefficient tensors where ``n`` denotes
        the level of decomposition. The first entry of the list (``cA_n``)
        is the approximation coefficient tensor.
        The following entries (``cD_n`` - ``cD1``) are the detail coefficient tensors
        of the respective level.

    Raises:
        ValueError: If the dtype of the input data tensor is unsupported or
            if more than one axis is provided.

    Example:
        >>> import ptwt, torch
        >>> # generate an input of even length.
        >>> data = torch.arange(8, dtype=torch.float32)
        >>> # compute the forward fwt coefficients
        >>> ptwt.wavedec(data, 'haar', mode='zero', level=2)
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
    coeffs: WaveletCoeff1d, wavelet: Union[Wavelet, str], axis: int = -1
) -> torch.Tensor:
    """Reconstruct a 1d signal from wavelet coefficients.

    Args:
        coeffs: The wavelet coefficient sequence produced by the forward transform
            :func:`wavedec`. See :data:`ptwt.constants.WaveletCoeff1d`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        axis (int): Compute the transform over this axis of the `data` tensor.
            Defaults to -1.

    Returns:
        The reconstructed signal tensor.

    Raises:
        ValueError: If the dtype of the coeffs tensor is unsupported or if the
            coefficients have incompatible shapes, dtypes or devices or if
            more than one axis is provided.

    Example:
        >>> import ptwt, torch
        >>> # generate an input of even length.
        >>> data = torch.arange(8, dtype=torch.float32)
        >>> # invert the fast wavelet transform.
        >>> coefficients = ptwt.wavedec(data, 'haar', mode='zero', level=2)
        >>> ptwt.waverec(coefficients, "haar")
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
