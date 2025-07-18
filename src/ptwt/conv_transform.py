"""Convolutional fast wavelet transformations.

The transformations in this module are based on ``torch.nn.functional.conv1d``
and its transpose. This module treats boundaries with edge-padding.
"""

from __future__ import annotations

from typing import Optional, Union

import pywt
import torch

from ._util import (
    _adjust_padding_at_reconstruction,
    _check_same_device_dtype,
    _get_filter_tensors,
    _get_len,
    _get_pad,
    _pad_symmetric,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
    _translate_boundary_strings,
)
from .constants import BoundaryMode, Wavelet, WaveletCoeff1d

__all__ = ["wavedec", "waverec"]


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
            See :data:`ptwt.constants.BoundaryMode`. Defaults to ``reflect``.
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

    Example:
        >>> import ptwt, torch
        >>> # generate an input of even length.
        >>> data = torch.arange(8, dtype=torch.float32)
        >>> # compute the forward fwt coefficients
        >>> ptwt.wavedec(data, 'haar', mode='zero', level=2)
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
        Its shape depends on the shape of the input to :func:`ptwt.wavedec`.

    Example:
        >>> import ptwt, torch
        >>> # generate an input of even length.
        >>> data = torch.arange(8, dtype=torch.float32)
        >>> # invert the fast wavelet transform.
        >>> coefficients = ptwt.wavedec(data, 'haar', mode='zero', level=2)
        >>> ptwt.waverec(coefficients, "haar")
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
