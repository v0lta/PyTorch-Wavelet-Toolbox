"""This module implements stationary wavelet transforms."""

from typing import List, Optional, Union

import pywt
import torch

from ._util import Wavelet, _as_wavelet, _unfold_axes
from .conv_transform import (
    _get_filter_tensors,
    _postprocess_result_list_dec1d,
    _preprocess_result_list_rec1d,
    _preprocess_tensor_dec1d,
)


def _swt(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    level: Optional[int] = None,
    axis: Optional[int] = -1,
) -> List[torch.Tensor]:
    """Compute a multilevel 1d stationary wavelet transform.

    Args:
        data (torch.Tensor): The input data of shape [batch_size, time].
        wavelet (Union[Wavelet, str]): The wavelet to use.
        level (Optional[int], optional): The number of levels to compute

    Returns:
        List[torch.Tensor]: Same as wavedec.
        Equivalent to pywt.swt with trim_approx=True.

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
        res_lo = torch.nn.functional.pad(res_lo, [padl, padr], mode="circular")
        res = torch.nn.functional.conv1d(res_lo, filt, stride=1, dilation=dilation)
        res_lo, res_hi = torch.split(res, 1, 1)
        # Trim_approx == False
        # result_list.append((res_lo.squeeze(1), res_hi.squeeze(1)))
        result_list.append(res_hi.squeeze(1))
    result_list.append(res_lo.squeeze(1))

    if ds:
        result_list = _postprocess_result_list_dec1d(result_list, ds)

    if axis != -1:
        result_list = [coeff.swapaxes(axis, -1) for coeff in result_list]

    return result_list[::-1]


def _conv_transpose_dedilate(
    conv_res: torch.Tensor,
    rec_filt: torch.Tensor,
    dilation: int,
    length: int,
) -> torch.Tensor:
    """Undo the forward dilated convolution from the analysis transform.

    Args:
        conv_res (torch.Tensor): The dilated coeffcients
            of shape [batch, 2, length].
        rec_filt (torch.Tensor): The reconstruction filter pair
            of shape [1, 2, filter_length].
        dilation (int): The dilation factor.
        length (int): The signal length.

    Returns:
        torch.Tensor: The deconvolution result.
    """
    to_conv_t_list = [
        conv_res[..., fl : (fl + dilation * rec_filt.shape[-1]) : dilation]
        for fl in range(length)
    ]
    to_conv_t = torch.cat(to_conv_t_list, 0)
    padding = rec_filt.shape[-1] - 1
    rec = torch.nn.functional.conv_transpose1d(
        to_conv_t, rec_filt, stride=1, padding=padding, output_padding=0
    )
    rec = rec / 2.0
    splits = torch.split(rec, rec.shape[0] // len(to_conv_t_list))
    return torch.cat(splits, -1)


def _iswt(
    coeffs: List[torch.Tensor],
    wavelet: Union[pywt.Wavelet, str],
    axis: Optional[int] = -1,
) -> torch.Tensor:
    """Inverts a 1d stationary wavelet transform.

    Args:
        coeffs (List[torch.Tensor]): The coefficients as computed by the swt function.
        wavelet (Union[pywt.Wavelet, str]): The wavelet used for the forward transform.
        axis (int, optional): The axis the forward trasform was computed over.
            Defaults to -1.

    Raises:
        ValueError: If the axis argument is not an integer.

    Returns:
        torch.Tensor: A reconstruction of the original swt input.
    """
    if axis != -1:
        swap = []
        if isinstance(axis, int):
            for coeff in coeffs:
                swap.append(coeff.swapaxes(axis, -1))
            coeffs = swap
        else:
            raise ValueError("iswt transforms a single axis only.")

    ds = None
    length = coeffs[0].shape[-1]
    if coeffs[0].ndim > 2:
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
        res_lo = torch.nn.functional.pad(res_lo, (padl, padr), mode="circular")
        res_lo = _conv_transpose_dedilate(
            res_lo, rec_filt, dilation=dilation, length=length
        )
        res_lo = res_lo.squeeze(1)

    if ds:
        res_lo = _unfold_axes(res_lo, ds, 1)

    if axis != -1:
        res_lo = res_lo.swapaxes(axis, -1)

    return res_lo
