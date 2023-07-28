"""This module implements stationary wavelet transforms."""

from typing import List, Optional, Union

import pywt
import torch

from src.ptwt._util import Wavelet
from src.ptwt.conv_transform import _get_filter_tensors


def _swt(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    level: Optional[int] = None,
) -> List[torch.Tensor]:
    """Compute a multilevel 1d stationary wavelet transform.

    Args:
        data (torch.Tensor): The input data of shape [batch_size, time].
        wavelet (Union[Wavelet, str]): The wavelet to use.
        level (Optional[int], optional): The number of levels to compute

    Returns:
        List[torch.Tensor]: Same as wavedec.
        Equivalent to pywt.swt with trim_approx=True.
    """
    if data.dim() == 1:
        # assume time series
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 2:
        # assume batched time series
        data = data.unsqueeze(1)

    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    filt_len = dec_lo.shape[-1]
    filt = torch.stack([dec_lo, dec_hi], 0)

    if level is None:
        level = pywt.swt_max_level(data.shape[-1])

    result_lst = []
    res_lo = data
    for current_level in range(level):
        dilation = 2**current_level
        padl, padr = dilation * (filt_len // 2 - 1), dilation * (filt_len // 2)
        res_lo = torch.nn.functional.pad(res_lo, [padl, padr], mode="circular")
        res = torch.nn.functional.conv1d(res_lo, filt, stride=1, dilation=dilation)
        res_lo, res_hi = torch.split(res, 1, 1)
        # Trim_approx == False
        # result_lst.append((res_lo.squeeze(1), res_hi.squeeze(1)))
        result_lst.append(res_hi.squeeze(1))
    result_lst.append(res_lo.squeeze(1))
    return result_lst[::-1]
