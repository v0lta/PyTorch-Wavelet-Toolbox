"""Code for three dimensional transforms."""

from typing import Dict, List, Optional, Union

import pywt
import torch

from ._util import Wavelet, _as_wavelet
from .conv_transform import (
    _get_pad,
    _outer,
    _translate_boundary_strings,
    get_filter_tensors,
)


def _construct_3d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct three dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 3d filters of dimension
            [8, 1, length, height, width].
            The four filters are ordered ll, lh, hl, hh.

    """
    dim_size = lo.shape[-1]
    size = [dim_size] * 3
    lll = _outer(lo, _outer(lo, lo)).reshape(size)
    llh = _outer(lo, _outer(lo, hi)).reshape(size)
    lhl = _outer(lo, _outer(hi, lo)).reshape(size)
    lhh = _outer(lo, _outer(hi, hi)).reshape(size)
    hll = _outer(hi, _outer(lo, lo)).reshape(size)
    hlh = _outer(hi, _outer(lo, hi)).reshape(size)
    hhl = _outer(hi, _outer(hi, lo)).reshape(size)
    hhh = _outer(hi, _outer(hi, hi)).reshape(size)
    filt = torch.stack([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], 0)
    filt = filt.unsqueeze(1)
    return filt


def _fwt_pad3(
    data: torch.Tensor, wavelet: Union[Wavelet, str], mode: str
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode. Supported modes are "zero".

    Returns:
        The padded output tensor.

    """
    mode = _translate_boundary_strings(mode)

    wavelet = _as_wavelet(wavelet)
    pad_back, pad_front = _get_pad(data.shape[-3], len(wavelet.dec_lo))
    pad_bottom, pad_top = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    pad_right, pad_left = _get_pad(data.shape[-1], len(wavelet.dec_lo))
    data_pad = torch.nn.functional.pad(
        data, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back], mode=mode
    )
    return data_pad


def wavedec3(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    level: Optional[int] = None,
    mode: str = "zero",
) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute a three dimensional wavelet transform.

    Args:
        data (torch.Tensor): The input data of shape
            [batch_size, length, height, width]
        wavelet (Union[Wavelet, str]): The wavelet to be used.
        level (Optional[int]): The maximum decomposition level.
            Defaults to None.
        mode (str): The padding mode. Defaults to "zero".

    Returns:
        list: A list with the lll coefficients and dicts with filter
            order strings as keys.

    Raises:
        ValueError: If the input has fewer than 3 dimensions.

    """
    if data.dim() < 3:
        raise ValueError("Three dimensional inputs required for 3d wavedec.")
    elif data.dim() == 3:
        data = data.unsqueeze(0)

    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_3d_filt(lo=dec_lo, hi=dec_hi)

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2], data.shape[-3]], wavelet
        )

    result_lst: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []
    res_lll = data
    for _ in range(level):
        res_lll = _fwt_pad3(res_lll, wavelet, mode=mode)
        res = torch.nn.functional.conv3d(res_lll.unsqueeze(1), dec_filt, stride=2)
        res_lll, res_llh, res_lhl, res_lhh, res_hll, res_hlh, res_hhl, res_hhh = [
            sr.squeeze(1) for sr in torch.split(res, 1, 1)
        ]
        result_lst.append(
            {
                "aad": res_llh,
                "ada": res_lhl,
                "add": res_lhh,
                "daa": res_hll,
                "dad": res_hlh,
                "dda": res_hhl,
                "ddd": res_hhh,
            }
        )
    result_lst.append(res_lll)
    return result_lst[::-1]
