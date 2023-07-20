from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pywt
import torch

from src.ptwt._util import Wavelet, _as_wavelet, _pad_symmetric
from src.ptwt.conv_transform import _fwt_pad, get_filter_tensors


def swt(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    level: Optional[int] = None,
) -> List[torch.Tensor]:
    if data.dim() == 1:
        # assume time series
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 2:
        # assume batched time series
        data = data.unsqueeze(1)

    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    filt_len = dec_lo.shape[-1]
    filt = torch.stack([dec_lo, dec_hi], 0)

    if level is None:
        # TODO fixme.
        level = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for l in range(level):
        dilation = l + 1
        padl, padr = dilation * (filt_len // 2 - 1), dilation * (filt_len // 2)
        res_lo = torch.nn.functional.pad(res_lo, [padl, padr], mode="circular")
        # res_lo = torch.nn.functional.pad(res_lo, [padl, padr], mode="constant")
        res = torch.nn.functional.conv1d(res_lo, filt, stride=1, dilation=dilation)
        res_lo, res_hi = torch.split(res, 1, 1)
        # result_lst.append((res_lo.squeeze(1), res_hi.squeeze(1)))
        result_lst.append(res_hi.squeeze(1))
    result_lst.append(res_lo.squeeze(1))
    return result_lst[::-1]


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("size", [12, 32])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym4", "db5", "sym6"])
def test_swt(level, size, wavelet):
    """Test the 1d swt."""
    signal = np.expand_dims(np.arange(size).astype(np.float64), 0)
    ptwt_res = swt(torch.from_numpy(signal), wavelet, level=level)
    pywt_res = pywt.swt(signal, wavelet, level, trim_approx=True, norm=False)

    test_list = []
    for a, b in zip(ptwt_res, pywt_res):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])

    assert all(test_list)
    pass
