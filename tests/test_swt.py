from typing import List, Union, Optional

import pywt
import pytest
import torch
import numpy as np

from src.ptwt._util import Wavelet
from src.ptwt.conv_transform import get_filter_tensors
from src.ptwt.conv_transform import _fwt_pad

import matplotlib.pyplot as plt

def swt(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    mode: str = "reflect",
    level: Optional[int] = None,
) -> List[torch.Tensor]:

    if data.dim() == 1:
        # assume time series
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 2:
        # assume batched time series
        data = data.unsqueeze(1)


    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=False, device=data.device, dtype=data.dtype
    )
    filt_len = dec_lo.shape[-1]
    filt = torch.stack([dec_lo, dec_hi], 1)

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for _ in range(level):
        res_lo = _fwt_pad(res_lo, wavelet, mode=mode)
        res = torch.nn.functional.conv_transpose1d(res_lo, filt, stride=1, dilation=1)
        res_lo, res_hi = torch.split(res, 1, 1)
        result_lst.append(res_hi.squeeze(1))
    result_lst.append(res_lo.squeeze(1))
    return result_lst[::-1]



@pytest.mark.parametrize("level", [1, 3, None])
@pytest.mark.parametrize("size", [(1, 32), (1, 31)])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym4"])
def test_swt(level, size, wavelet):
    """Test the 1d swt."""
    signal = np.random.randn(size[0], size[1])
    my_res = swt(torch.from_numpy(signal), wavelet, level=level)
    res = pywt.swt(signal, wavelet, level)
    plt.plot(my_res[0][0][:])
    plt.plot(res[0][0][0])
    plt.show()

    plt.plot(my_res[1][0][:])
    plt.plot(res[0][1][0])
    plt.show()

    pass