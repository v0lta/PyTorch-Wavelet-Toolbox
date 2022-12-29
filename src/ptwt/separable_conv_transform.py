"""Implement separable convolution based transforms.

Under the hood code in this module transforms all dimensions
individually using torch.nn.functional.conv1d and it's
transpose.
"""
from typing import Union

import pywt
import torch
from _util import _as_wavelet


def fswavedec(
    data: torch.Tensor,
    wavelet: Union[str, pywt.Wavelet],
    mode: str = "symmetric",
    levels: int = None,
):

    wavelet = _as_wavelet(wavelet)
    return None
