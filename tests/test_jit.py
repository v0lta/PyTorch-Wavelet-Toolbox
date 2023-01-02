"""Ensure pytorch's torch.jit.trace feature works properly."""
from typing import NamedTuple

import numpy as np
import pytest
import pywt
import torch

import src.ptwt as ptwt
from tests._mackey_glass import MackeyGenerator


class WaveletTuple(NamedTuple):
    """Replaces namedtuple("Wavelet", ("dec_lo", "dec_hi", "rec_lo", "rec_hi"))."""

    dec_lo: torch.Tensor
    dec_hi: torch.Tensor
    rec_lo: torch.Tensor
    rec_hi: torch.Tensor


def _set_up_wavelet_tuple(wavelet, dtype):
    return WaveletTuple(
        torch.tensor(wavelet.dec_lo).type(dtype),
        torch.tensor(wavelet.dec_hi).type(dtype),
        torch.tensor(wavelet.rec_lo).type(dtype),
        torch.tensor(wavelet.rec_hi).type(dtype),
    )

def jit_wavedec_fun(data, wavelet, level, mode='reflect'):
    return ptwt.wavedec(data, wavelet, mode, level)


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_string", ["db1", "db3", "db4", "sym5"])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("length", [64, 65])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_fwt_jit(wavelet_string, level, length, batch_size, dtype):
    """Test multiple convolution fwt, for various levels and padding options."""
    generator = MackeyGenerator(
        batch_size=batch_size, tmax=length, delta_t=1, device="cpu"
    )

    mackey_data_1 = torch.squeeze(generator(), -1).type(dtype)
    wavelet = pywt.Wavelet(wavelet_string)
    wavelet = _set_up_wavelet_tuple(wavelet, dtype)

    with pytest.warns(None):
        jit_wavedec = torch.jit.trace(
            jit_wavedec_fun, (mackey_data_1, wavelet, torch.tensor(level)), strict=False
        )
        ptcoeff = jit_wavedec(mackey_data_1, wavelet, level=torch.tensor(level))
        jit_waverec = torch.jit.trace(ptwt.waverec, (ptcoeff, wavelet))
        res = jit_waverec(ptcoeff, wavelet)
    assert np.allclose(mackey_data_1.numpy(), res.numpy()[:, : mackey_data_1.shape[-1]])