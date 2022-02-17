"""Test the continuous transformation code."""

import numpy as np
import pytest
import pywt
import torch
from scipy import signal
from src.ptwt.continuous_transform import cwt

continuous_wavelets = [
    "cgau1",
    "cgau2",
    "cgau3",
    "cgau4",
    "cgau5",
    "cgau6",
    "cgau7",
    "cgau8",
    "gaus1",
    "gaus2",
    "gaus3",
    "gaus4",
    "gaus5",
    "gaus6",
    "gaus7",
    "gaus8",
    "mexh",
    "morl",
]

@pytest.mark.slow
@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt(wavelet, cuda):
    """Test the cwt implementation for various wavelets and with a GPU."""
    t = np.linspace(-2, 2, 800, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
    widths = np.arange(1, 31)
    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet)
    sig = torch.from_numpy(sig)
    if cuda:
        if torch.cuda.is_available():
            sig = sig.cuda()
    cwtmatr_pt, freqs_pt = cwt(sig, widths, wavelet)
    if cuda:
        cwtmatr_pt = cwtmatr_pt.cpu()
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt_batched(wavelet):
    """Test batched transforms."""
    sig = np.random.randn(10, 200)
    widths = np.arange(1, 30)
    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet)
    sig = torch.from_numpy(sig)
    cwtmatr_pt, freqs_pt = cwt(sig, widths, wavelet)
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)
